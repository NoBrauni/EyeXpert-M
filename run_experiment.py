"""
COMPLETE EXPERIMENT RUNNER
Runs ablations + evaluation and saves results to CSV for easy transport.

Usage:
  python run_experiment.py

This will:
1. Run all ablations (base, base_plus, hard_5, moe_5)
2. Evaluate all models on test set
3. Save comprehensive results to experiment_results.csv

The CSV can be easily shared and analyzed.

SCIENTIFIC DESIGN:
- base: Minimal non-modular baseline (4.7M params)
- base_plus: Parameter-matched non-modular (18.9M params)
- hard_5: Language-family hard routing (18.9M params, 5 experts = 1 per family)
- moe_5: Learned soft routing (18.9M params, 5 experts = 1 per family)

Note: 5 experts ensures 1 expert per language family. Using <5 experts would create
collisions (e.g., family 3→expert 0, family 4→expert 1), undermining the modularity hypothesis.
"""

import json
import random
import time
import csv
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM, LANG_FAMILY
from train_meco import MECO, collate


# =========================================================
# CONFIGURATION
# =========================================================
FAST_MODE = True  # Set to True for quick testing (1 epoch, 0.5% data)
EPOCHS = 1 if FAST_MODE else 3
SUBSAMPLE_FRACTION = 0.005 if FAST_MODE else 0.2
SKIP_INFERENCE_TIME = FAST_MODE
EARLY_STOPPING_PATIENCE = 2 if not FAST_MODE else 1

# Hyperparameters
LR = 1e-4
BATCH_SIZE = 4  # Optimized for T4
ACCUM_STEPS = 2
LAMBDA_DUR = 0.01
MOE_LAMBDA = 0.01

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# Experiments to run
# NOTE: Using 5 experts ensures one expert per language family
# Hard routing with <5 experts creates collisions (e.g., family 3→expert 0, family 4→expert 1)
# This would undermine the core hypothesis of language-family-based modularity
ABLATIONS = {
    "base": ("base", 5),
    "base_plus": ("base+", 5),      # Parameter-matched non-modular baseline
    "hard_5": ("hard", 5),          # Hard routing: 1 expert per language family
    "moe_5": ("moe", 5),            # MoE routing: learned expert selection
}


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subsample(data, fraction=0.1, seed=42):
    set_seed(seed)
    buckets = defaultdict(list)

    for x in data:
        buckets[x["lang"]].append(x)

    out = []
    for _, items in buckets.items():
        random.shuffle(items)
        n = max(1, int(len(items) * fraction))
        out.extend(items[:n])

    random.shuffle(out)
    return out


def split_data(data):
    random.shuffle(data)
    n = len(data)
    return (
        data[:int(0.7 * n)],
        data[int(0.7 * n):int(0.85 * n)],
        data[int(0.85 * n):]
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, loader, device, num_batches=5):
    if SKIP_INFERENCE_TIME:
        return 0.0

    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            start = time.time()
            _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"],
            )
            times.append(time.time() - start)

    return sum(times) / len(times) if times else 0.0


# =========================================================
# TRAINING
# =========================================================
def loss_fn(logits, dur_pred, moe_loss, batch):
    B, T, V = logits.shape
    mask = batch["scanpath_mask"].to(DEVICE)

    # Fixation loss
    fix = F.cross_entropy(
        logits.reshape(B * T, V),
        batch["scanpath"].reshape(B * T).to(DEVICE),
        reduction="none"
    ).reshape(B, T)
    fix = (fix * mask).sum() / mask.sum()

    # Duration loss
    target_dur = torch.log1p(batch["durations"]).to(DEVICE)
    dur = F.mse_loss(dur_pred.squeeze(-1), target_dur, reduction="none")
    dur = (dur * mask).sum() / mask.sum()

    return fix + LAMBDA_DUR * dur + MOE_LAMBDA * moe_loss


def forward(model, batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    out = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["word_ids"],
        batch["scanpath"],
        batch["durations"],
        batch["family"],
        batch["lang_id"],
    )

    if len(out) == 3:
        logits, dur, moe_metrics = out
        # Extract scalar loss from moe_metrics dict
        moe_loss = moe_metrics.get("load_balance", 0.0) if isinstance(moe_metrics, dict) else 0.0
    else:
        logits, dur = out
        moe_loss = 0.0

    return logits, dur, moe_loss


def train(model, train_loader, val_data, name, epochs=None):
    if epochs is None:
        epochs = EPOCHS

    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best = float("inf")
    patience_counter = 0
    results = {"best_val_loss": float("inf"), "best_epoch": -1}

    # Create validation loader
    val_loader = DataLoader(
        MECO(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,  # Use 0 for validation to avoid issues
        pin_memory=False if DEVICE.type == "cpu" else True
    )

    for ep in range(epochs):
        model.train()
        total = 0
        optim.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            logits, dur, moe = forward(model, batch)

            loss = loss_fn(logits, dur, moe, batch)
            loss = loss / ACCUM_STEPS

            if USE_AMP:
                with torch.autocast("cuda"):
                    scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                if USE_AMP:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

            total += loss.item() * ACCUM_STEPS

        val_loss = evaluate(model, val_loader)

        print(f"[{name}] Epoch {ep}: Train={total/len(train_loader):.4f}, Val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            patience_counter = 0
            results["best_val_loss"] = val_loss
            results["best_epoch"] = ep
            torch.save(model.state_dict(), f"{name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE and not FAST_MODE:
                print(f"Early stopping at epoch {ep}")
                break

    return results


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader):
    model.eval()
    total, count = 0, 0

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = forward(model, batch)
            B, T, V = logits.shape
            mask = batch["scanpath_mask"].to(DEVICE)

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                batch["scanpath"].reshape(B * T).to(DEVICE),
                reduction="none"
            ).reshape(B, T)

            total += (loss * mask).sum().item()
            count += mask.sum().item()

    return total / max(count, 1.0)


def evaluate_detailed(model, loader):
    """Detailed evaluation with all metrics."""
    model.eval()

    totals = None
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            logits, dur_pred, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"]
            )

            pred = logits.argmax(dim=-1)
            metrics = evaluate_batch_detailed(pred, batch["scanpath"], batch["scanpath_mask"], dur_pred, batch["durations"])

            if totals is None:
                totals = {k: 0.0 for k in metrics if k != "tokens"}

            tokens = metrics["tokens"]
            total_tokens += tokens

            for k in totals:
                totals[k] += metrics[k] * tokens

    for k in totals:
        totals[k] /= total_tokens

    return totals


def evaluate_batch_detailed(pred, gold, mask, pred_dur, gold_dur):
    B = pred.size(0)

    acc = fixation_accuracy(pred, gold, mask)

    lev_total = 0
    reg_p = 0
    reg_g = 0
    exact_match = 0

    for i in range(B):
        L = int(mask[i].sum().item())

        p = pred[i][:L].tolist()
        g = gold[i][:L].tolist()

        lev_total += normalized_levenshtein(p, g)
        reg_p += regression_rate(p)
        reg_g += regression_rate(g)
        
        if p == g:
            exact_match += 1

    lev = lev_total / B
    reg_p /= B
    reg_g /= B
    exact_match_rate = exact_match / B

    # Duration metrics
    pred_d = pred_dur.squeeze(-1)
    gold_d = torch.log1p(gold_dur)
    dur_mse = ((pred_d - gold_d) ** 2 * mask).sum() / mask.sum()
    dur_mae = (torch.abs(pred_d - gold_d) * mask).sum() / mask.sum()

    return {
        "accuracy": acc.item(),
        "levenshtein": lev,
        "regression_pred": reg_p,
        "regression_true": reg_g,
        "exact_match": exact_match_rate,
        "dur_mse": dur_mse.item(),
        "dur_mae": dur_mae.item(),
        "tokens": mask.sum().item()
    }


def evaluate_per_language(model, loader):
    model.eval()
    family_stats = defaultdict(lambda: {"loss": 0.0, "count": 0})

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            logits, _, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"]
            )

            pred = logits.argmax(dim=-1)
            B, T, V = logits.shape
            mask = batch["scanpath_mask"].to(DEVICE)

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                batch["scanpath"].reshape(B * T).to(DEVICE),
                reduction="none"
            ).reshape(B, T)

            masked_loss = (loss * mask).sum(dim=1)
            masked_count = mask.sum(dim=1)

            families = batch["family"].cpu().numpy()
            for b in range(B):
                family_id = int(families[b])
                family_stats[family_id]["loss"] += masked_loss[b].item()
                family_stats[family_id]["count"] += masked_count[b].item()

    family_results = {}
    for family_id, stats in family_stats.items():
        if stats["count"] > 0:
            family_results[family_id] = stats["loss"] / stats["count"]

    return family_results


def track_expert_usage(model, loader):
    if not hasattr(model.decoder, "last_probs"):
        return None

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
            _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"]
            )
            if model.decoder.last_probs is not None:
                # Flatten batch and sequence dimensions: [B, T, n_experts] -> [B*T, n_experts]
                batch_probs = model.decoder.last_probs.view(-1, model.decoder.last_probs.size(-1))
                all_probs.append(batch_probs)

    if not all_probs:
        return None

    # Concatenate all flattened probabilities: [total_samples, n_experts]
    all_probs = torch.cat(all_probs, dim=0)
    mean_probs = all_probs.mean(dim=0)

    return {
        "mean_probs": mean_probs.cpu().numpy(),
        "load_balance": (mean_probs ** 2).sum().item(),
        "entropy": -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item(),
    }


# =========================================================
# METRICS
# =========================================================
def fixation_accuracy(pred, gold, mask):
    correct = (pred == gold).float()
    return (correct * mask).sum() / mask.sum()


def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]


def normalized_levenshtein(pred, gold):
    return levenshtein(pred, gold) / max(len(gold), 1)


def regression_rate(seq):
    if len(seq) < 2:
        return 0.0
    jumps = torch.diff(torch.tensor(seq))
    return (jumps < 0).float().mean().item()


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def main():
    print("="*80)
    print("EYEEXPERT-M COMPLETE EXPERIMENT")
    print("="*80)

    if FAST_MODE:
        print("⚡ FAST MODE: Quick testing")
        print(f"   Epochs: {EPOCHS}, Data: {SUBSAMPLE_FRACTION*100:.1f}%")
    else:
        print("🚀 FULL MODE: Complete experiment")
        print(f"   Epochs: {EPOCHS}, Data: {SUBSAMPLE_FRACTION*100:.1f}%")

    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print("="*80)

    # Load and prepare data
    print("\n📊 Loading data...")
    train_val_data = json.load(open("meco_train.json"))
    train_val_data = stratified_subsample(train_val_data, fraction=SUBSAMPLE_FRACTION)
    train_data, val_data, _ = split_data(train_val_data)  # Split into train/val, ignore test split from train data
    
    test_data = json.load(open("meco_test.json"))
    test_data = stratified_subsample(test_data, fraction=SUBSAMPLE_FRACTION)  # Subsample test for consistency in FAST_MODE

    train_loader = DataLoader(
        MECO(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Run ablations
    print("\n🏃 Running ablations...")
    all_results = []

    for name, (mode, n_experts) in ABLATIONS.items():
        print(f"\n🔬 Training {name} (mode={mode}, experts={n_experts})")

        model = EyeXpertM(mode=mode, n_experts=n_experts)
        total_params, trainable_params = count_parameters(model)
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")

        train_results = train(model, train_loader, val_data, name)
        print(f"   Best validation loss: {train_results['best_val_loss']:.4f} (epoch {train_results['best_epoch']})")

        # Load best model and evaluate
        model.load_state_dict(torch.load(f"{name}.pt"))
        model.to(DEVICE)

        # Test evaluation
        test_loader = DataLoader(MECO(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        test_loss = evaluate(model, test_loader)

        # Detailed metrics
        detailed_metrics = evaluate_detailed(model, test_loader)

        # Per-language performance
        family_performance = evaluate_per_language(model, test_loader)

        # Expert usage (if applicable)
        expert_usage = track_expert_usage(model, test_loader)

        # Inference time
        inference_time = measure_inference_time(model, test_loader, DEVICE)

        result = {
            "model": name,
            "mode": mode,
            "n_experts": n_experts,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_val_loss": train_results["best_val_loss"],
            "best_epoch": train_results["best_epoch"],
            "test_loss": test_loss,
            "accuracy": detailed_metrics["accuracy"],
            "levenshtein": detailed_metrics["levenshtein"],
            "regression_pred": detailed_metrics["regression_pred"],
            "regression_true": detailed_metrics["regression_true"],
            "exact_match": detailed_metrics["exact_match"],
            "dur_mse": detailed_metrics["dur_mse"],
            "dur_mae": detailed_metrics["dur_mae"],
            "inference_time_ms": inference_time * 1000,
        }

        # Add family performance
        for family_id in range(5):
            family_name = {0: "germanic", 1: "romance", 2: "north_germanic", 3: "slavic", 4: "finno_ugric"}.get(family_id, f"family_{family_id}")
            result[f"{family_name}_loss"] = family_performance.get(family_id, float("nan"))

        # Add expert usage if available
        if expert_usage:
            result["load_balance"] = expert_usage["load_balance"]
            result["router_entropy"] = expert_usage["entropy"]
            for i, prob in enumerate(expert_usage["mean_probs"]):
                result[f"expert_{i}_prob"] = prob

        all_results.append(result)

    # Save results to CSV
    print("\n💾 Saving results to CSV...")

    fieldnames = [
        "model", "mode", "n_experts", "total_params", "trainable_params",
        "best_val_loss", "best_epoch", "test_loss", "accuracy",
        "levenshtein", "regression_pred", "regression_true", "exact_match",
        "dur_mse", "dur_mae", "inference_time_ms",
        "germanic_loss", "romance_loss", "north_germanic_loss", "slavic_loss", "finno_ugric_loss",
        "load_balance", "router_entropy"
    ]

    # Add expert probability columns
    for i in range(5):
        fieldnames.append(f"expert_{i}_prob")

    with open("experiment_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print("   ✓ Results saved to experiment_results.csv")

    # Print summary
    print("\n📈 EXPERIMENT SUMMARY")
    print("="*80)

    base_result = next(r for r in all_results if r["mode"] == "base")
    base_plus_result = next(r for r in all_results if r["mode"] == "base+")
    hard_5_result = next(r for r in all_results if r["mode"] == "hard" and r["n_experts"] == 5)
    moe_5_result = next(r for r in all_results if r["mode"] == "moe" and r["n_experts"] == 5)

    print("Test Losses:")
    print(f"  Base: {base_result['test_loss']:.4f}")
    print(f"  Base+: {base_plus_result['test_loss']:.4f}")
    print(f"  Hard-5: {hard_5_result['test_loss']:.4f}")
    print(f"  MoE-5: {moe_5_result['test_loss']:.4f}")

    print("Modularity Effect (vs parameter-matched baseline):")
    hard_improvement = (1 - hard_5_result["test_loss"] / base_plus_result["test_loss"]) * 100
    moe_improvement = (1 - moe_5_result["test_loss"] / base_plus_result["test_loss"]) * 100
    print(f"  Hard routing: {hard_improvement:.2f}%")
    print(f"  MoE routing: {moe_improvement:.2f}%")

    print("Routing Strategy Comparison:")
    routing_diff = (moe_5_result["test_loss"] - hard_5_result["test_loss"]) / hard_5_result["test_loss"] * 100
    print(f"  MoE vs Hard: {routing_diff:.2f}%")

    best_model = min([base_result, hard_5_result, moe_5_result], key=lambda x: x["test_loss"])
    print(f"\n Best Model: {best_model['model']} (Test Loss: {best_model['test_loss']:.4f})")

    print("\n Experiment complete! Share experiment_results.csv for analysis.")
    print("="*80)


if __name__ == "__main__":
    main()
