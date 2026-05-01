import json
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM, LANG_FAMILY
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

USE_AMP = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# =========================================================
# HYPERPARAMETERS
# =========================================================
LR = 1e-4
BATCH_SIZE = 4  # Increased from 2 (T4 can handle this)
ACCUM_STEPS = 2  # Reduced from 4 (effective batch 8, same as before)

LAMBDA_DUR = 0.01
MOE_LAMBDA = 0.01

# =========================================================
# OPTIMIZATION FLAGS
# =========================================================
FAST_MODE = True  # Set to False for full results
EPOCHS = 1 if FAST_MODE else 3  # 1 epoch for quick test, 3 for full
SUBSAMPLE_FRACTION = 0.005 if FAST_MODE else 0.01  # 0.5% for fast, 1% for full
SKIP_INFERENCE_TIME = FAST_MODE  # Skip slow inference timing in fast mode
EARLY_STOPPING_PATIENCE = 2 if not FAST_MODE else 1


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA SPLIT
# =========================================================
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


# =========================================================
# LOSS (CLEAN + CORRECT)
# =========================================================
def loss_fn(logits, dur_pred, moe_loss, batch):

    B, T, V = logits.shape
    mask = batch["scanpath_mask"].to(logits.device)

    # fixation loss
    fix = F.cross_entropy(
        logits.reshape(B * T, V),
        batch["scanpath"].reshape(B * T).to(logits.device),
        reduction="none"
    ).reshape(B, T)

    fix = (fix * mask).sum() / mask.sum()

    # duration loss
    target_dur = torch.log1p(batch["durations"]).to(logits.device)

    dur = F.mse_loss(dur_pred.squeeze(-1), target_dur, reduction="none")
    dur = (dur * mask).sum() / mask.sum()

    return fix + LAMBDA_DUR * dur + MOE_LAMBDA * moe_loss


# =========================================================
# FORWARD
# =========================================================
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

    # unify outputs
    if len(out) == 3:
        logits, dur, moe = out
    else:
        logits, dur = out
        moe = 0.0

    return logits, dur, moe


# =========================================================
# TRAIN
# =========================================================
def train(model, train_loader, val_loader, name, epochs=None):

    if epochs is None:
        epochs = EPOCHS
    
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best = float("inf")
    patience_counter = 0
    results = {"best_val_loss": float("inf"), "best_epoch": -1}

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

        print(f"\n[{name}] Epoch {ep}")
        print("Train:", total / len(train_loader))
        print("Val  :", val_loss)

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
# COMPUTATIONAL METRICS
# =========================================================
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, loader, device, num_batches=5):
    """Measure average inference time per batch."""
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
# PER-LANGUAGE EVALUATION
# =========================================================
def evaluate_per_language(model, loader, lang_to_family):
    """Evaluate loss per language family and individual language."""
    model.eval()
    lang_stats = defaultdict(lambda: {"loss": 0.0, "count": 0})
    family_stats = defaultdict(lambda: {"loss": 0.0, "count": 0})
    total, count = 0, 0

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = forward(model, batch)

            B, T, V = logits.shape
            mask = batch["scanpath_mask"].to(logits.device)

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                batch["scanpath"].reshape(B * T).to(logits.device),
                reduction="none"
            ).reshape(B, T)

            masked_loss = (loss * mask).sum(dim=1)  # per sample
            masked_count = mask.sum(dim=1)

            # Accumulate global metrics
            total += masked_loss.sum().item()
            count += masked_count.sum().item()

            # Per-language metrics (would need language info in batch)
            # For now, track by family
            families = batch["family"].cpu().numpy()
            for b in range(B):
                family_id = int(families[b])
                family_stats[family_id]["loss"] += masked_loss[b].item()
                family_stats[family_id]["count"] += masked_count[b].item()

    global_loss = total / max(count, 1.0)
    
    # Normalize family stats
    family_results = {}
    for family_id, stats in family_stats.items():
        if stats["count"] > 0:
            family_results[family_id] = stats["loss"] / stats["count"]
    
    return global_loss, family_results


# =========================================================
# EXPERT USAGE TRACKING
# =========================================================
def track_expert_usage(model, loader, device):
    """Track which experts are used and their load distribution."""
    if not hasattr(model.decoder, "last_probs"):
        return None  # Not an MoE model
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            _ = forward(model, batch)
            if model.decoder.last_probs is not None:
                all_probs.append(model.decoder.last_probs.cpu())
    
    if not all_probs:
        return None
    
    # Concatenate all probabilities
    all_probs = torch.cat(all_probs, dim=0)  # [B*T_total, n_experts]
    mean_probs = all_probs.mean(dim=0)  # Average across all timesteps/samples
    
    return {
        "mean_probs": mean_probs.numpy(),
        "load_balance": (mean_probs ** 2).sum().item(),
        "entropy": -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item(),
    }


# =========================================================
# EVAL
# =========================================================
def evaluate(model, loader):

    model.eval()
    total, count = 0, 0

    with torch.no_grad():
        for batch in loader:

            logits, _, _ = forward(model, batch)

            B, T, V = logits.shape
            mask = batch["scanpath_mask"].to(logits.device)

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                batch["scanpath"].reshape(B * T).to(logits.device),
                reduction="none"
            ).reshape(B, T)

            total += (loss * mask).sum().item()
            count += mask.sum().item()

    return total / max(count, 1.0)


# =========================================================
# ABLATIONS
# =========================================================
ABLATIONS = {
    "base": ("base", 5),
    "base_plus": ("base+", 5),
    "hard_5": ("hard", 5),
    "moe_5": ("moe", 5),
    "hard_3": ("hard", 3),
    "moe_3": ("moe", 3),
}

LANG_FAMILY_NAMES = {
    0: "Germanic",
    1: "Romance",
    2: "North Germanic",
    3: "Slavic",
    4: "Finno-Ugric",
}


def run(name, mode, n_experts, train_loader, val_data, test_data):
    """Run a single ablation and collect comprehensive metrics."""

    print("\n" + "="*60)
    print(f"{name.upper()} | mode={mode}, experts={n_experts}")
    print("="*60)

    model = EyeXpertM(mode=mode, n_experts=n_experts)
    
    # Compute parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Create val_loader for training
    val_loader = DataLoader(
        MECO(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )
    
    # Train
    train_results = train(model, train_loader, val_loader, name, epochs=3)
    
    # Load best model
    model.load_state_dict(torch.load(f"{name}.pt"))
    model.to(DEVICE)
    
    # Evaluate on test data
    test_loader = DataLoader(
        MECO(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )
    
    test_loss = evaluate(model, test_loader)
    
    # Per-language evaluation
    _, family_performance = evaluate_per_language(model, test_loader, LANG_FAMILY)
    
    # Expert usage (if MoE/hard)
    expert_usage = None
    if mode in ["moe", "hard"]:
        expert_usage = track_expert_usage(model, test_loader, DEVICE)
    
    # Inference time
    inference_time = measure_inference_time(model, test_loader, DEVICE, num_batches=5)
    
    results = {
        "name": name,
        "mode": mode,
        "n_experts": n_experts,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_loss": train_results.get("best_val_loss", float("inf")),
        "best_epoch": train_results.get("best_epoch", -1),
        "test_loss": test_loss,
        "family_performance": family_performance,
        "expert_usage": expert_usage,
        "inference_time_ms": inference_time * 1000,
    }
    
    # Print summary
    print(f"\nResults:")
    print(f"  Best Val Loss: {results['best_val_loss']:.4f} (epoch {results['best_epoch']})")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Inference Time: {results['inference_time_ms']:.2f} ms/batch")
    
    if family_performance:
        print(f"\n  Per-Language Family Performance:")
        for family_id, loss in sorted(family_performance.items()):
            family_name = LANG_FAMILY_NAMES.get(family_id, f"Family {family_id}")
            print(f"    {family_name}: {loss:.4f}")
    
    if expert_usage:
        print(f"\n  Expert Usage:")
        for i, prob in enumerate(expert_usage["mean_probs"]):
            print(f"    Expert {i}: {prob:.4f}")
        print(f"  Load Balance Score: {expert_usage['load_balance']:.4f}")
        print(f"  Router Entropy: {expert_usage['entropy']:.4f}")
    
    return results


# =========================================================
# ANALYSIS & REPORTING
# =========================================================
def print_comprehensive_report(all_results):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ABLATION ANALYSIS")
    print("="*80)
    
    # Summary table
    print("\n1. OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"{'Model':<15} {'Mode':<8} {'Experts':<8} {'Val Loss':<12} {'Test Loss':<12} {'Params (M)':<12}")
    print("-" * 80)
    for res in all_results:
        params_m = res["total_params"] / 1e6
        print(f"{res['name']:<15} {res['mode']:<8} {res['n_experts']:<8} "
              f"{res['best_val_loss']:<12.4f} {res['test_loss']:<12.4f} {params_m:<12.2f}")
    
    # Efficiency metrics
    print("\n2. COMPUTATIONAL EFFICIENCY")
    print("-" * 80)
    print(f"{'Model':<15} {'Params (M)':<15} {'Inference (ms)':<20}")
    print("-" * 80)
    for res in all_results:
        params_m = res["total_params"] / 1e6
        inf_time = res["inference_time_ms"]
        print(f"{res['name']:<15} {params_m:<15.2f} {inf_time:<20.2f}")
    
    # Per-language analysis
    print("\n3. PER-LANGUAGE FAMILY ADAPTABILITY")
    print("-" * 80)
    if all_results[0].get("family_performance"):
        for family_id in sorted(all_results[0]["family_performance"].keys()):
            family_name = LANG_FAMILY_NAMES.get(family_id, f"Family {family_id}")
            print(f"\n{family_name}:")
            print(f"  {'Model':<15} {'Loss':<10}")
            for res in all_results:
                if family_id in res.get("family_performance", {}):
                    loss = res["family_performance"][family_id]
                    print(f"  {res['name']:<15} {loss:<10.4f}")
    
    # Expert specialization (MoE models)
    print("\n4. EXPERT SPECIALIZATION & LOAD BALANCING")
    print("-" * 80)
    for res in all_results:
        if res.get("expert_usage"):
            print(f"\n{res['name']} ({res['mode']}):")
            expert_usage = res["expert_usage"]
            print(f"  Load Balance Score: {expert_usage['load_balance']:.4f}")
            print(f"  Router Entropy: {expert_usage['entropy']:.4f}")
            print(f"  Expert Distribution: {[f'{p:.3f}' for p in expert_usage['mean_probs']]}")
    
    # Research question summary
    print("\n5. RESEARCH QUESTION: MODULARITY & SCALABILITY")
    print("-" * 80)
    
    base_results = [r for r in all_results if r["mode"] == "base"][0]
    base_plus_results = [r for r in all_results if r["name"] == "base_plus"]
    hard_results = [r for r in all_results if r["mode"] == "hard" and r["n_experts"] == 5]
    moe_results = [r for r in all_results if r["mode"] == "moe" and r["n_experts"] == 5]
    
    print(f"\n  Model Legend:")
    print(f"    - base: Minimal non-modular decoder (4.7M trainable params)")
    print(f"    - base+: Extra layers but NO modularity (18.9M params, matches hard/moe)")
    print(f"    - hard_5: Language-family-based routing (18.9M params)")
    print(f"    - moe_5: Learned soft routing (18.9M params)")
    
    print(f"\n  Test Losses:")
    print(f"    base: {base_results['test_loss']:.4f}")
    if base_plus_results:
        base_plus_res = base_plus_results[0]
        print(f"    base+ (param-matched): {base_plus_res['test_loss']:.4f}")
    
    if hard_results and moe_results:
        hard_res = hard_results[0]
        moe_res = moe_results[0]
        
        print(f"    hard_5: {hard_res['test_loss']:.4f}")
        print(f"    moe_5: {moe_res['test_loss']:.4f}")
        
        # Comparisons
        if base_plus_results:
            print(f"\n  Key Comparisons:")
            print(f"    Modularity effect (base+ vs hard_5): {hard_res['test_loss'] - base_plus_res['test_loss']:+.4f}")
            print(f"    Modularity effect (base+ vs moe_5): {moe_res['test_loss'] - base_plus_res['test_loss']:+.4f}")
            print(f"    Routing strategy (hard_5 vs moe_5): {moe_res['test_loss'] - hard_res['test_loss']:+.4f}")
        
        hard_improvement = (1 - hard_res['test_loss'] / base_results['test_loss']) * 100
        moe_improvement = (1 - moe_res['test_loss'] / base_results['test_loss']) * 100
        
        print(f"\n  Improvement vs minimal baseline:")
        print(f"    Hard Routing: {hard_improvement:+.2f}%")
        print(f"    MoE: {moe_improvement:+.2f}%")
        
        best_model = min(hard_res, moe_res, key=lambda x: x['test_loss'])
        print(f"\n  Best Model: {best_model['name']} ({best_model['mode']})")
        
        # Expert count sensitivity
        hard_3 = [r for r in all_results if r["mode"] == "hard" and r["n_experts"] == 3]
        hard_5 = [r for r in all_results if r["mode"] == "hard" and r["n_experts"] == 5]
        if hard_3 and hard_5:
            print(f"\n  Expert Count Sensitivity (Hard Routing):")
            print(f"    3 Experts: {hard_3[0]['test_loss']:.4f}")
            print(f"    5 Experts: {hard_5[0]['test_loss']:.4f}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("="*80)
    if FAST_MODE:
        print("FAST MODE: Quick test with minimal data")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Subsample: {SUBSAMPLE_FRACTION*100:.1f}%")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Skip inference timing: {SKIP_INFERENCE_TIME}")
    else:
        print("FULL MODE: Complete ablation study")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Subsample: {SUBSAMPLE_FRACTION*100:.1f}%")
    print("="*80)

    data = json.load(open("meco_train.json"))

    data = stratified_subsample(data, fraction=SUBSAMPLE_FRACTION)
    train_data, val_data, test_data = split_data(data)

    train_loader = DataLoader(
        MECO(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )

    all_results = []
    for name, (mode, n_experts) in ABLATIONS.items():
        result = run(name, mode, n_experts, train_loader, val_data, test_data)
        all_results.append(result)
    
    # Print comprehensive report
    print_comprehensive_report(all_results)
