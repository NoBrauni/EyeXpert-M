import json
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from collections import defaultdict

import torch.nn.functional as F

from model import EyeXpertM
from train_meco import MECO, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ABLATIONS = {
    "base": ("base", 5),
    "base_plus": ("base+", 5),
    "hard_5": ("hard", 5),
    "moe_5": ("moe", 5),
}


# -----------------------------
# PER-SAMPLE METRICS
# -----------------------------
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
            metrics = evaluate_batch_detailed(pred, batch["scanpath"], batch["scanpath_mask"], dur_pred,
                                              batch["durations"])

            if totals is None:
                totals = {k: 0.0 for k in metrics if k != "tokens"}

            tokens = metrics["tokens"]
            total_tokens += tokens

            for k in totals:
                totals[k] += metrics[k] * tokens

    for k in totals:
        totals[k] /= total_tokens

    return totals


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


# -----------------------------
# EVALUATE MODEL
# -----------------------------
def evaluate_model(model, loader):
    model.eval()
    all_results = []

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

            results = evaluate_batch_detailed(
                pred,
                batch["scanpath"],
                batch["scanpath_mask"],
                dur_pred,
                batch["durations"]
            )

            all_results.append(results)

    return all_results


# -----------------------------
# STATS
# -----------------------------
def compute_stats(values):
    values = np.array(values)
    mean = values.mean()
    std = values.std()
    ci95 = 1.96 * std / np.sqrt(len(values))
    return mean, std, ci95


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading test data...")
    test_data = json.load(open("meco_test.json"))
    test_data = test_data[:100]
    loader = DataLoader(MECO(test_data), batch_size=4, shuffle=False, collate_fn=collate)

    summary_rows = []

    for name, (mode, n_experts) in ABLATIONS.items():
        print(f"\nEvaluating {name}...")

        model = EyeXpertM(mode=mode, n_experts=n_experts)
        model.load_state_dict(torch.load(f"{name}.pt", map_location=DEVICE))
        model.to(DEVICE)

        # Test evaluation
        detailed_metrics = evaluate_detailed(model, loader)

        # Per-language performance
        family_performance = evaluate_per_language(model, loader)

        # Compute stats for detailed metrics
        acc_stats = compute_stats([detailed_metrics["accuracy"]])
        lev_stats = compute_stats([detailed_metrics["levenshtein"]])
        reg_pred_stats = compute_stats([detailed_metrics["regression_pred"]])
        reg_true_stats = compute_stats([detailed_metrics["regression_true"]])
        exact_match_stats = compute_stats([detailed_metrics["exact_match"]])
        dur_mse_stats = compute_stats([detailed_metrics["dur_mse"]])
        dur_mae_stats = compute_stats([detailed_metrics["dur_mae"]])

        summary_rows.append({
            "model": name,
            "accuracy_mean": acc_stats[0],
            "accuracy_std": acc_stats[1],
            "accuracy_ci95": acc_stats[2],
            "levenshtein_mean": lev_stats[0],
            "levenshtein_std": lev_stats[1],
            "levenshtein_ci95": lev_stats[2],
            "regression_pred_mean": reg_pred_stats[0],
            "regression_pred_std": reg_pred_stats[1],
            "regression_pred_ci95": reg_pred_stats[2],
            "regression_true_mean": reg_true_stats[0],
            "regression_true_std": reg_true_stats[1],
            "regression_true_ci95": reg_true_stats[2],
            "exact_match_mean": exact_match_stats[0],
            "exact_match_std": exact_match_stats[1],
            "exact_match_ci95": exact_match_stats[2],
            "dur_mse_mean": dur_mse_stats[0],
            "dur_mse_std": dur_mse_stats[1],
            "dur_mse_ci95": dur_mse_stats[2],
            "dur_mae_mean": dur_mae_stats[0],
            "dur_mae_std": dur_mae_stats[1],
            "dur_mae_ci95": dur_mae_stats[2],
        })

        # Add family performance
        for family_id in range(5):
            family_name = {0: "germanic", 1: "romance", 2: "north_germanic", 3: "slavic", 4: "finno_ugric"}.get(
                family_id, f"family_{family_id}")
            summary_rows[-1][f"{family_name}_loss"] = family_performance.get(family_id, float("nan"))

    # Save summary
    fieldnames = [
        "model",
        "accuracy_mean", "accuracy_std", "accuracy_ci95",
        "levenshtein_mean", "levenshtein_std", "levenshtein_ci95",
        "regression_pred_mean", "regression_pred_std", "regression_pred_ci95",
        "regression_true_mean", "regression_true_std", "regression_true_ci95",
        "exact_match_mean", "exact_match_std", "exact_match_ci95",
        "dur_mse_mean", "dur_mse_std", "dur_mse_ci95",
        "dur_mae_mean", "dur_mae_std", "dur_mae_ci95",
        "germanic_loss", "romance_loss", "north_germanic_loss", "slavic_loss", "finno_ugric_loss"
    ]
    with open("experiment_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDone!")
    print("✔ Summary stats: experiment_stats.csv")


if __name__ == "__main__":
    main()
