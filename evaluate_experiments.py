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
# METRICS
# -----------------------------
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


def compute_stats(values):
    values = np.array(values)
    mean = values.mean()
    std = values.std()
    ci95 = 1.96 * std / np.sqrt(len(values))
    return mean, std, ci95


# -----------------------------
# PER-SAMPLE EVALUATION
# -----------------------------
def evaluate_detailed(model, loader):
    """
    Returns per-sample metrics
    """
    model.eval()
    results = []

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

            B = pred.size(0)

            for i in range(B):
                L = int(batch["scanpath_mask"][i].sum().item())

                p = pred[i][:L].tolist()
                g = batch["scanpath"][i][:L].tolist()

                # fixation accuracy
                acc = float(np.mean(np.array(p) == np.array(g)))

                # scanpath similarity
                lev = normalized_levenshtein(p, g)

                # regression behavior
                reg_p = regression_rate(p)
                reg_g = regression_rate(g)

                # duration error
                pred_d = dur_pred[i][:L].squeeze(-1)
                gold_d = torch.log1p(batch["durations"][i][:L])

                dur_mse = F.mse_loss(pred_d, gold_d).item()
                dur_mae = F.l1_loss(pred_d, gold_d).item()

                exact = float(p == g)

                results.append({
                    "accuracy": acc,
                    "levenshtein": lev,
                    "regression_pred": reg_p,
                    "regression_true": reg_g,
                    "exact_match": exact,
                    "dur_mse": dur_mse,
                    "dur_mae": dur_mae
                })

    return results


# -----------------------------
# PER-LANGUAGE
# -----------------------------
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
                fam = int(families[b])
                family_stats[fam]["loss"] += masked_loss[b].item()
                family_stats[fam]["count"] += masked_count[b].item()

    out = {}
    for k, v in family_stats.items():
        out[k] = v["loss"] / v["count"]

    return out


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading test data...")
    test_data = json.load(open("meco_test.json"))
    loader = DataLoader(MECO(test_data), batch_size=4, shuffle=False, collate_fn=collate)

    summary_rows = []

    for name, (mode, n_experts) in ABLATIONS.items():
        print(f"\nEvaluating {name}...")

        model = EyeXpertM(mode=mode, n_experts=n_experts)
        model.load_state_dict(torch.load(f"{name}.pt", map_location=DEVICE))
        model.to(DEVICE)

        # FIXED: per-sample metrics
        results = evaluate_detailed(model, loader)

        def get(key):
            return [r[key] for r in results]

        acc = compute_stats(get("accuracy"))
        lev = compute_stats(get("levenshtein"))
        reg_p = compute_stats(get("regression_pred"))
        reg_g = compute_stats(get("regression_true"))
        exact = compute_stats(get("exact_match"))
        mse = compute_stats(get("dur_mse"))
        mae = compute_stats(get("dur_mae"))

        family_perf = evaluate_per_language(model, loader)

        row = {
            "model": name,

            "accuracy_mean": acc[0],
            "accuracy_std": acc[1],
            "accuracy_ci95": acc[2],

            "levenshtein_mean": lev[0],
            "levenshtein_std": lev[1],
            "levenshtein_ci95": lev[2],

            "regression_pred_mean": reg_p[0],
            "regression_true_mean": reg_g[0],

            "exact_match_mean": exact[0],

            "dur_mse_mean": mse[0],
            "dur_mae_mean": mae[0],
        }

        fam_map = {
            0: "germanic",
            1: "romance",
            2: "north_germanic",
            3: "slavic",
            4: "finno_ugric"
        }

        for k in range(5):
            row[f"{fam_map[k]}_loss"] = family_perf.get(k, float("nan"))

        summary_rows.append(row)

    # save CSV
    fieldnames = list(summary_rows[0].keys())

    with open("experiment_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDone!")
    print("✔ Fixed, statistically valid results saved to experiment_stats.csv")


if __name__ == "__main__":
    main()