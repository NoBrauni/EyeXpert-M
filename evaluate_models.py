import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODELS
# =========================================================
MODEL_FILES = {
    "A0_baseline": "A0_baseline_best.pt",
    "A1_lang_cond": "A1_lang_cond_best.pt",
    "A2_hard_moe": "A2_hard_moe_best.pt",
    "A3_hard_moe_lang": "A3_hard_moe_lang_best.pt",
    "A4_soft_moe": "A4_soft_moe_best.pt",
}


# =========================================================
# FIXATION ACCURACY
# =========================================================
def fixation_accuracy(pred, gold, mask):
    correct = (pred == gold).float()
    return (correct * mask).sum() / mask.sum()


# =========================================================
# NORMALIZED LEVENSHTEIN (SCANPATH SIMILARITY)
# =========================================================
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


# =========================================================
# REGRESSION RATE (COGNITIVE PROXY)
# =========================================================
def regression_rate(seq):
    if len(seq) < 2:
        return 0.0
    jumps = torch.diff(torch.tensor(seq))
    return (jumps < 0).float().mean().item()


# =========================================================
# DURATION METRICS (LOG SPACE STANDARD IN PSYCHOLINGUISTICS)
# =========================================================
def duration_metrics(pred_dur, gold_dur, mask):

    pred = torch.log1p(pred_dur)
    gold = torch.log1p(gold_dur)

    mse = ((pred - gold) ** 2 * mask).sum() / mask.sum()
    mae = (torch.abs(pred - gold) * mask).sum() / mask.sum()

    return {
        "dur_mse": mse.item(),
        "dur_mae": mae.item()
    }


# =========================================================
# FULL BATCH EVALUATION
# =========================================================
def evaluate_batch(pred, gold, mask, pred_dur, gold_dur):

    B = pred.size(0)

    acc = fixation_accuracy(pred, gold, mask)

    lev_total = 0
    reg_p = 0
    reg_g = 0

    for i in range(B):
        L = int(mask[i].sum().item())

        p = pred[i][:L].tolist()
        g = gold[i][:L].tolist()

        lev_total += normalized_levenshtein(p, g)
        reg_p += regression_rate(p)
        reg_g += regression_rate(g)

    lev = lev_total / B
    reg_p /= B
    reg_g /= B

    dur = duration_metrics(pred_dur, gold_dur, mask)

    return {
        "accuracy": acc.item(),
        "levenshtein": lev,
        "regression_pred": reg_p,
        "regression_true": reg_g,
        **dur
    }


# =========================================================
# EVALUATION LOOP
# =========================================================
def evaluate(model, loader):

    model.eval()

    totals = None
    steps = 0

    with torch.no_grad():
        for batch in loader:

            batch = {
                k: v.to(DEVICE) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

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

            metrics = evaluate_batch(
                pred,
                batch["scanpath"],
                batch["mask"],
                dur_pred,
                batch["durations"]
            )

            if totals is None:
                totals = {k: 0.0 for k in metrics}

            for k in metrics:
                totals[k] += metrics[k]

            steps += 1

    for k in totals:
        totals[k] /= steps

    return totals


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(name, path):

    cfg = {
        "A0_baseline": dict(use_experts=False, routing="none", use_lang=False),
        "A1_lang_cond": dict(use_experts=False, routing="none", use_lang=True),
        "A2_hard_moe": dict(use_experts=True, routing="hard", use_lang=False),
        "A3_hard_moe_lang": dict(use_experts=True, routing="hard", use_lang=True),
        "A4_soft_moe": dict(use_experts=True, routing="soft", use_lang=False),
    }[name]

    model = EyeXpertM(**cfg)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)

    return model


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading test set...")

    test_data = json.load(open("meco_test.json", "r", encoding="utf-8"))[:100]

    loader = DataLoader(
        MECO(test_data),
        batch_size=2,
        shuffle=False,
        collate_fn=collate
    )

    print(f"Device: {DEVICE}\n")

    for name, path in MODEL_FILES.items():

        print(f"Evaluating {name}...")

        model = load_model(name, path)

        metrics = evaluate(model, loader)

        print(f"\n{name}")
        print(f"  Accuracy        : {metrics['accuracy']:.4f}")
        print(f"  Levenshtein     : {metrics['levenshtein']:.4f}")
        print(f"  Regression (P)  : {metrics['regression_pred']:.4f}")
        print(f"  Regression (T)  : {metrics['regression_true']:.4f}")
        print(f"  Duration MSE    : {metrics['dur_mse']:.4f}")
        print(f"  Duration MAE    : {metrics['dur_mae']:.4f}")
        print("")