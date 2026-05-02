import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM, LANG_FAMILY
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# ABLATIONS
# =========================================================
MODEL_FILES = {
    "base": "base.pt",
    "base_plus": "base_plus.pt", 
    "hard_5": "hard_5.pt",
    "moe_5": "moe_5.pt",
}


MODEL_CONFIGS = {
    "base": {"mode": "base", "n_experts": 5},        # No experts, no routing
    "base_plus": {"mode": "base+", "n_experts": 5},  # Parameter-matched baseline
    "hard_5": {"mode": "hard", "n_experts": 5},      # Hard routing by language family
    "moe_5": {"mode": "moe", "n_experts": 5},        # Soft MoE routing
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


def duration_metrics(pred_dur, gold_dur, mask):
    pred = pred_dur.squeeze(-1)
    gold = torch.log1p(gold_dur)

    mse = ((pred - gold) ** 2 * mask).sum() / mask.sum()
    mae = (torch.abs(pred - gold) * mask).sum() / mask.sum()

    return {
        "dur_mse": mse.item(),
        "dur_mae": mae.item()
    }


# =========================================================
# BATCH EVAL
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
        **dur,
        "tokens": mask.sum().item()
    }


# =========================================================
# FULL EVAL LOOP
# =========================================================
def evaluate(model, loader):

    model.eval()

    totals = None
    total_tokens = 0

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
                batch["scanpath_mask"],
                dur_pred,
                batch["durations"]
            )

            if totals is None:
                totals = {k: 0.0 for k in metrics if k != "tokens"}

            tokens = metrics["tokens"]
            total_tokens += tokens

            for k in totals:
                totals[k] += metrics[k] * tokens

    for k in totals:
        totals[k] /= total_tokens

    return totals


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(name, path):

    cfg = MODEL_CONFIGS[name]

    model = EyeXpertM(**cfg)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)

    return model


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading test set...")

    test_data = json.load(open("meco_test.json", "r", encoding="utf-8"))

    dataset = MECO(test_data)

    loader = DataLoader(
        dataset,
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