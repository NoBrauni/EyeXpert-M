import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertXLMR
from train_meco import MECO, collate


# =========================================================
# CONFIG
# =========================================================
DEVICE = "cpu"

MODEL_FILES = {
    "A0_full": "A0_full_best.pt",
    "A1_no_moe": "A1_no_moe_best.pt",
    "A2_no_saccade": "A2_no_saccade_best.pt",
    "A3_no_lang": "A3_no_lang_best.pt",
    "A4_simple": "A4_simple_best.pt",
}


# =========================================================
# LEVENSHTEIN
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


# =========================================================
# DTW (Dynamic Time Warping)
# =========================================================
def dtw_distance(a, b):
    n, m = len(a), len(b)

    dtw = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i][j] = cost + min(
                dtw[i - 1][j],
                dtw[i][j - 1],
                dtw[i - 1][j - 1]
            )

    return dtw[n][m]


# =========================================================
# METRICS
# =========================================================
def compute_metrics(logits, dur_pred, batch):

    B, T, V = logits.shape
    mask = batch["mask"]

    pred = logits.argmax(dim=-1)

    # -------------------------
    # Accuracy
    # -------------------------
    correct = (pred == batch["scanpath"]).float()
    acc = (correct * mask).sum() / mask.sum()

    # -------------------------
    # Fixation loss
    # -------------------------
    fix_loss = F.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T),
        reduction="none"
    ).view(B, T)

    fix_loss = (fix_loss * mask).sum() / mask.sum()

    # -------------------------
    # Duration loss
    # -------------------------
    dur_loss = F.mse_loss(
        torch.log1p(dur_pred),
        torch.log1p(batch["durations"]),
        reduction="none"
    )

    dur_loss = (dur_loss * mask).sum() / mask.sum()

    # -------------------------
    # Saccade error
    # -------------------------
    pred_jump = pred[:, 1:] - pred[:, :-1]
    true_jump = batch["scanpath"][:, 1:] - batch["scanpath"][:, :-1]

    jump_mask = mask[:, 1:]

    sacc_err = torch.abs(pred_jump - true_jump).float()
    sacc_err = (sacc_err * jump_mask).sum() / jump_mask.sum()

    # -------------------------
    # Sequence metrics
    # -------------------------
    lev_total = 0
    dtw_total = 0
    reg_pred_total = 0
    reg_true_total = 0
    count = 0

    pred_seq = pred.cpu().tolist()
    true_seq = batch["scanpath"].cpu().tolist()
    mask_seq = mask.cpu().tolist()

    for p, t, m in zip(pred_seq, true_seq, mask_seq):

        L = int(sum(m))

        p = p[:L]
        t = t[:L]

        # Levenshtein
        lev = levenshtein(p, t) / max(len(t), 1)

        # DTW
        dtw = dtw_distance(p, t) / max(len(t), 1)

        # Regression rate (backward movements)
        def regression_rate(seq):
            if len(seq) < 2:
                return 0
            jumps = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
            reg = sum(1 for j in jumps if j < 0)
            return reg / len(jumps)

        reg_pred = regression_rate(p)
        reg_true = regression_rate(t)

        lev_total += lev
        dtw_total += dtw
        reg_pred_total += reg_pred
        reg_true_total += reg_true

        count += 1

    return {
        "acc": acc.item(),
        "fix_loss": fix_loss.item(),
        "dur_loss": dur_loss.item(),
        "saccade_error": sacc_err.item(),
        "levenshtein": lev_total / count,
        "dtw": dtw_total / count,
        "reg_pred": reg_pred_total / count,
        "reg_true": reg_true_total / count,
    }


# =========================================================
# EVALUATE
# =========================================================
def evaluate(model, loader):

    model.eval()

    totals = None
    count = 0

    with torch.no_grad():
        for batch in loader:

            batch = {
                k: v.to(DEVICE) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            logits, dur_pred, _ = model(
                batch["sentences"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                torch.zeros(len(batch["family"]), dtype=torch.long, device=DEVICE)
            )

            metrics = compute_metrics(logits, dur_pred, batch)

            if totals is None:
                totals = {k: 0 for k in metrics}

            for k in metrics:
                totals[k] += metrics[k]

            count += 1

    for k in totals:
        totals[k] /= count

    return totals


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(name, path):

    cfg_map = {
        "A0_full": dict(use_moe=True, use_saccade=True, use_lang=True),
        "A1_no_moe": dict(use_moe=False, use_saccade=True, use_lang=True),
        "A2_no_saccade": dict(use_moe=True, use_saccade=False, use_lang=True),
        "A3_no_lang": dict(use_moe=True, use_saccade=True, use_lang=False),
        "A4_simple": dict(use_moe=False, use_saccade=False, use_lang=False),
    }

    model = EyeXpertXLMR(**cfg_map[name])
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)

    return model


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading TEST dataset...")

    test_data = json.load(open("meco_test.json", "r", encoding="utf-8"))
    test_data = test_data[:100]

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

        m = evaluate(model, loader)

        print(f"\n{name}")
        print(f"  Accuracy        : {m['acc']:.4f}")
        print(f"  Fix Loss        : {m['fix_loss']:.4f}")
        print(f"  Duration Loss   : {m['dur_loss']:.4f}")
        print(f"  Saccade Error   : {m['saccade_error']:.4f}")
        print(f"  Levenshtein     : {m['levenshtein']:.4f}")
        print(f"  DTW             : {m['dtw']:.4f}")
        print(f"  Reg (pred)      : {m['reg_pred']:.4f}")
        print(f"  Reg (true)      : {m['reg_true']:.4f}")
        print("")