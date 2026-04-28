import optuna
import torch
import json
import torch.nn.functional as F
import random
from collections import defaultdict
from torch.utils.data import DataLoader

from model import EyeXpertM
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

USE_AMP = DEVICE.type == "cuda"


# =========================================================
# STRATIFIED SAMPLING (IMPORTANT FIX)
# =========================================================
def stratified_subset(data, max_per_lang=50, seed=42):

    random.seed(seed)
    buckets = defaultdict(list)

    for item in data:
        buckets[item["lang"]].append(item)

    subset = []

    for lang, items in buckets.items():
        random.shuffle(items)
        subset.extend(items[:max_per_lang])

    random.shuffle(subset)
    return subset


# =========================================================
# DATA
# =========================================================
data = json.load(open("meco_train.json", "r", encoding="utf-8"))

# STRATIFIED SUBSET (replaces naive slicing)
data = stratified_subset(data, max_per_lang=50)

from ablations import split_data
train_data, val_data, _ = split_data(data)

train_dataset = MECO(train_data)
val_dataset = MECO(val_data)


# =========================================================
# LOSS (MATCHES FINAL ABLATION SETUP)
# =========================================================
def compute_loss(logits, dur_pred, batch, lambda_dur):

    B, T, V = logits.shape
    mask = batch["mask"].to(logits.device)

    # -------------------------
    # FIXATION LOSS
    # -------------------------
    fix = F.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T),
        reduction="none"
    ).view(B, T)

    fix = (fix * mask).sum() / mask.sum()

    # -------------------------
    # DURATION LOSS
    # -------------------------
    target = torch.log1p(batch["durations"]).to(logits.device)

    dur_loss = F.mse_loss(
        torch.log1p(dur_pred.squeeze(-1)),
        target,
        reduction="none"
    )

    dur_loss = (dur_loss * mask).sum() / mask.sum()

    return fix + lambda_dur * dur_loss


# =========================================================
# OBJECTIVE
# =========================================================
def objective(trial):

    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 2e-4])
    lambda_dur = trial.suggest_categorical("lambda_dur", [0.01, 0.05, 0.1])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    # -------------------------
    # MODEL (BASELINE ONLY FOR HPO)
    # -------------------------
    model = EyeXpertM(
        use_experts=False,
        routing="none",
        use_lang=False,
        use_duration=True
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    model.train()

    # =====================================================
    # TRAIN (1 EPOCH)
    # =====================================================
    for batch in train_loader:

        batch = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=USE_AMP):

            logits, dur_pred, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"]
            )

            loss = compute_loss(logits, dur_pred, batch, lambda_dur)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # =====================================================
    # VALIDATION
    # =====================================================
    model.eval()
    total, steps = 0.0, 0.0

    with torch.no_grad():
        for batch in val_loader:

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

            loss = compute_loss(logits, dur_pred, batch, lambda_dur)

            bs = batch["mask"].sum().item()
            total += loss.item() * bs
            steps += bs

    return total / max(steps, 1.0)


# =========================================================
# RUN STUDY
# =========================================================
if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\nBEST RESULT:")
    print(study.best_value)

    print("\nBEST PARAMS:")
    for k, v in study.best_params.items():
        print(k, v)