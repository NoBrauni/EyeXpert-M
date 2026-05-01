import json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM
from train_meco import MECO, collate, tokenizer


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

USE_AMP = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# =========================================================
# HYPERPARAMETERS
# =========================================================
LR = 1e-4
BATCH_SIZE = 2
ACCUM_STEPS = 4

LAMBDA_DUR_MAX = 0.01
MOE_LAMBDA = 0.01


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# STRATIFIED SUBSET
# =========================================================
def stratified_subsample(data, fraction=0.7, seed=42):
    set_seed(seed)

    buckets = defaultdict(list)
    for item in data:
        buckets[item["lang"]].append(item)

    new_data = []
    for lang, items in buckets.items():
        random.shuffle(items)
        n = max(1, int(len(items) * fraction))
        new_data.extend(items[:n])

    random.shuffle(new_data)
    return new_data


# =========================================================
# SPLIT
# =========================================================
def split_data(data, seed=42):
    set_seed(seed)
    data = data.copy()
    random.shuffle(data)

    n = len(data)
    return (
        data[: int(0.7 * n)],
        data[int(0.7 * n): int(0.85 * n)],
        data[int(0.85 * n):]
    )


# =========================================================
# ABLATIONS
# =========================================================
ABLATIONS = {

    "A0_baseline": dict(
        use_experts=False,
        routing="none",
        use_lang=False,
        n_experts=5
    ),

    "A1_lang": dict(
        use_experts=False,
        routing="none",
        use_lang=True,
        n_experts=5
    ),

    "A2_hard_moe": dict(
        use_experts=True,
        routing="hard",
        use_lang=True,
        n_experts=5
    ),

    "A3_soft_moe": dict(
        use_experts=True,
        routing="soft",
        use_lang=True,
        n_experts=5
    ),

    "A4_single_expert": dict(
        use_experts=True,
        routing="hard",
        use_lang=True,
        n_experts=1
    ),
}


# =========================================================
# LOSS
# =========================================================
def loss_fn(logits, dur_pred, moe_metrics, batch, lambda_dur):

    B, T, V = logits.shape
    mask = batch["mask"].to(logits.device)

    fix = F.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T).to(logits.device),
        reduction="none"
    ).view(B, T)

    fix = (fix * mask).sum() / mask.sum()

    target_dur = torch.log1p(batch["durations"]).to(logits.device)

    dur = F.mse_loss(
        dur_pred.squeeze(-1),
        target_dur,
        reduction="none"
    )

    dur = (dur * mask).sum() / mask.sum()

    moe_loss = 0.0
    if moe_metrics:
        moe_loss = moe_metrics.get("load_balance", 0.0)

    return fix + lambda_dur * dur + MOE_LAMBDA * moe_loss


# =========================================================
# FORWARD
# =========================================================
def forward(model, batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}

    return model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["word_ids"],
        batch["scanpath"],
        batch["durations"],
        batch["family"],
        batch["lang_id"],
    )


# =========================================================
# TRAIN
# =========================================================
def train(model, train_loader, val_loader, name, epochs=3):

    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best = float("inf")

    for ep in range(epochs):

        model.train()
        total = 0
        optim.zero_grad(set_to_none=True)

        lambda_dur = min(LAMBDA_DUR_MAX, (ep + 1) / 3 * LAMBDA_DUR_MAX)

        for i, batch in enumerate(train_loader):

            logits, dur_pred, moe = forward(model, batch)

            loss = loss_fn(logits, dur_pred, moe, batch, lambda_dur)
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
            torch.save(model.state_dict(), f"{name}.pt")


# =========================================================
# EVAL
# =========================================================
def evaluate(model, loader):

    model.eval()
    total = 0
    count = 0

    with torch.no_grad():
        for batch in loader:

            logits, _, _ = forward(model, batch)

            B, T, V = logits.shape
            mask = batch["mask"].to(logits.device)

            loss = F.cross_entropy(
                logits.view(B * T, V),
                batch["scanpath"].view(B * T).to(logits.device),
                reduction="none"
            ).view(B, T)

            total += (loss * mask).sum().item()
            count += mask.sum().item()

    return total / max(count, 1.0)


# =========================================================
# RUN
# =========================================================
def run_ablation(name, cfg, train_loader, val_loader):

    print("\n========================")
    print("ABLATION:", name)
    print("========================\n")

    model = EyeXpertM(
        use_experts=cfg["use_experts"],
        routing=cfg["routing"],
        use_lang=cfg["use_lang"],
        n_experts=cfg["n_experts"],
        use_duration=True
    )

    train(model, train_loader, val_loader, name)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    data = json.load(open("meco_train.json", "r", encoding="utf-8"))

    data = stratified_subsample(data, fraction=0.7)

    train_data, val_data, _ = split_data(data)

    train_dataset = MECO(train_data, tokenizer)
    val_dataset = MECO(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    for name, cfg in ABLATIONS.items():
        run_ablation(name, cfg, train_loader, val_loader)