import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import EyeXpertM
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

torch.backends.cudnn.benchmark = True

USE_AMP = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA SPLIT
# =========================================================
def split_data(data, seed=42):
    set_seed(seed)
    data = data.copy()
    random.shuffle(data)

    n = len(data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print("Dataset split:")
    print("Train:", len(train_data))
    print("Val  :", len(val_data))
    print("Test :", len(test_data))

    return train_data, val_data, test_data


# =========================================================
# ABLATIONS (CLEAN FACTORIAL DESIGN)
# =========================================================
ABLATIONS = {
    # baseline
    "A0_baseline": dict(use_experts=False, routing="none", use_lang=False),

    # language effect only
    "A1_lang": dict(use_experts=False, routing="none", use_lang=True),

    # MoE effect (hard routing)
    "A2_hard_moe": dict(use_experts=True, routing="hard", use_lang=True),

    # routing comparison (soft)
    "A3_soft_moe": dict(use_experts=True, routing="soft", use_lang=True),
}


# =========================================================
# LOSS FUNCTION
# =========================================================
def loss_fn(logits, dur_pred, moe_loss, batch):

    B, T, V = logits.shape
    mask = batch["mask"].to(logits.device)

    # -------------------------
    # FIXATION LOSS (PRIMARY)
    # -------------------------
    fix = F.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T),
        reduction="none"
    ).view(B, T)

    fix = (fix * mask).sum() / mask.sum()

    # -------------------------
    # DURATION LOSS (AUXILIARY, ALWAYS USED)
    # -------------------------
    target_dur = torch.log1p(batch["durations"]).to(logits.device)

    dur_loss = F.mse_loss(
        dur_pred.squeeze(-1),
        target_dur,
        reduction="none"
    )

    dur_loss = (dur_loss * mask).sum() / mask.sum()

    # -------------------------
    # MOE LOSS (REGULARIZER)
    # -------------------------
    moe = moe_loss if torch.is_tensor(moe_loss) else torch.tensor(0.0, device=logits.device)

    # -------------------------
    # FIXED MULTI-TASK WEIGHTING (IMPORTANT FOR THESIS)
    # -------------------------
    return fix + 0.2 * dur_loss + 0.01 * moe


# =========================================================
# FORWARD PASS
# =========================================================
def forward(model, batch):
    return model(
        batch["input_ids"].to(DEVICE),
        batch["attention_mask"].to(DEVICE),
        batch["word_ids"].to(DEVICE),
        batch["scanpath"].to(DEVICE),
        batch["durations"].to(DEVICE),
        batch["family"].to(DEVICE),
        batch["lang_id"].to(DEVICE),
    )


# =========================================================
# TRAIN STEP
# =========================================================
def train_step(model, batch, optim):

    optim.zero_grad(set_to_none=True)

    if USE_AMP:
        with torch.autocast("cuda"):

            logits, dur_pred, moe = forward(model, batch)
            loss = loss_fn(logits, dur_pred, moe, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optim)
        scaler.update()

        return loss.item()

    else:
        logits, dur_pred, moe = forward(model, batch)
        loss = loss_fn(logits, dur_pred, moe, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        return loss.item()


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader):

    model.eval()
    total_loss = 0.0
    total_count = 0.0

    with torch.no_grad():
        for batch in loader:

            logits, dur_pred, moe = forward(model, batch)
            loss = loss_fn(logits, dur_pred, moe, batch)

            bs = batch["mask"].sum().item()
            total_loss += loss.item() * bs
            total_count += bs

    model.train()
    return total_loss / max(total_count, 1.0)


# =========================================================
# TRAIN LOOP
# =========================================================
def train(model, train_loader, val_loader, name, epochs=3):

    model.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

    best_val = float("inf")

    for ep in range(epochs):

        total = 0.0

        for i, batch in enumerate(train_loader):

            loss = train_step(model, batch, optim)
            total += loss

            if i % 20 == 0:
                print(f"[{name}] Epoch {ep} Step {i} | Loss {loss:.4f}")

        val_loss = evaluate(model, val_loader)

        print(f"\n[{name}] Epoch {ep}")
        print("Train:", total / len(train_loader))
        print("Val  :", val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{name}_best.pt")
            print("✔ saved best model")


# =========================================================
# RUN ABLATION
# =========================================================
def run_ablation(name, cfg, train_loader, val_loader):

    print("\n========================")
    print("ABLATION:", name)
    print("========================\n")

    model = EyeXpertM(
        use_experts=cfg["use_experts"],
        routing=cfg["routing"],
        use_lang=cfg["use_lang"],
        use_duration=True  # ALWAYS ENABLED (core design choice)
    )

    train(model, train_loader, val_loader, name)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading dataset...")

    data = json.load(open("meco_train.json", "r", encoding="utf-8"))
    data = data[:500]

    train_data, val_data, test_data = split_data(data)

    train_loader = DataLoader(
        MECO(train_data),
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        MECO(val_data),
        batch_size=4,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )

    print("Starting ablations...\n")

    for name, cfg in ABLATIONS.items():
        run_ablation(name, cfg, train_loader, val_loader)