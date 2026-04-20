import json
import random
import torch
from torch.utils.data import DataLoader

from model import EyeXpertXLMR, LANG_FAMILY
from train_meco import MECO, collate


# =========================================================
# DEVICE
# =========================================================
DEVICE = "cpu"


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# SAFE SPLIT (NO LEAKAGE)
# =========================================================
def split_data(data, seed=42):

    set_seed(seed)

    # shuffle once deterministically
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
# ABLATIONS (A0–A4)
# =========================================================
ABLATIONS = {
    "A0_full":       dict(use_moe=True,  use_saccade=True,  use_lang=True),
    "A1_no_moe":     dict(use_moe=False, use_saccade=True,  use_lang=True),
    "A2_no_saccade": dict(use_moe=True,  use_saccade=False, use_lang=True),
    "A3_no_lang":    dict(use_moe=True,  use_saccade=True,  use_lang=False),
    "A4_simple":     dict(use_moe=False, use_saccade=False, use_lang=False),
}


# =========================================================
# LOSS (same as training)
# =========================================================
def loss_fn(logits, dur_pred, moe_loss, batch):

    B, T, V = logits.shape

    fix = torch.nn.functional.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T),
        reduction="none"
    ).view(B, T)

    mask = batch["mask"]

    fix = (fix * mask).sum() / mask.sum()

    dur = torch.nn.functional.mse_loss(
        torch.log1p(dur_pred),
        torch.log1p(batch["durations"]),
        reduction="none"
    )

    dur = (dur * mask).sum() / mask.sum()

    moe = moe_loss if torch.is_tensor(moe_loss) else torch.tensor(0.0)

    return fix + 0.2 * dur + 0.01 * moe


# =========================================================
# FORWARD PASS
# =========================================================
def forward(model, batch):

    lang_id = torch.zeros(len(batch["family"]), dtype=torch.long, device=DEVICE)

    return model(
        batch["sentences"],
        batch["scanpath"],
        batch["durations"],
        batch["family"],
        lang_id
    )


# =========================================================
# TRAIN STEP
# =========================================================
def train_step(model, batch, optim):

    logits, dur, moe = forward(model, batch)

    loss = loss_fn(logits, dur, moe, batch)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader):

    model.eval()
    total = 0

    with torch.no_grad():
        for batch in loader:

            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            logits, dur, moe = forward(model, batch)
            loss = loss_fn(logits, dur, moe, batch)

            total += loss.item()

    model.train()
    return total / len(loader)


# =========================================================
# TRAIN LOOP
# =========================================================
def train(model, train_loader, val_loader, name, epochs=2):

    model.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

    best_val = float("inf")

    for ep in range(epochs):

        total = 0

        for i, batch in enumerate(train_loader):

            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            loss = train_step(model, batch, optim)
            total += loss

            if i % 10 == 0:
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

    model = EyeXpertXLMR(
        use_moe=cfg["use_moe"],
        use_saccade=cfg["use_saccade"],
        use_lang=cfg["use_lang"]
    )

    train(model, train_loader, val_loader, name)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading dataset...")

    data = json.load(open("meco_train.json", "r", encoding="utf-8"))
    data = data[:500]  # CPU subset

    train_data, val_data, test_data = split_data(data)

    train_loader = DataLoader(
        MECO(train_data),
        batch_size=2,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        MECO(val_data),
        batch_size=2,
        shuffle=False,
        collate_fn=collate
    )

    print("Device:", DEVICE)

    for name, cfg in ABLATIONS.items():
        run_ablation(name, cfg, train_loader, val_loader)