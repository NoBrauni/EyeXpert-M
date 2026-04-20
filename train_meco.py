import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import EyeXpertXLMR, LANG_FAMILY

# =========================================================
# DEVICE (ONLY CHANGE THIS FOR GPU)
# =========================================================
DEVICE = "cpu"


# =========================================================
# DATASET
# =========================================================
class MECO(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        return {
            "sentence": item["sentence"],
            "scanpath": torch.tensor(item["scanpath"], dtype=torch.long),
            "durations": torch.tensor(item["durations"], dtype=torch.float),
            "family": torch.tensor(LANG_FAMILY.get(item["lang"], 0), dtype=torch.long),
        }


# =========================================================
# COLLATE
# =========================================================
def collate(batch):

    sentences = [b["sentence"] for b in batch]

    scanpaths, durations, families, masks = [], [], [], []

    max_len = max(len(b["scanpath"]) for b in batch)

    for b in batch:

        sp = b["scanpath"]
        dur = b["durations"]
        L = len(sp)

        if L < max_len:
            pad = max_len - L
            sp = torch.cat([sp, torch.zeros(pad, dtype=torch.long)])
            dur = torch.cat([dur, torch.zeros(pad)])

        mask = torch.zeros(max_len)
        mask[:L] = 1

        scanpaths.append(sp)
        durations.append(dur)
        families.append(b["family"])
        masks.append(mask)

    return {
        "sentences": sentences,
        "scanpath": torch.stack(scanpaths),
        "durations": torch.stack(durations),
        "family": torch.stack(families),
        "mask": torch.stack(masks)
    }


# =========================================================
# LOSS
# =========================================================
def loss_fn(logits, dur_pred, moe_loss, batch):

    B, T, V = logits.shape

    fix = F.cross_entropy(
        logits.view(B * T, V),
        batch["scanpath"].view(B * T),
        reduction="none"
    ).view(B, T)

    mask = batch["mask"]

    fix = (fix * mask).sum() / mask.sum()

    dur = F.mse_loss(
        torch.log1p(dur_pred),
        torch.log1p(batch["durations"]),
        reduction="none"
    )

    dur = (dur * mask).sum() / mask.sum()

    moe = moe_loss

    return fix + 0.2 * dur + 0.01 * moe


# =========================================================
# TRAIN STEP
# =========================================================
def train_step(model, batch, optim):

    # move tensors to device
    batch = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    B = len(batch["family"])

    # IMPORTANT FIX:
    # pass real family as lang_id unless you explicitly ablate it later
    lang_id = batch["family"].clone()

    logits, dur, moe_loss = model(
        batch["sentences"],
        batch["scanpath"],
        batch["durations"],
        batch["family"],
        lang_id
    )

    loss = loss_fn(logits, dur, moe_loss, batch)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


# =========================================================
# TRAIN LOOP
# =========================================================
def train(model, loader, epochs=2):

    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    for ep in range(epochs):

        total = 0

        for i, batch in enumerate(loader):

            loss = train_step(model, batch, optim)
            total += loss

            if i % 10 == 0:
                print(f"[Epoch {ep}] Step {i} | Loss {loss:.4f}")

        print(f"\nEpoch {ep} | Avg Loss {total / len(loader):.4f}\n")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Loading dataset...")

    data = json.load(open("meco_train.json", "r", encoding="utf-8"))
    data = data[:300]  # CPU debug subset

    split = int(0.8 * len(data))

    train_data = data[:split]
    val_data = data[split:]

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

    print("Building model...")

    model = EyeXpertXLMR()

    print("Starting training...")

    train(model, train_loader, epochs=2)