import optuna
import torch
import json
from torch.utils.data import DataLoader

from model import EyeXpertM
from train_meco import MECO, collate
from ablations import split_data, train, evaluate


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# =========================================================
# LOAD DATA (same split as ablations = CRITICAL)
# =========================================================
data = json.load(open("meco_train.json", "r", encoding="utf-8"))
data = data[:500]  # keep consistent with your ablation setup

train_data, val_data, _ = split_data(data)

train_dataset = MECO(train_data)
val_dataset = MECO(val_data)


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def objective(trial):

    # -------------------------
    # Hyperparameters
    # -------------------------
    lr = trial.suggest_categorical("lr", [1e-4, 2e-4, 3e-4])
    batch_size = trial.suggest_categorical("batch_size", [2, 4])

    lambda_dur = trial.suggest_categorical("lambda_dur", [0.05, 0.1, 0.2])
    lambda_moe = trial.suggest_categorical("lambda_moe", [0.0, 0.001, 0.01])

    # -------------------------
    # DataLoaders
    # -------------------------
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
    # Model (BASELINE ONLY)
    # -------------------------
    model = EyeXpertM(
        use_experts=False,   # IMPORTANT: stable tuning setup
        routing="none",
        use_lang=False
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")

    # -------------------------
    # Small training budget (important for HPO speed)
    # -------------------------
    epochs = 2

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch in train_loader:

            batch = {
                k: v.to(DEVICE) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            logits, dur_pred, moe_loss = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_ids"],
                batch["scanpath"],
                batch["durations"],
                batch["family"],
                batch["lang_id"]
            )

            # -------------------------
            # LOSS (same as training file)
            # -------------------------
            fix_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["scanpath"].view(-1),
                reduction="mean"
            )

            dur_loss = torch.nn.functional.mse_loss(
                torch.log1p(dur_pred.squeeze(-1)),
                torch.log1p(batch["durations"])
            )

            loss = fix_loss + lambda_dur * dur_loss + lambda_moe * moe_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------------------------
        # VALIDATION
        # -------------------------
        val_loss = evaluate(model, val_loader)

        best_val = min(best_val, val_loss)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


# =========================================================
# RUN STUDY
# =========================================================
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    study.optimize(objective, n_trials=20)

    print("\n========================")
    print("BEST RESULT")
    print("========================")

    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")