import optuna
import torch
import torch.optim as optim
import random
import gc

from model_definition import EyeExpertM
from train_model import (
    train_samples,
    val_samples,
    train_epoch,
    evaluate,
    embedding_size,
    embedding_cache
)

# ------------------------------
# Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

# ------------------------------
# Subset dataset for faster tuning
# ------------------------------
def sample_subset(samples, n_samples=500, seed=42):
    random.seed(seed)
    if len(samples) <= n_samples:
        return samples
    return random.sample(samples, n_samples)


# ------------------------------
# Optuna objective
# ------------------------------
def objective(trial):

    # ------------------------------
    # Hyperparameters
    # ------------------------------
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 1, 3)

    attention_type = trial.suggest_categorical(
        "attention_type",
        ["dot", "additive", None]
    )

    window_size = trial.suggest_int("window_size", 4, 12)

    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    alpha = trial.suggest_float("alpha", 0.3, 0.8)

    # ------------------------------
    # Model
    # ------------------------------
    model = EyeExpertM(
        hidden_dim=hidden_dim,
        encoder_dim=768,
        n_experts=5,
        max_seq_len=embedding_size,
        n_layers=n_layers,
        attention_type=attention_type,
        window_size=window_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------------
    # Use subsets
    # ------------------------------
    tuning_train = sample_subset(train_samples, 500)
    tuning_val   = sample_subset(val_samples, 300)

    # ------------------------------
    # Training loop
    # ------------------------------
    epochs = 3
    val_loss = None

    for epoch in range(epochs):

        # ⚠️ IMPORTANT: fix expert_id (no curriculum in tuning)
        expert_id = 0

        train_loss = train_epoch(
            model,
            tuning_train,
            optimizer,
            batch_size=batch_size,
            alpha=alpha,
            device=device,
            expert_id=expert_id,
            embedding_cache=embedding_cache
        )

        val_loss = evaluate(
            model,
            tuning_val,
            batch_size=batch_size,
            alpha=alpha,
            device=device,
            expert_id=expert_id,
            embedding_cache=embedding_cache,
            desc="Optuna Val"
        )

        # Report to Optuna
        trial.report(val_loss, epoch)

        if trial.should_prune():
            del model
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()

    # ------------------------------
    # Cleanup
    # ------------------------------
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss


# ------------------------------
# Run study
# ------------------------------
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=50)

    print("\n=============================")
    print("Best trial:", study.best_trial.value)

    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")