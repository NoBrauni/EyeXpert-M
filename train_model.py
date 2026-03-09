import os
import random
import torch
import torch.optim as optim
from model_definition import MECODataset
from model_definition import EyeExpertM, LANG_TO_EXPERT, batch_precompute_embeddings
from test_model import collate_batch, train_epoch, evaluate

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load all CSVs
# -------------------------------
csv_paths = [
    # Expert 0 — Germanic
    "processed_data/meco_l1_ge_processed.csv",
    "processed_data/meco_l1_ge_po_processed.csv",
    "processed_data/meco_l1_ge_zu_processed.csv",
    "processed_data/meco_l1_du_processed.csv",
    "processed_data/meco_l1_en_processed.csv",
    "processed_data/meco_l1_en_uk_processed.csv",
    "processed_data/meco_l1_ic_processed.csv",
    # Expert 1 — Nordic
    "processed_data/meco_l1_no_processed.csv",
    "processed_data/meco_l1_se_processed.csv",
    "processed_data/meco_l1_da_processed.csv",
    # Expert 2 — Romance
    "processed_data/meco_l1_sp_processed.csv",
    "processed_data/meco_l1_sp_ch_processed.csv",
    "processed_data/meco_l1_it_processed.csv",
    "processed_data/meco_l1_bp_processed.csv",
    # Expert 3 — Slavic
    "processed_data/meco_l1_ru_processed.csv",
    "processed_data/meco_l1_ru_mo_processed.csv",
    # Expert 4 — Uralic
    "processed_data/meco_l1_fi_processed.csv",
    "processed_data/meco_l1_ee_processed.csv",
]

all_samples = []
for path in csv_paths:
    ds = MECODataset(path)
    all_samples.extend(ds.samples)

print(f"Total samples loaded: {len(all_samples)}")

# -------------------------------
# Precompute embeddings
# -------------------------------
all_samples = batch_precompute_embeddings(all_samples, batch_size=16, cache_path="embeddings_cached.pkl")
print("Embeddings precomputed.")

# -------------------------------
# Split data by reader (train/test/verify)
# -------------------------------
all_readers = list({s["reader"] for s in all_samples})
random.seed(42)
random.shuffle(all_readers)
n = len(all_readers)
train_readers = all_readers[:int(0.7*n)]
test_readers = all_readers[int(0.7*n):int(0.85*n)]
verify_readers = all_readers[int(0.85*n):]

train_samples = [s for s in all_samples if s["reader"] in train_readers]
test_samples  = [s for s in all_samples if s["reader"] in test_readers]
verify_samples = [s for s in all_samples if s["reader"] in verify_readers]

print(f"Train: {len(train_samples)}, Test: {len(test_samples)}, Verify: {len(verify_samples)}")

# -------------------------------
# Initialize model and optimizer
# -------------------------------
model = EyeExpertM(hidden_dim=256, encoder_dim=768, n_experts=5, max_seq_len=200).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
alpha = 0.5  # balance between word prediction and duration loss



# -------------------------------
# Main training loop
# -------------------------------
epochs = 10
batch_size = 16
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1} ===")
    train_global, train_expert = train_epoch(model, train_samples, optimizer, batch_size, alpha)
    test_global, test_expert = evaluate(model, test_samples, batch_size, alpha)
    verify_global, verify_expert = evaluate(model, verify_samples, batch_size, alpha)

    print(f"Train loss: {train_global:.4f} | Test loss: {test_global:.4f} | Verify loss: {verify_global:.4f}")
    print(f"Per-expert train losses: {train_expert}")
    print(f"Per-expert test losses: {test_expert}")
    print(f"Per-expert verify losses: {verify_expert}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/eyeexpert_epoch{epoch+1}.pt")