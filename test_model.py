from model_definition import MECODataset, batch_precompute_embeddings, EyeExpertM, train_epoch
import torch
import random

# Load cached embeddings
print("Loading cached embeddings...")
all_samples = batch_precompute_embeddings([], cache_path="embeddings_cache.pkl")

# Shuffle and split
random.shuffle(all_samples)
n = len(all_samples)
train_samples = all_samples[:int(0.8*n)]
val_samples = all_samples[int(0.8*n):int(0.9*n)]
test_samples = all_samples[int(0.9*n):]

train_dataset = MECODataset()
train_dataset.samples = train_samples

# Initialize model and optimizer
model = EyeExpertM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Short training
for epoch in range(3):
    global_loss, expert_losses = train_epoch(model, train_dataset, optimizer, batch_size=8)
    print(f"\nEpoch {epoch} — Global Avg Loss: {global_loss:.4f}")
    for eid in sorted(expert_losses.keys()):
        print(f"  Expert {eid} Avg Loss: {expert_losses[eid]:.4f}")