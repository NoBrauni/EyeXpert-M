import os
import random
import pickle
import torch
import torch.optim as optim
from model_definition import MECODataset, EyeExpertM, LANG_TO_EXPERT, collate_batch

# -------------------------------
# Experiment configuration
# -------------------------------
SPLIT_BY_READER = True
ENSURE_UNSEEN_SENTENCES = True
LEAVE_OUT_LANGUAGE = None

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)

# -------------------------------
# Expert curriculum training
# -------------------------------
USE_EXPERT_CURRICULUM = True
CURRICULUM_MIX_RATIO = 0.2

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load original samples
# -------------------------------
data_dir = "data_full_sentence_fixed"
all_samples = []
for fname in os.listdir(data_dir):
    if fname.endswith(".csv"):
        ds = MECODataset(os.path.join(data_dir, fname))
        all_samples.extend(ds.samples)

print(f"Loaded {len(all_samples)} samples from CSVs.")

# -------------------------------
# Load cached embeddings
# -------------------------------
cache_path = "all_embeddings_cache.pkl"
with open(cache_path, "rb") as f:
    embedding_cache = pickle.load(f)

for s in all_samples:
    s["word_embeddings"] = embedding_cache[s["sentence"]]

print("Embeddings assigned to all samples.")

# -------------------------------
# Compute global max fixation index
# -------------------------------
all_fixations = [fix for s in all_samples for fix in s["scanpath"]]
max_fix_idx = max(all_fixations)
embedding_size = max_fix_idx + 1
print("Max fixation index overall:", max_fix_idx)

# -------------------------------
# Dataset splitting
# -------------------------------
def split_dataset(samples):
    if LEAVE_OUT_LANGUAGE is not None:
        train_samples = [s for s in samples if s["lang"] != LEAVE_OUT_LANGUAGE]
        heldout = [s for s in samples if s["lang"] == LEAVE_OUT_LANGUAGE]
        random.shuffle(heldout)
        n = len(heldout)
        val_samples = heldout[:int(0.5 * n)]
        test_samples = heldout[int(0.5 * n):]
        return train_samples, val_samples, test_samples

    if ENSURE_UNSEEN_SENTENCES:
        sentences = list({s["sentence"] for s in samples})
        random.shuffle(sentences)
        n = len(sentences)
        train_s = set(sentences[:int(TRAIN_RATIO * n)])
        val_s = set(sentences[int(TRAIN_RATIO * n):int((TRAIN_RATIO + VAL_RATIO) * n)])
        test_s = set(sentences[int((TRAIN_RATIO + VAL_RATIO) * n):])
        train_samples = [s for s in samples if s["sentence"] in train_s]
        val_samples = [s for s in samples if s["sentence"] in val_s]
        test_samples = [s for s in samples if s["sentence"] in test_s]
        return train_samples, val_samples, test_samples

    if SPLIT_BY_READER:
        readers = list({s["reader"] for s in samples})
        random.shuffle(readers)
        n = len(readers)
        train_r = readers[:int(TRAIN_RATIO * n)]
        val_r = readers[int(TRAIN_RATIO * n):int((TRAIN_RATIO + VAL_RATIO) * n)]
        test_r = readers[int((TRAIN_RATIO + VAL_RATIO) * n):]
        train_samples = [s for s in samples if s["reader"] in train_r]
        val_samples = [s for s in samples if s["reader"] in val_r]
        test_samples = [s for s in samples if s["reader"] in test_r]
        return train_samples, val_samples, test_samples

train_samples, val_samples, test_samples = split_dataset(all_samples)
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# -------------------------------
# Expert datasets
# -------------------------------
expert_datasets = {}
for s in train_samples:
    lang = s["lang"].lower()
    eid = LANG_TO_EXPERT.get(lang)
    if eid is None:
        continue
    expert_datasets.setdefault(eid, []).append(s)

print("Expert dataset sizes:")
for eid, data in expert_datasets.items():
    print(f"Expert {eid}: {len(data)} samples")

def sample_curriculum_dataset(epoch, expert_datasets, mix_ratio=0.2):
    target_expert = epoch % len(expert_datasets)
    target_samples = expert_datasets[target_expert]
    other_samples = []
    for eid, data in expert_datasets.items():
        if eid != target_expert:
            other_samples.extend(data)
    mix_size = int(len(target_samples) * mix_ratio)
    mixed_samples = random.sample(other_samples, min(mix_size, len(other_samples)))
    dataset = target_samples + mixed_samples
    random.shuffle(dataset)
    print(f"Curriculum epoch expert: {target_expert}, Samples: {len(dataset)} ({len(target_samples)} main + {len(mixed_samples)} mixed)")
    return dataset

# -------------------------------
# Training / evaluation functions
# -------------------------------
PAD_IDX = 0

def safe_collate_batch(expert_batch, device="cpu"):
    collated = collate_batch(expert_batch, device=device)
    if collated is None:
        return None
    inputs, fix_seqs, dur_seqs, full_word_embeddings, lengths = collated

    # Sanity check
    if fix_seqs.max() > embedding_size:
        raise ValueError(f"Fixation index out of range: {fix_seqs.max()} > {embedding_size}")

    return inputs, fix_seqs, dur_seqs, full_word_embeddings, lengths

def train_epoch(model, samples, optimizer, batch_size=16, alpha=0.5, device="cpu"):
    model.train()
    expert_loss_totals = {}
    expert_counts = {}

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        expert_batches = {}
        for s in batch_samples:
            lang = s["lang"].lower()
            eid = LANG_TO_EXPERT.get(lang)
            if eid is None:
                continue
            expert_batches.setdefault(eid, []).append(s)

        for eid, expert_batch in expert_batches.items():
            collated = safe_collate_batch(expert_batch, device=device)
            if collated is None:
                continue
            inputs, fix_seqs, dur_seqs, full_word_embeddings, lengths = collated

            optimizer.zero_grad()
            logits, dur_pred = model(inputs, fix_seqs, full_word_embeddings, lengths, eid)

            logits_flat = logits.view(-1, logits.size(-1))
            fix_flat = fix_seqs.view(-1)
            dur_flat = dur_seqs.view(-1)
            mask = (fix_flat != PAD_IDX) & (fix_flat >= 0)
            if not mask.any():
                continue

            loss_scanpath = torch.nn.functional.cross_entropy(logits_flat[mask], fix_flat[mask], ignore_index=PAD_IDX)
            loss_duration = torch.nn.functional.mse_loss(dur_pred.view(-1)[mask], dur_flat[mask])
            loss = alpha * loss_scanpath + (1 - alpha) * loss_duration
            loss.backward()
            optimizer.step()

            expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses

def evaluate(model, samples, batch_size=16, alpha=0.5, device="cpu"):
    model.eval()
    expert_loss_totals = {}
    expert_counts = {}

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            expert_batches = {}
            for s in batch_samples:
                lang = s["lang"].lower()
                eid = LANG_TO_EXPERT.get(lang)
                if eid is None:
                    continue
                expert_batches.setdefault(eid, []).append(s)

            for eid, expert_batch in expert_batches.items():
                collated = safe_collate_batch(expert_batch, device=device)
                if collated is None:
                    continue
                inputs, fix_seqs, dur_seqs, full_word_embeddings, lengths = collated

                logits, dur_pred = model(inputs, fix_seqs, full_word_embeddings, lengths, eid)
                logits_flat = logits.view(-1, logits.size(-1))
                fix_flat = fix_seqs.view(-1)
                dur_flat = dur_seqs.view(-1)

                mask = (fix_flat != PAD_IDX) & (fix_flat >= 0)
                if not mask.any():
                    continue

                loss_scanpath = torch.nn.functional.cross_entropy(logits_flat[mask], fix_flat[mask], ignore_index=PAD_IDX)
                loss_duration = torch.nn.functional.mse_loss(dur_pred.view(-1)[mask], dur_flat[mask])
                loss = alpha * loss_scanpath + (1 - alpha) * loss_duration

                expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
                expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses

# -------------------------------
# Model & optimizer
# -------------------------------
model = EyeExpertM(
    hidden_dim=256,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=embedding_size,
    window_size=8
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
alpha = 0.5

# -------------------------------
# Training loop
# -------------------------------
epochs = 1
batch_size = 8
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
    if USE_EXPERT_CURRICULUM:
        epoch_dataset = sample_curriculum_dataset(epoch, expert_datasets, CURRICULUM_MIX_RATIO)
    else:
        epoch_dataset = train_samples

    train_loss, train_expert = train_epoch(model, epoch_dataset, optimizer, batch_size, alpha, device=device)
    val_loss, val_expert = evaluate(model, val_samples, batch_size, alpha, device=device)
    test_loss, test_expert = evaluate(model, test_samples, batch_size, alpha, device=device)

    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Test loss: {test_loss:.4f}")
    print(f"Per-expert train losses: {train_expert}")
    print(f"Per-expert val losses:   {val_expert}")
    print(f"Per-expert test losses:  {test_expert}")

    torch.save(model.state_dict(), f"checkpoints/eyeexpert_epoch{epoch + 1}.pt")