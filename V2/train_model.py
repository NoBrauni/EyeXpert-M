# ------------------------------- Imports -------------------------------
import os
import json
import random
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from embeddings import generate_universal_sentence_embeddings
from model_definition import EyeExpertM, collate_batch, LANG_TO_EXPERT

# ------------------------------- Config -------------------------------
DATA_PATH = "../meco_train.json"
EMBEDDING_CACHE = "full_sentence_embeddings.pkl"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 10

ALPHA = 0.79

USE_EXPERT_CURRICULUM = True
CURRICULUM_EPOCHS = 5
CURRICULUM_MIX_RATIO = 0.2

random.seed(42)
os.makedirs("checkpoints", exist_ok=True)

# ------------------------------- Step 1: Load JSON dataset -------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

all_samples = []
for s in data:
    if not all(k in s for k in ["sentence", "scanpath", "durations", "words"]):
        continue

    # HARD FIX: enforce alignment
    min_len = min(len(s["scanpath"]), len(s["durations"]))
    if min_len < 2:
        continue

    s["scanpath"] = s["scanpath"][:min_len]
    s["durations"] = s["durations"][:min_len]

    all_samples.append(s)

print(f"Loaded {len(all_samples)} valid JSON samples.")

# ------------------------------- Step 2: Embeddings -------------------------------
if not os.path.exists(EMBEDDING_CACHE):
    print("Generating embedding cache...")
    generate_universal_sentence_embeddings(
        pickle_dir="..",
        cache_path=EMBEDDING_CACHE,
        batch_size=4,
        device=DEVICE,
        max_sentences=None
    )

import pickle
with open(EMBEDDING_CACHE, "rb") as f:
    embedding_cache = pickle.load(f)

print(f"Loaded embedding cache: {len(embedding_cache)} sentences")

# ------------------------------- Step 3: Split -------------------------------
def split_dataset(samples):
    sentences = list({s["sentence"] for s in samples})
    random.shuffle(sentences)

    n = len(sentences)
    train_s = set(sentences[:int(TRAIN_RATIO * n)])
    val_s = set(sentences[int(TRAIN_RATIO * n):int((TRAIN_RATIO + VAL_RATIO) * n)])
    test_s = set(sentences[int((TRAIN_RATIO + VAL_RATIO) * n):])

    train = [s for s in samples if s["sentence"] in train_s]
    val = [s for s in samples if s["sentence"] in val_s]
    test = [s for s in samples if s["sentence"] in test_s]

    return train, val, test


train_samples, val_samples, test_samples = split_dataset(all_samples)

print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

# ------------------------------- Step 4: Expert grouping -------------------------------
expert_datasets = {}
for s in train_samples:
    lang = s["lang"].lower()
    eid = LANG_TO_EXPERT.get(lang, 0)
    expert_datasets.setdefault(eid, []).append(s)


def get_expert(sample, epoch):
    if USE_EXPERT_CURRICULUM and epoch < CURRICULUM_EPOCHS:
        return epoch % len(expert_datasets)
    return LANG_TO_EXPERT.get(sample["lang"].lower(), 0)


def split_by_expert(samples, epoch):
    groups = defaultdict(list)
    for s in samples:
        groups[get_expert(s, epoch)].append(s)
    return groups

# ------------------------------- Step 5: Model -------------------------------
max_fix = max(max(s["scanpath"]) for s in all_samples if len(s["scanpath"]) > 0)
max_seq_len = max_fix + 1

model = EyeExpertM(
    hidden_dim=512,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=max_seq_len,
    window_size=11,
    attention_type=None
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=4e-4)

# ------------------------------- Loss -------------------------------
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
mse_loss = torch.nn.MSELoss()

# ------------------------------- Train -------------------------------
def train_epoch(epoch):
    model.train()
    groups = split_by_expert(train_samples, epoch)

    total_loss = 0
    total = 0

    for expert_id, samples in groups.items():
        random.shuffle(samples)

        for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Expert {expert_id}"):
            batch_samples = samples[i:i + BATCH_SIZE]

            batch = collate_batch(
                batch_samples,
                embedding_cache=embedding_cache,
                device=DEVICE
            )

            if batch is None:
                continue

            inputs, fix, dur, full_emb, lengths, pad_idx = batch

            optimizer.zero_grad()

            logits, dur_pred = model(
                inputs, fix, full_emb, lengths, expert_id
            )

            loss_fix = ce_loss(
                logits.view(-1, logits.size(-1)),
                fix.view(-1)
            )

            loss_dur = mse_loss(
                dur_pred.view(-1),
                dur.view(-1)
            )

            loss = ALPHA * loss_fix + (1 - ALPHA) * loss_dur

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_samples)
            total += len(batch_samples)

    return total_loss / max(1, total)


# ------------------------------- Eval -------------------------------
@torch.no_grad()
def evaluate(samples, epoch, name="Eval"):
    model.eval()
    groups = split_by_expert(samples, epoch)

    total_loss = 0
    total = 0

    for expert_id, samples in groups.items():
        for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"{name} {expert_id}"):
            batch = collate_batch(
                samples[i:i + BATCH_SIZE],
                embedding_cache=embedding_cache,
                device=DEVICE
            )

            if batch is None:
                continue

            inputs, fix, dur, full_emb, lengths, pad_idx = batch

            logits, dur_pred = model(inputs, fix, full_emb, lengths, expert_id)

            loss_fix = ce_loss(logits.view(-1, logits.size(-1)), fix.view(-1))
            loss_dur = mse_loss(dur_pred.view(-1), dur.view(-1))

            loss = ALPHA * loss_fix + (1 - ALPHA) * loss_dur

            total_loss += loss.item() * len(samples)
            total += len(samples)

    return total_loss / max(1, total)

# ------------------------------- Training loop -------------------------------
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1} ===")

    train_loss = train_epoch(epoch)
    val_loss = evaluate(val_samples, epoch, "Val")
    test_loss = evaluate(test_samples, epoch, "Test")

    print(f"\nTrain: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/eyeexpert_{epoch}.pt")
    print("Saved checkpoint")