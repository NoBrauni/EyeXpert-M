# ------------------------------- Imports -------------------------------
import os
import random
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from embeddings import generate_universal_sentence_embeddings, normalize_sentence
from model_definition import EyeExpertM, collate_batch, LANG_TO_EXPERT, PAD_IDX

# ------------------------------- Config -------------------------------
DATA_DIR = "datasets"
EMBEDDING_CACHE = "full_sentence_embeddings.pkl"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LEAVE_OUT_LANGUAGE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 10
ALPHA = 0.7956699876598292

USE_EXPERT_CURRICULUM = True
CURRICULUM_EPOCHS = 5
CURRICULUM_MIX_RATIO = 0.2

random.seed(42)
os.makedirs("checkpoints", exist_ok=True)

# ------------------------------- Step 1: Load dataset -------------------------------
all_samples = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".pkl"):
        with open(os.path.join(DATA_DIR, fname), "rb") as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
all_samples = [s for s in all_samples if s.get("sentence") is not None]
print(f"Loaded {len(all_samples)} samples.")

# ------------------------------- Step 2: Load or generate embeddings -------------------------------
if not os.path.exists(EMBEDDING_CACHE):
    print("No embeddings found. Generating embeddings...")
    generate_universal_sentence_embeddings(
        pickle_dir=DATA_DIR,
        cache_path=EMBEDDING_CACHE,
        batch_size=4,
        device=DEVICE,
        max_sentences=None
    )

with open(EMBEDDING_CACHE, "rb") as f:
    embedding_cache = pickle.load(f)
print(f"✅ Loaded embedding cache ({len(embedding_cache)})")


# ------------------------------- Step 3: Dataset splitting -------------------------------
def split_dataset(samples):
    if LEAVE_OUT_LANGUAGE:
        train_samples = [s for s in samples if s["lang"] != LEAVE_OUT_LANGUAGE]
        heldout = [s for s in samples if s["lang"] == LEAVE_OUT_LANGUAGE]
        n = len(heldout)
        val_samples = heldout[:n // 2]
        test_samples = heldout[n // 2:]
        return train_samples, val_samples, test_samples

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


train_samples, val_samples, test_samples = split_dataset(all_samples)
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# ------------------------------- Step 4: Expert datasets -------------------------------
expert_datasets = {}
for s in train_samples:
    lang = s["lang"].lower()
    eid = LANG_TO_EXPERT.get(lang)
    if eid is not None:
        expert_datasets.setdefault(eid, []).append(s)


def sample_curriculum_dataset(epoch, expert_datasets, mix_ratio=0.2):
    target_expert = epoch % len(expert_datasets)
    target_samples = expert_datasets[target_expert]
    other_samples = [s for eid, data in expert_datasets.items() if eid != target_expert for s in data]
    mix_size = int(len(target_samples) * mix_ratio)
    mixed_samples = random.sample(other_samples, min(mix_size, len(other_samples)))
    dataset = target_samples + mixed_samples
    random.shuffle(dataset)
    print(f"Curriculum epoch expert: {target_expert}, Samples: {len(dataset)}")
    return dataset, target_expert


# ------------------------------- Step 4b: Expert selection helper -------------------------------
def get_expert_id(sample, epoch):
    """Select expert per sample based on curriculum or mixed training."""
    # Curriculum phase
    if USE_EXPERT_CURRICULUM and epoch < CURRICULUM_EPOCHS:
        return epoch % len(expert_datasets)
    # Mixed phase
    lang = sample.get("lang", "en").lower()
    return LANG_TO_EXPERT.get(lang, 0)


def split_by_expert(samples, epoch):
    groups = defaultdict(list)
    for s in samples:
        eid = get_expert_id(s, epoch)
        groups[eid].append(s)
    return groups


# ------------------------------- Step 5: Model -------------------------------
max_fix_idx = max([max(s["scanpath"]) for s in all_samples])
embedding_size = max_fix_idx + 1

model = EyeExpertM(
    hidden_dim=512,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=embedding_size,
    window_size=11,
    n_layers=1,
    attention_type=None
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.00048745727789273525)


# ------------------------------- Step 6: Train / Evaluate -------------------------------
def train_epoch(model, samples, optimizer, batch_size=BATCH_SIZE, alpha=ALPHA, device=DEVICE, embedding_cache=None,
                epoch=0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    expert_groups = split_by_expert(samples, epoch)

    for expert_id, expert_samples in expert_groups.items():
        n_expert_samples = len(expert_samples)
        random.shuffle(expert_samples)
        pbar = tqdm(range(0, n_expert_samples, batch_size), desc=f"Training Expert {expert_id}", unit="batch")
        for i in pbar:
            batch_samples = expert_samples[i:i + batch_size]
            batch = collate_batch(batch_samples, device=device, embedding_cache=embedding_cache)
            if batch is None:
                continue
            inputs, fix_targets, dur_targets, full_word_embeddings, lengths, pad_idx = batch

            optimizer.zero_grad()
            logits, dur_pred = model(inputs, fix_targets, full_word_embeddings, lengths, expert_id)

            loss_fix = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=pad_idx
            )
            loss_dur = torch.nn.functional.mse_loss(dur_pred.view(-1), dur_targets.view(-1))
            loss = alpha * loss_fix + (1 - alpha) * loss_dur

            # Weight loss by number of samples in expert batch
            weighted_loss = loss * len(batch_samples)
            weighted_loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_samples)
            total_samples += len(batch_samples)
            pbar.set_postfix({"batch_loss": total_loss / max(1, total_samples)})

    return total_loss / max(1, total_samples)


def evaluate(model, samples, batch_size=BATCH_SIZE, alpha=ALPHA, device=DEVICE, embedding_cache=None, epoch=0,
             desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    expert_groups = split_by_expert(samples, epoch)

    with torch.no_grad():
        for expert_id, expert_samples in expert_groups.items():
            n_expert_samples = len(expert_samples)
            pbar = tqdm(range(0, n_expert_samples, batch_size), desc=f"{desc} Expert {expert_id}", unit="batch")
            for i in pbar:
                batch_samples = expert_samples[i:i + batch_size]
                batch = collate_batch(batch_samples, device=device, embedding_cache=embedding_cache)
                if batch is None:
                    continue
                inputs, fix_targets, dur_targets, full_word_embeddings, lengths, pad_idx = batch

                logits, dur_pred = model(inputs, fix_targets, full_word_embeddings, lengths, expert_id)
                loss_fix = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=pad_idx
                )
                loss_dur = torch.nn.functional.mse_loss(dur_pred.view(-1), dur_targets.view(-1))
                loss = alpha * loss_fix + (1 - alpha) * loss_dur

                # Weight loss by number of samples in batch
                total_loss += loss.item() * len(batch_samples)
                total_samples += len(batch_samples)
                pbar.set_postfix({"batch_loss": total_loss / max(1, total_samples)})

    return total_loss / max(1, total_samples)


# ------------------------------- Step 7: Training loop -------------------------------
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")

    if USE_EXPERT_CURRICULUM and epoch < CURRICULUM_EPOCHS:
        epoch_dataset, _ = sample_curriculum_dataset(epoch, expert_datasets, CURRICULUM_MIX_RATIO)
    else:
        epoch_dataset = train_samples  # full dataset for mixed expert training

    # --- Train ---
    train_loss = train_epoch(
        model,
        epoch_dataset,
        optimizer,
        embedding_cache=embedding_cache,
        epoch=epoch
    )

    # --- Validate ---
    val_loss = evaluate(
        model,
        val_samples,
        embedding_cache=embedding_cache,
        epoch=epoch,
        desc="Validation"
    )

    # --- Test ---
    test_loss = evaluate(
        model,
        test_samples,
        embedding_cache=embedding_cache,
        epoch=epoch,
        desc="Test"
    )

    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")

    # --- Save checkpoint ---
    torch.save(model.state_dict(), f"checkpoints/eyeexpert_epoch{epoch + 1}.pt")
    print("✅ Saved checkpoint")