import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random
import os

from model_definition_3 import MecoDataset, EyeXpertModel
from transformers import XLMRobertaTokenizer

# ----------------------
# Settings
# ----------------------
INITIAL_CSVS = ["processed_data/meco_l1_en_processed.csv"]  # Start with English
NEW_CSVS = ["processed_data/meco_l2_fr_processed.csv"]      # Add French later
MAX_LEN = 64
BATCH_SIZE = 8
SUBSET_SIZE = 50
NUM_EXPERTS = 4
EPOCHS = 5
DEVICE = "cpu"
CHECKPOINT_PATH = "checkpoints/eyexpert_checkpoint.pth"

# ----------------------
# Tokenizer
# ----------------------
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# ----------------------
# Helper function to load datasets
# ----------------------
def load_language_datasets(csv_files, starting_lang_idx=0):
    datasets = []
    dur_stats = {}
    lang_idx = starting_lang_idx

    for csv_file in csv_files:
        ds = MecoDataset(csv_file, tokenizer, max_len=MAX_LEN)
        print(f"Dataset {csv_file} duration mean/std: {ds.dur_mean:.2f}, {ds.dur_std:.2f}")
        dur_stats[lang_idx] = (ds.dur_mean, ds.dur_std)

        # Attach lang_id
        for i in range(len(ds)):
            ds[i]['lang_id'] = lang_idx

        datasets.append(ds)
        lang_idx += 1

    return datasets, dur_stats, lang_idx

# ----------------------
# Initial load
# ----------------------
all_datasets, DUR_STATS, next_lang_idx = load_language_datasets(INITIAL_CSVS)
full_dataset = ConcatDataset(all_datasets)

# Random subset for CPU
subset_indices = random.sample(range(len(full_dataset)), min(SUBSET_SIZE, len(full_dataset)))
subset_dataset = Subset(full_dataset, subset_indices)

def custom_collate(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'word_map': [item['word_map'] for item in batch],
        'fix_seq': [item['fix_seq'] for item in batch],
        'dur_seq': [item['dur_seq'] for item in batch],
        'lang_id': torch.tensor([item['lang_id'] for item in batch], dtype=torch.long)
    }

train_loader = DataLoader(
    subset_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate
)

# ----------------------
# Model & optimizer
# ----------------------
model = EyeXpertModel(
    num_languages=next_lang_idx,
    num_experts=NUM_EXPERTS
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
ce_loss_fn = nn.CrossEntropyLoss()

def duration_nll(d_mean, d_logvar, target):
    var = torch.exp(d_logvar)
    nll = 0.5 * (
        (target - d_mean) ** 2 / var
        + d_logvar
        + torch.log(torch.tensor(2 * 3.1415926, device=d_mean.device))
    )
    return nll.mean()

# ----------------------
# Load checkpoint if exists
# ----------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

# ----------------------
# Training loop
# ----------------------
for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_loss = 0.0
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        word_maps = [wm.to(DEVICE) for wm in batch['word_map']]
        fix_seqs = [fs.to(DEVICE) for fs in batch['fix_seq']]
        dur_seqs = [ds.to(DEVICE) for ds in batch['dur_seq']]
        lang_ids = batch['lang_id'].to(DEVICE)

        fix_out, dur_mean_out, dur_logvar_out, gate_trace, lengths = model(
            input_ids,
            attention_mask,
            word_maps,
            lang_ids,
            target_fix_seq=fix_seqs,
            target_dur_seq=dur_seqs
        )

        batch_fix_loss = 0.0
        batch_dur_loss = 0.0

        for b_idx in range(len(fix_out)):
            steps = len(fix_out[b_idx])
            lang_id = lang_ids[b_idx].item()
            dur_mean_val, dur_std_val = DUR_STATS[lang_id]

            for t in range(steps):
                fix_logits = fix_out[b_idx][t]
                dur_mean = dur_mean_out[b_idx][t]
                dur_logvar = dur_logvar_out[b_idx][t]

                target_fix = fix_seqs[b_idx][t].clamp(0, lengths[b_idx] - 1)
                target_dur = (dur_seqs[b_idx][t] - dur_mean_val) / dur_std_val

                batch_fix_loss += ce_loss_fn(
                    fix_logits.unsqueeze(0),
                    target_fix.unsqueeze(0)
                )
                batch_dur_loss += duration_nll(dur_mean, dur_logvar, target_dur)

        loss = batch_fix_loss + batch_dur_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 5 == 0:
            dur_loss_ms = batch_dur_loss.item() * dur_std_val
            print(f"Batch {batch_idx}/{len(train_loader)}  Loss: {loss.item():.4f}")
            print(f"  Fix Loss: {batch_fix_loss.item():.2f} | Dur Loss (norm): {batch_dur_loss.item():.2f} | Dur Loss (ms approx): {dur_loss_ms:.2f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)

# ----------------------
# Incremental language addition
# ----------------------
if NEW_CSVS:
    new_datasets, new_stats, next_lang_idx = load_language_datasets(NEW_CSVS, starting_lang_idx=next_lang_idx)
    DUR_STATS.update(new_stats)
    all_datasets.extend(new_datasets)
    full_dataset = ConcatDataset(all_datasets)

    subset_indices = random.sample(range(len(full_dataset)), min(SUBSET_SIZE, len(full_dataset)))
    subset_dataset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    print("\nNew languages added. Continue training from checkpoint next run.")