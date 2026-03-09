import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model_definition import LANG_TO_EXPERT
import random

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Collate function
# ===============================
def collate_batch(samples):
    batch_inputs, fix_seqs, dur_seqs, lengths = [], [], [], []
    full_word_embeddings = []

    for sample in samples:
        word_embeddings = sample["embeddings"]

        # Flatten all fixations per word
        unique_wordnums = sorted(set([num for seq in sample["fix_seq"] for num in seq]))
        wordnum_to_idx = {num: i for i, num in enumerate(unique_wordnums)}
        fix_indices = [wordnum_to_idx[n] for seq in sample["fix_seq"] for n in seq[:-1]]

        if len(fix_indices) == 0 or max(fix_indices) >= len(word_embeddings):
            continue

        batch_inputs.append(word_embeddings[torch.tensor(fix_indices)])
        dur_seqs.append(torch.tensor([d for seq in sample["dur_seq"] for d in seq[1:]], dtype=torch.float) / 1000.0)
        fix_seqs.append(torch.tensor(fix_indices))
        lengths.append(len(fix_indices))
        full_word_embeddings.append(word_embeddings)

    if not batch_inputs:
        return None

    padded_inputs = pad_sequence(batch_inputs, batch_first=True)         # [batch, seq_len, encoder_dim]
    padded_fixes = pad_sequence(fix_seqs, batch_first=True)              # [batch, seq_len] -> for scanpath
    padded_durs = pad_sequence(dur_seqs, batch_first=True)               # [batch, seq_len]
    padded_full_words = pad_sequence(full_word_embeddings, batch_first=True)  # [batch, num_words, encoder_dim]

    return padded_inputs, padded_fixes, padded_durs, padded_full_words, lengths

# ===============================
# Training function
# ===============================
def train_epoch(model, samples, optimizer, batch_size=8, alpha=0.5):
    random.shuffle(samples)
    expert_loss_totals = {}
    expert_counts = {}

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]

        # Organize samples per expert
        expert_batches = {}
        for s in batch_samples:
            lang = s["lang"].lower()
            eid = LANG_TO_EXPERT.get(lang)
            if eid is None:
                continue
            expert_batches.setdefault(eid, []).append(s)

        # Train each expert
        for eid, expert_batch in expert_batches.items():
            collated = collate_batch(expert_batch)
            if collated is None:
                continue

            inputs, fix_seqs, dur_seqs, full_words, lengths = collated
            inputs = inputs.to(device)
            fix_seqs = fix_seqs.to(device)
            dur_seqs = dur_seqs.to(device)
            full_words = full_words.to(device)

            model.train()
            optimizer.zero_grad()
            logits, dur_pred = model(inputs, fix_seqs, full_words, lengths, eid)

            # Mask padded positions
            mask = torch.zeros_like(dur_seqs)
            for j, l in enumerate(lengths):
                mask[j, :l] = 1

            targets = fix_seqs
            loss_words = nn.CrossEntropyLoss()(
                logits[mask.bool()].view(-1, logits.size(-1)),
                targets[mask.bool()].view(-1)
            )
            loss_dur = nn.MSELoss()(dur_pred[mask.bool()], dur_seqs[mask.bool()])
            loss = alpha * loss_words + (1 - alpha) * loss_dur

            loss.backward()
            optimizer.step()

            expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())

    return global_avg, expert_avg_losses

# ===============================
# Evaluation function
# ===============================
def evaluate(model, samples, batch_size=8, alpha=0.5):
    model.eval()
    expert_loss_totals = {}
    expert_counts = {}

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]

            # Organize samples per expert
            expert_batches = {}
            for s in batch_samples:
                lang = s["lang"].lower()
                eid = LANG_TO_EXPERT.get(lang)
                if eid is None:
                    continue
                expert_batches.setdefault(eid, []).append(s)

            # Evaluate each expert
            for eid, expert_batch in expert_batches.items():
                collated = collate_batch(expert_batch)
                if collated is None:
                    continue

                inputs, fix_seqs, dur_seqs, full_words, lengths = collated
                inputs = inputs.to(device)
                fix_seqs = fix_seqs.to(device)
                dur_seqs = dur_seqs.to(device)
                full_words = full_words.to(device)

                logits, dur_pred = model(inputs, fix_seqs, full_words, lengths, eid)

                # Mask padded positions
                mask = torch.zeros_like(dur_seqs)
                for j, l in enumerate(lengths):
                    mask[j, :l] = 1

                targets = fix_seqs
                loss_words = nn.CrossEntropyLoss()(
                    logits[mask.bool()].view(-1, logits.size(-1)),
                    targets[mask.bool()].view(-1)
                )
                loss_dur = nn.MSELoss()(dur_pred[mask.bool()], dur_seqs[mask.bool()])
                loss = alpha * loss_words + (1 - alpha) * loss_dur

                expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
                expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())

    return global_avg, expert_avg_losses