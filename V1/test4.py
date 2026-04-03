import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from model_definition import MECODataset, EyeExpertM, LANG_TO_EXPERT, batch_precompute_embeddings, collate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Loss functions
# -------------------------------
ce_loss = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padded tokens
mse_loss = nn.MSELoss(reduction='sum')            # sum to later normalize

# -------------------------------
# Training function
# -------------------------------
def train_epoch(model, samples, optimizer, batch_size=8, alpha=0.5, device="cpu"):
    random.shuffle(samples)
    expert_loss_totals = {}
    expert_counts = {}

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]

        # Organize per expert
        expert_batches = {}
        for s in batch_samples:
            lang = s["lang"].lower()
            eid = LANG_TO_EXPERT.get(lang)
            if eid is None:
                continue
            expert_batches.setdefault(eid, []).append(s)

        for eid, expert_batch in expert_batches.items():
            collated = collate_batch(expert_batch, device=device)
            if collated is None:
                continue

            inputs, fix_seqs, dur_seqs, full_word_embeds, lengths = collated  # note: no mask
            optimizer.zero_grad()
            model.train()

            logits, dur_pred = model(inputs, fix_seqs, full_word_embeds, lengths, eid)

            # Pad targets with -100 to ignore padding for CE
            padded_fix = fix_seqs.clone()
            max_len = fix_seqs.size(1)
            for idx, l in enumerate(lengths):
                if l < max_len:
                    padded_fix[idx, l:] = -100  # mark padding

            loss_scanpath = ce_loss(logits.view(-1, logits.size(-1)), padded_fix.view(-1))

            # For duration, mask padded positions
            mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
            loss_duration = mse_loss(dur_pred[mask], dur_seqs[mask]) / mask.sum()  # normalize

            loss = alpha * loss_scanpath + (1 - alpha) * loss_duration
            loss.backward()
            optimizer.step()

            expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(model, samples, batch_size=8, alpha=0.5, device="cpu"):
    model.eval()
    expert_loss_totals = {}
    expert_counts = {}

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]

            expert_batches = {}
            for s in batch_samples:
                lang = s["lang"].lower()
                eid = LANG_TO_EXPERT.get(lang)
                if eid is None:
                    continue
                expert_batches.setdefault(eid, []).append(s)

            for eid, expert_batch in expert_batches.items():
                collated = collate_batch(expert_batch, device=device)
                if collated is None:
                    continue

                inputs, fix_seqs, dur_seqs, full_word_embeds, lengths = collated
                logits, dur_pred = model(inputs, fix_seqs, full_word_embeds, lengths, eid)

                padded_fix = fix_seqs.clone()
                max_len = fix_seqs.size(1)
                for idx, l in enumerate(lengths):
                    if l < max_len:
                        padded_fix[idx, l:] = -100

                loss_scanpath = ce_loss(logits.view(-1, logits.size(-1)), padded_fix.view(-1))
                mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
                loss_duration = mse_loss(dur_pred[mask], dur_seqs[mask]) / mask.sum()

                loss = alpha * loss_scanpath + (1 - alpha) * loss_duration
                expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
                expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid] for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses

