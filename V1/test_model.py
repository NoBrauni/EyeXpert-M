import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model_definition import LANG_TO_EXPERT, collate_batch
import random

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Training function
# ===============================
def train_epoch(model, samples, optimizer, batch_size=8, alpha=0.5, device="cpu"):
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
            collated = collate_batch(expert_batch, device=device)
            if collated is None:
                continue

            inputs, dur_seqs, fix_seqs, _, lengths, mask = collated
            inputs = inputs.to(device)
            fix_seqs = fix_seqs.to(device).long()
            dur_seqs = dur_seqs.to(device)

            model.train()
            optimizer.zero_grad()
            logits, dur_pred = model(inputs, fix_seqs, _, lengths, eid)

            # Scanpath loss (CrossEntropy over next word)
            # logits: [B, seq_len, num_words], targets: fix_seqs [B, seq_len]
            loss_scanpath = nn.CrossEntropyLoss()(
                logits[mask.bool()].view(-1, logits.size(-1)),
                fix_seqs[mask.bool()].view(-1)
            )

            # Duration loss (MSE)
            loss_duration = nn.MSELoss()(dur_pred[mask.bool()], dur_seqs[mask.bool()])

            # Combined loss
            loss = alpha * loss_scanpath + (1 - alpha) * loss_duration

            loss.backward()
            optimizer.step()

            # Track losses per expert
            expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid]
                         for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses


# ===============================
# Evaluation function
# ===============================
def evaluate(model, samples, batch_size=8, alpha=0.5, device="cpu"):
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
                collated = collate_batch(expert_batch, device=device)
                if collated is None:
                    continue

                inputs, dur_seqs, fix_seqs, _, lengths, mask = collated
                inputs = inputs.to(device)
                fix_seqs = fix_seqs.to(device).long()
                dur_seqs = dur_seqs.to(device)

                logits, dur_pred = model(inputs, fix_seqs, _, lengths, eid)

                # Scanpath loss
                loss_scanpath = nn.CrossEntropyLoss()(
                    logits[mask.bool()].view(-1, logits.size(-1)),
                    fix_seqs[mask.bool()].view(-1)
                )

                # Duration loss
                loss_duration = nn.MSELoss()(dur_pred[mask.bool()], dur_seqs[mask.bool()])

                # Combined
                loss = alpha * loss_scanpath + (1 - alpha) * loss_duration

                expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss.item()
                expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid] / expert_counts[eid]
                         for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses