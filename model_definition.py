import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
import os
import pickle

# ===============================
# Language to Expert mapping
# ===============================
LANG_TO_EXPERT = {
    # Expert 0 — Germanic
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0, "ic": 0,
    # Expert 1 — Nordic
    "no": 1, "da": 1, "se": 1,
    # Expert 2 — Romance
    "sp": 2, "sp_ch": 2, "it": 2, "bp": 2,
    # Expert 3 — Slavic
    "ru": 3, "ru_mo": 3,
    # Expert 4 — Uralic
    "fi": 4, "ee": 4,
}

# ===============================
# Dataset
# ===============================
class MECODataset:
    def __init__(self, csv_path=None, min_dur=60):
        self.samples = []
        if csv_path:
            df = pd.read_csv(csv_path)
            df = df[(df["blink"] == 0) & (df["dur"] >= min_dur)]
            df = df.sort_values(["subid", "unique_sentence_id", "fix_index"])
            grouped = df.groupby(["subid", "unique_sentence_id"])
            for (_, _), group in grouped:
                words = group.drop_duplicates("wordnum").sort_values("wordnum")["word"].tolist()
                words = [str(w) for w in words if isinstance(w, str) and w.strip()]
                sentence = " ".join(words)
                fix_seq = group["wordnum"].astype(int).tolist()
                dur_seq = group["dur"].tolist()
                if len(fix_seq) > 1:
                    lang_code = str(group["lang"].iloc[0]).strip().lower()
                    self.samples.append({
                        "sentence": sentence,
                        "words": words,
                        "fix_seq": fix_seq,
                        "dur_seq": dur_seq,
                        "lang": lang_code
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ===============================
# Encoder & Precompute embeddings
# ===============================
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

def batch_precompute_embeddings(samples, batch_size=16, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_sentences = [s["sentence"] for s in samples]
    all_words = [s["words"] for s in samples]

    for start_idx in range(0, len(samples), batch_size):
        batch_sentences = all_sentences[start_idx:start_idx+batch_size]
        batch_words = all_words[start_idx:start_idx+batch_size]

        encoding = tokenizer(batch_sentences, return_tensors="pt", padding=True,
                             truncation=True, return_offsets_mapping=True)
        with torch.no_grad():
            outputs = encoder(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
        hidden_states = outputs.last_hidden_state

        for i, sample_idx in enumerate(range(start_idx, start_idx+len(batch_sentences))):
            sample = samples[sample_idx]
            offsets = encoding["offset_mapping"][i]
            word_embeddings = []

            for word in sample["words"]:
                vecs = []
                for j, (start, end) in enumerate(offsets):
                    if start == 0 and end == 0:
                        continue
                    token_text = batch_sentences[i][start:end]
                    if token_text.strip() == word.strip():
                        vecs.append(hidden_states[i, j])
                if vecs:
                    word_embeddings.append(torch.stack(vecs).mean(dim=0))
                else:
                    word_embeddings.append(torch.zeros(hidden_states.size(-1)))
            sample["embeddings"] = torch.stack(word_embeddings)

    if cache_path:
        print(f"Saving embeddings cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)

    return samples

# ===============================
# MoE Decoder
# ===============================
class EyeExpertM(nn.Module):
    def __init__(self, hidden_dim=256, encoder_dim=768, n_experts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.GRU(input_size=encoder_dim, hidden_size=hidden_dim, batch_first=True)
            for _ in range(n_experts)
        ])
        self.output_layer = nn.Linear(hidden_dim, encoder_dim)
        self.duration_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, full_word_embeddings, lengths, expert_id):
        expert = self.experts[expert_id]
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = expert(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        proj = self.output_layer(outputs)
        logits = torch.matmul(proj, full_word_embeddings.transpose(1, 2))
        dur_pred = self.duration_layer(outputs).squeeze(-1)
        return logits, dur_pred

# ===============================
# Collate
# ===============================
def collate_batch(samples):
    batch_inputs, fix_seqs, dur_seqs, lengths = [], [], [], []
    full_word_embeddings = []

    for sample in samples:
        word_embeddings = sample["embeddings"]
        unique_wordnums = sorted(set(sample["fix_seq"]))
        wordnum_to_idx = {num: i for i, num in enumerate(unique_wordnums)}
        fix_indices = [wordnum_to_idx[n] for n in sample["fix_seq"][:-1]]
        if max(fix_indices) >= len(word_embeddings):
            continue
        batch_inputs.append(word_embeddings[torch.tensor(fix_indices)])
        dur_seqs.append(torch.tensor(sample["dur_seq"][1:], dtype=torch.float)/1000.0)
        fix_seqs.append(torch.tensor(fix_indices))
        lengths.append(len(fix_indices))
        full_word_embeddings.append(word_embeddings)

    if not batch_inputs:
        return None

    padded_inputs = pad_sequence(batch_inputs, batch_first=True)
    padded_durs = pad_sequence(dur_seqs, batch_first=True)
    padded_fixes = pad_sequence(fix_seqs, batch_first=True)
    padded_full_words = pad_sequence(full_word_embeddings, batch_first=True)
    return padded_inputs, padded_fixes, padded_durs, padded_full_words, lengths

# ===============================
# Training
# ===============================
def train_batch(model, inputs, targets, durs, full_words, lengths, optimizer, expert_id, alpha=0.5):
    model.train()
    optimizer.zero_grad()
    logits, dur_pred = model(inputs, full_words, lengths, expert_id)
    mask = torch.zeros_like(durs)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    loss_words = F.cross_entropy(logits[mask.bool()].view(-1, logits.size(-1)),
                                 targets[mask.bool()].view(-1))
    loss_dur = F.mse_loss(dur_pred[mask.bool()], durs[mask.bool()])
    loss = alpha * loss_words + (1 - alpha) * loss_dur
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, dataset, optimizer, batch_size=8, alpha=0.5):
    random.shuffle(dataset.samples)
    expert_loss_totals = {}
    expert_counts = {}

    for i in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[i:i+batch_size]

        expert_batches = {}
        for s in batch_samples:
            lang = str(s["lang"]).strip().lower()
            eid = LANG_TO_EXPERT.get(lang)
            if eid is None:
                continue
            expert_batches.setdefault(eid, []).append(s)

        for eid, batch in expert_batches.items():
            collated = collate_batch(batch)
            if collated is None:
                continue
            inputs, targets, durs, full_words, lengths = collated

            loss = train_batch(model, inputs, targets, durs, full_words, lengths, optimizer, eid, alpha)
            expert_loss_totals[eid] = expert_loss_totals.get(eid, 0.0) + loss
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

    expert_avg_losses = {eid: expert_loss_totals[eid]/expert_counts[eid]
                         for eid in expert_loss_totals}
    global_avg = sum(expert_loss_totals.values()) / sum(expert_counts.values())
    return global_avg, expert_avg_losses

# ===============================
# Verification
# ===============================
def verify_expert_model(model, dataset, samples_per_expert=5):
    expert_to_langs = {}
    for lang, eid in LANG_TO_EXPERT.items():
        expert_to_langs.setdefault(eid, []).append(lang)

    for expert_id, langs in expert_to_langs.items():
        print(f"\n--- Verifying Expert {expert_id} (langs: {langs}) ---")
        expert_samples = [s for s in dataset.samples if s["lang"] in langs]
        if not expert_samples:
            print("No samples for this expert!")
            continue

        for i, sample in enumerate(expert_samples[:samples_per_expert]):
            collated = collate_batch([sample])
            if collated is None:
                continue
            inputs, targets, durs, full_words, lengths = collated
            with torch.no_grad():
                logits, dur_pred = model(inputs, full_words, lengths, expert_id)
            pred_fix = logits.argmax(dim=-1)
            print(f"\nSample {i+1}:")
            print("Target indices:", targets[0][:10].tolist())
            print("Predicted indices:", pred_fix[0][:10].tolist())
            print("True durations:", durs[0][:10].tolist())
            print("Pred durations:", dur_pred[0][:10].tolist())