import torch
import torch.nn as nn
import pandas as pd
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
            for (subid, sent_id), group in grouped:

                word_table = group[["wordnum", "word"]].drop_duplicates().sort_values("wordnum")
                words = [str(w) for w in word_table["word"] if isinstance(w, str) and w.strip()]
                sentence_text = " ".join(words)

                fix_seq_per_word = []
                dur_seq_per_word = []

                for wordnum, word_group in group.groupby("wordnum"):
                    fix_seq_per_word.append(word_group["fix_index"].astype(int).tolist())
                    dur_seq_per_word.append(word_group["dur"].tolist())

                if sum(len(x) for x in fix_seq_per_word) > 1:
                    lang_code = str(group["lang"].iloc[0]).strip().lower()
                    self.samples.append({
                        "sentence": sentence_text,
                        "words": words,
                        "fix_seq": fix_seq_per_word,
                        "dur_seq": dur_seq_per_word,
                        "lang": lang_code,
                        "reader": subid,
                        "sentence_id": sent_id
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
    def __init__(self, hidden_dim=256, encoder_dim=768, n_experts=5, max_seq_len=200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.GRU(input_size=encoder_dim + 32, hidden_size=hidden_dim, batch_first=True)
            for _ in range(n_experts)
        ])
        self.output_layer = nn.Linear(hidden_dim, encoder_dim)
        self.duration_layer = nn.Linear(hidden_dim, 1)

        # Scanpath embedding: encode the fixated word positions as learnable embeddings
        self.scanpath_embedding = nn.Embedding(max_seq_len, 32)

    def forward(self, inputs, fix_seq, full_word_embeddings, lengths, expert_id):
        scanpath_embeds = self.scanpath_embedding(fix_seq)
        rnn_inputs = torch.cat([inputs, scanpath_embeds], dim=-1)

        # Run through the expert GRU
        expert = self.experts[expert_id]
        packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = expert(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        proj = self.output_layer(outputs)
        logits = torch.matmul(proj, full_word_embeddings.transpose(1, 2))
        dur_pred = self.duration_layer(outputs).squeeze(-1)
        return logits, dur_pred

# ===============================
# Embedding precompute
# ===============================
def batch_precompute_embeddings(samples, batch_size=16, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Precomputing embeddings for {len(samples)} samples (batch size={batch_size})...")
    all_sentences = [s["sentence"] for s in samples]
    all_words = [s["words"] for s in samples]

    for start_idx in range(0, len(samples), batch_size):
        batch_sentences = all_sentences[start_idx:start_idx+batch_size]

        encoding = tokenizer(batch_sentences, return_tensors="pt", padding=True,
                             truncation=True, return_offsets_mapping=True)
        with torch.no_grad():
            outputs = encoder(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
        hidden_states = outputs.last_hidden_state

        for i, sample_idx in enumerate(range(start_idx, start_idx + len(batch_sentences))):
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

        # Print progress
        print(f"  Processed samples {start_idx + 1} to {start_idx + len(batch_sentences)} / {len(samples)}")

    if cache_path:
        print(f"Saving embeddings cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)

    print("All embeddings precomputed.")
    return samples