import torch
import torch.nn as nn
import pandas as pd
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import pickle
import string
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# ===============================
# Language to Expert mapping
# ===============================
LANG_TO_EXPERT = {
    # Expert 0 — Germanic
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0,
    # Expert 1 — Nordic
    "no": 1, "da": 1, "ic": 1,
    # Expert 2 — Romance
    "sp": 2, "sp_ch": 2, "it": 2, "bp": 2,
    # Expert 3 — Slavic
    "ru": 3, "ru_mo": 3, "se": 3,
    # Expert 4 — Uralic
    "fi": 4, "ee": 4,
}


class MECODataset:
    def __init__(self, csv_path, min_dur=60):

        df = pd.read_csv(csv_path, low_memory=False)

        df = df[(df["blink"] == 0) & (df["dur"] >= min_dur)]
        df = df.sort_values(["subid", "unique_sentence_id", "fix_index"])

        self.samples = []
        grouped = df.groupby(["subid", "unique_sentence_id"])

        for (subid, sent_id), group in grouped:

            sentence = str(group["sentence"].iloc[0])
            lang = str(group["lang"].iloc[0]).lower()

            # Canonical word list
            word_table = (
                group[["ianum", "word"]]
                .drop_duplicates(subset="ianum")
                .sort_values("ianum")
            )

            words = word_table["word"].astype(str).tolist()

            # Map interest-area index → local index
            ia_to_local = {
                ia: i for i, ia in enumerate(word_table["ianum"].tolist())
            }

            fix_seq = []
            dur_seq = []

            for _, row in group.iterrows():

                ia = row["ianum"]
                dur = row["dur"]

                if ia in ia_to_local:
                    fix_seq.append(ia_to_local[ia])
                    dur_seq.append(float(dur))

            if len(fix_seq) == 0:
                continue

            # Normalize durations
            dur_mean, dur_std = np.load("duration_stats.npy")
            dur_seq = np.array(dur_seq)
            dur_seq = (dur_seq - dur_mean) / (dur_std + 1e-6)

            self.samples.append(
                {
                    "sentence": sentence,
                    "words": words,
                    "scanpath": fix_seq,
                    "durations": dur_seq.tolist(),
                    "lang": lang,
                    "reader": subid,
                    "sentence_id": sent_id,
                    "sentence_len": len(words),
                }
            )

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

def batch_precompute_embeddings(samples, batch_size=16, cache_path=None, device="cpu"):
    """
    Computes word-level embeddings for each sample in 'samples' using XLM-R.
    Each sentence returns a tensor of shape [num_words, hidden_dim].
    Caches results to 'cache_path' if provided.
    """
    # Ensure cache directory exists
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir != "":
            os.makedirs(cache_dir, exist_ok=True)

    # Load cache if available
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            sentence_cache = pickle.load(f)
    else:
        sentence_cache = {}

    # Unique sentences to compute
    sentences = list(set(s["sentence"] for s in samples))
    sentences_to_compute = [s for s in sentences if s not in sentence_cache]
    print(f"Need to compute embeddings for {len(sentences_to_compute)} sentences")

    # Compute embeddings in batches
    for start in range(0, len(sentences_to_compute), batch_size):
        batch_sentences = sentences_to_compute[start:start + batch_size]

        encoding = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True
        )

        offsets = encoding["offset_mapping"]
        encoding = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}

        with torch.no_grad():
            outputs = encoder(**encoding)

        hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        # Align tokens to words
        for i, sentence in enumerate(batch_sentences):
            token_embeds = hidden[i]  # [seq_len, hidden_dim]
            token_offsets = offsets[i]

            words = sentence.split()
            word_vectors = []

            for word in words:
                word_clean = word.strip(string.punctuation).lower()
                subword_vectors = []

                for tok_i, (start_c, end_c) in enumerate(token_offsets):
                    token_text = sentence[start_c:end_c].strip(string.punctuation).lower()
                    if token_text == word_clean:
                        subword_vectors.append(token_embeds[tok_i])

                if subword_vectors:
                    word_vectors.append(torch.stack(subword_vectors).mean(dim=0))
                else:
                    # fallback to first token embedding of sentence
                    word_vectors.append(token_embeds[0])

            sentence_cache[sentence] = torch.stack(word_vectors)

        print(f"Computed {start + len(batch_sentences)} / {len(sentences_to_compute)}")

    # Save cache
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(sentence_cache, f)
        print(f"Saved cache to {cache_path}")

    # Assign embeddings to samples
    for sample in samples:
        sample["word_embeddings"] = sentence_cache[sample["sentence"]]

    print("Word embeddings assigned.")
    return samples

PAD_IDX = 0

def collate_batch(samples, device="cpu"):
    batch_inputs = []
    fix_seqs = []
    dur_seqs = []
    lengths = []
    full_word_embeddings = []

    for sample in samples:
        word_embeddings = sample["word_embeddings"]
        fix_seq = sample["scanpath"]
        dur_seq = sample["durations"]

        # skip first fixation (no prediction target)
        if len(fix_seq) <= 1:
            continue
        fix_indices = torch.tensor(fix_seq[1:], dtype=torch.long) + 1
        dur_values = torch.tensor(dur_seq[1:], dtype=torch.float)

        # ensure indices are within bounds
        num_words = word_embeddings.size(0)
        mask_valid = fix_indices < num_words
        fix_indices = fix_indices[mask_valid]
        dur_values = dur_values[mask_valid]

        if len(fix_indices) == 0:
            continue

        batch_inputs.append(word_embeddings[fix_indices])
        fix_seqs.append(fix_indices)
        dur_seqs.append(dur_values)
        lengths.append(len(fix_indices))
        full_word_embeddings.append(word_embeddings)

    if not batch_inputs:
        return None

    # pad sequences with PAD_IDX for fixations
    padded_inputs = pad_sequence(batch_inputs, batch_first=True)
    padded_fixes = pad_sequence(fix_seqs, batch_first=True, padding_value=PAD_IDX)
    padded_durs = pad_sequence(dur_seqs, batch_first=True)
    padded_full_words = pad_sequence(full_word_embeddings, batch_first=True)

    # move to device
    padded_inputs = padded_inputs.to(device)
    padded_fixes = padded_fixes.to(device)
    padded_durs = padded_durs.to(device)
    padded_full_words = padded_full_words.to(device)

    return padded_inputs, padded_fixes, padded_durs, padded_full_words, lengths

# ===============================
# MoE Decoder
# ===============================
class EyeExpertM(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        encoder_dim=768,
        n_experts=5,
        max_seq_len=200,
        n_layers=1,
        dropout=0.1,
        attention_type="dot",   # "dot", "additive", or None
        window_size=8
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.n_experts = n_experts
        self.window_size = window_size
        self.attention_type = attention_type

        # Experts (GRU)
        self.experts = nn.ModuleList([
            nn.GRU(
                input_size=encoder_dim + 32,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
            for _ in range(n_experts)
        ])

        # Scanpath embedding
        self.scanpath_embedding = nn.Embedding(
            max_seq_len + 1,
            32,
            padding_idx=PAD_IDX
        )

        # -------------------------------
        # Projection layer
        # -------------------------------
        self.output_layer = nn.Linear(hidden_dim, encoder_dim)

        # -------------------------------
        # Duration prediction
        # -------------------------------
        self.duration_layer = nn.Linear(hidden_dim, 1)

        # -------------------------------
        # Attention mechanism
        # -------------------------------
        if attention_type == "additive":
            self.attention = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
            self.attention_v = nn.Linear(hidden_dim, 1)

        elif attention_type == "dot":
            self.attention = nn.Linear(hidden_dim, encoder_dim)

        else:
            self.attention = None

    def forward(self, inputs, fix_seq, full_word_embeddings, lengths, expert_id):

        # -------------------------------
        # Scanpath embeddings
        # -------------------------------
        scanpath_embeds = self.scanpath_embedding(fix_seq)
        rnn_inputs = torch.cat([inputs, scanpath_embeds], dim=-1)

        expert = self.experts[expert_id]

        lengths = torch.tensor(lengths, dtype=torch.long, device="cpu")

        packed = pack_padded_sequence(
            rnn_inputs,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        outputs, _ = expert(packed)

        outputs, _ = pad_packed_sequence(
            outputs,
            batch_first=True,
            total_length=fix_seq.size(1)
        )

        # -------------------------------
        # Attention
        # -------------------------------
        if self.attention_type == "dot":

            proj = self.output_layer(outputs)
            logits = torch.matmul(
                proj,
                full_word_embeddings.transpose(1, 2)
            )

        elif self.attention_type == "additive":

            B, T, H = outputs.shape
            W = full_word_embeddings.size(1)

            outputs_exp = outputs.unsqueeze(2).expand(B, T, W, H)
            words_exp = full_word_embeddings.unsqueeze(1).expand(B, T, W, self.encoder_dim)

            energy = torch.tanh(
                self.attention(torch.cat([outputs_exp, words_exp], dim=-1))
            )

            logits = self.attention_v(energy).squeeze(-1)

        else:
            proj = self.output_layer(outputs)
            logits = torch.matmul(
                proj,
                full_word_embeddings.transpose(1, 2)
            )

        # -------------------------------
        # Saccade constraint mask
        # -------------------------------
        cur_fix = fix_seq

        word_positions = torch.arange(
            full_word_embeddings.size(1),
            device=logits.device
        ).view(1, 1, -1)

        cur_fix_expanded = cur_fix.unsqueeze(-1)

        distance = torch.abs(word_positions - cur_fix_expanded)

        saccade_mask = distance <= self.window_size

        logits = logits.masked_fill(~saccade_mask, -1e9)

        # -------------------------------
        # Padding word mask
        # -------------------------------
        word_mask = (full_word_embeddings.abs().sum(-1) != 0)
        logits = logits.masked_fill(~word_mask.unsqueeze(1), -1e9)

        # -------------------------------
        # Duration prediction
        # -------------------------------
        dur_pred = self.duration_layer(outputs).squeeze(-1)

        return logits, dur_pred