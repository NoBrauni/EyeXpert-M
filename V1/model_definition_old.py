import torch
import torch.nn as nn
import pandas as pd
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import pickle
import string
from torch.nn.utils.rnn import pad_sequence

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

# ===============================
# Dataset
# ===============================
class MECODataset:
    def __init__(self, et_csv_path=None, min_dur=60):
        self.samples = []

        if et_csv_path:
            df = pd.read_csv(et_csv_path, low_memory=False)

            # remove blinks and extremely short fixations
            df = df[(df["blink"] == 0) & (df["dur"] >= min_dur)]

            # ensure temporal order
            df = df.sort_values(["subid", "unique_sentence_id", "fix_index"])

            grouped = df.groupby(["subid", "unique_sentence_id"])

            for (subid, sent_id), group in grouped:
                full_sentence = str(group["sentence"].iloc[0])

                # canonical word list per sentence
                word_table = (
                    group[["wordnum", "word"]]
                    .drop_duplicates(subset="wordnum")
                    .sort_values("wordnum")
                )

                words = word_table["word"].astype(str).tolist()
                wordnum_to_local = {wn: i for i, wn in enumerate(word_table["wordnum"].tolist())}

                # Build sentence-local fixation sequence and durations
                fix_seq = []
                dur_seq = []
                for wn, dur in zip(group["wordnum"], group["dur"]):
                    if wn in wordnum_to_local:
                        fix_seq.append(wordnum_to_local[wn])
                        dur_seq.append(float(dur))

                if len(fix_seq) == 0:
                    continue  # skip empty sentences

                # Language
                lang_code = str(group["lang"].iloc[0]).strip().lower()

                # Append sample
                self.samples.append({
                    "sentence": full_sentence,
                    "words": words,
                    "fix_seq": fix_seq,
                    "dur_seq": dur_seq,
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
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            sentence_cache = pickle.load(f)
    else:
        sentence_cache = {}

    # Sentences that need new embeddings
    sentences_to_compute = [s["sentence"] for s in samples if s["sentence"] not in sentence_cache]
    sentences_to_compute = list(set(sentences_to_compute))
    print(f"Need to compute embeddings for {len(sentences_to_compute)} new sentences")

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

        with torch.no_grad():
            outputs = encoder(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
        hidden = outputs.last_hidden_state  # [B, seq_len_tokens, hidden_dim]

        # Store embeddings and offsets in cache
        for i, sent in enumerate(batch_sentences):
            sentence_cache[sent] = {
                "hidden": hidden[i],
                "offsets": encoding["offset_mapping"][i],
                "tokens": tokenizer.convert_ids_to_tokens(encoding["input_ids"][i])
            }

        print(f"Computed {start + len(batch_sentences)} / {len(sentences_to_compute)}")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(sentence_cache, f)
        print(f"Saved embedding cache to {cache_path}")

    # Assign embeddings per fixation
    for sample in samples:
        cached = sentence_cache[sample["sentence"]]
        token_embeds = cached["hidden"]
        offsets = cached["offsets"]
        sentence_text = sample["sentence"]

        embeddings_per_fix = []

        for fix_idx in sample["fix_seq"]:
            # Get corresponding word (clipped if needed)
            word = sample["words"][min(fix_idx, len(sample["words"])-1)].strip(string.punctuation).lower()

            # Find matching token embeddings
            token_vectors = []
            for start, end in offsets:
                token_text = sentence_text[start:end].strip(string.punctuation).lower()
                if token_text == word:
                    token_vectors.append(token_embeds[start:end].mean(dim=0) if token_embeds.ndim > 1 else token_embeds[start])

            # Fallback to token at fix_idx if no exact match
            if not token_vectors:
                token_vectors.append(token_embeds[min(fix_idx, token_embeds.size(0)-1)])

            embeddings_per_fix.append(torch.stack(token_vectors).mean(dim=0))

        sample["embeddings"] = torch.stack(embeddings_per_fix, dim=0)

    print("All word-level embeddings assigned per fixation.")
    return samples


# ===============================
# Collate function
# ===============================
def collate_batch(samples, device="cpu"):
    batch_inputs, fix_seqs, dur_seqs, lengths = [], [], [], []
    full_word_embeddings = []

    for sample in samples:
        word_embeddings = sample["embeddings"]  # already per-word
        fix_seq = sample["fix_seq"]
        dur_seq = sample["dur_seq"]

        # Flatten fixations per word
        unique_wordnums = sorted(set(fix_seq))
        wordnum_to_idx = {num: i for i, num in enumerate(unique_wordnums)}
        fix_indices = [wordnum_to_idx[n] for n in fix_seq[1:]]  # skip first fixation for durations
        dur_values = dur_seq[1:]  # skip first fixation

        if len(fix_indices) == 0 or max(fix_indices) >= len(word_embeddings):
            continue

        # Add to batch
        batch_inputs.append(word_embeddings[torch.tensor(fix_indices)])
        fix_seqs.append(torch.tensor(fix_indices, dtype=torch.long))
        dur_seqs.append(torch.tensor(dur_values, dtype=torch.float) / 1000.0)  # convert ms -> sec
        lengths.append(len(fix_indices))
        full_word_embeddings.append(word_embeddings)

    if not batch_inputs:
        return None

    # Pad sequences
    padded_inputs = pad_sequence(batch_inputs, batch_first=True)         # [batch, seq_len, embed_dim]
    padded_fixes = pad_sequence(fix_seqs, batch_first=True)              # [batch, seq_len]
    padded_durs = pad_sequence(dur_seqs, batch_first=True)               # [batch, seq_len]
    padded_full_words = pad_sequence(full_word_embeddings, batch_first=True)  # [batch, num_words, embed_dim]

    # Move to device
    padded_inputs = padded_inputs.to(device)
    padded_fixes = padded_fixes.to(device)
    padded_durs = padded_durs.to(device)
    padded_full_words = padded_full_words.to(device)

    return padded_inputs, padded_fixes, padded_durs, padded_full_words, lengths


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