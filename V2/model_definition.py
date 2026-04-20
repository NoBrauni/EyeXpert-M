import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from embeddings import normalize_sentence  # import helper

PAD_IDX = 0

# -------------------------------
# Language → Expert mapping
# -------------------------------
LANG_TO_EXPERT = {
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0,
    "no": 1, "da": 1, "ic": 1,
    "sp": 2, "sp_ch": 2, "it": 2, "bp": 2,
    "ru": 3, "ru_mo": 3, "se": 3,
    "fi": 4, "ee": 4,
}

# -------------------------------
# Collate function for batching
# -------------------------------
import torch
from torch.nn.utils.rnn import pad_sequence
from embeddings import normalize_sentence

PAD_IDX = 0


def collate_batch(samples, embedding_cache, device="cpu"):
    batch_inputs = []
    batch_fix = []
    batch_dur = []
    batch_full_emb = []
    lengths = []

    for sample in samples:
        sentence_key = normalize_sentence(sample["sentence"])

        # embeddings
        if sentence_key not in embedding_cache:
            continue  # IMPORTANT: skip instead of zero-padding silently

        word_embeddings = embedding_cache[sentence_key]["embeddings"].to(device)

        fix_seq = torch.tensor(sample["scanpath"], dtype=torch.long, device=device)
        dur_seq = torch.tensor(sample["durations"], dtype=torch.float, device=device)

        # HARD ALIGN CHECK
        min_len = min(len(fix_seq), len(dur_seq))
        fix_seq = fix_seq[:min_len]
        dur_seq = dur_seq[:min_len]

        if min_len <= 2:
            continue

        # shift
        input_fix = fix_seq[:-1] + 1
        target_fix = fix_seq[1:] + 1
        dur_values = dur_seq[1:]

        # normalize duration safely
        dur_values = dur_values / (dur_values.mean() + 1e-6)

        num_words = word_embeddings.size(0)

        # mask validity
        mask_valid = (input_fix < num_words) & (target_fix < num_words)

        if mask_valid.sum() < 2:
            continue

        input_fix = input_fix[mask_valid]
        target_fix = target_fix[mask_valid]
        dur_values = dur_values[mask_valid]

        # final safety check (CRITICAL)
        if not (len(input_fix) == len(target_fix) == len(dur_values)):
            continue

        batch_inputs.append(word_embeddings[input_fix])
        batch_fix.append(target_fix)
        batch_dur.append(dur_values)
        batch_full_emb.append(word_embeddings)
        lengths.append(len(target_fix))

    if len(batch_inputs) == 0:
        return None

    padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    padded_fix = pad_sequence(batch_fix, batch_first=True, padding_value=PAD_IDX)
    padded_durs = pad_sequence(batch_dur, batch_first=True, padding_value=0.0)

    max_words = max(e.size(0) for e in batch_full_emb)
    hidden_dim = batch_full_emb[0].size(1)

    padded_full_emb = torch.stack([
        torch.cat([e, torch.zeros(max_words - e.size(0), hidden_dim, device=device)], dim=0)
        for e in batch_full_emb
    ], dim=0)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_inputs, padded_fix, padded_durs, padded_full_emb, lengths_tensor, PAD_IDX


# -------------------------------
# EyeExpertM (Improved Hard-Routed MoE)
# -------------------------------
class EyeExpertM(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        encoder_dim=768,
        n_experts=5,
        max_seq_len=200,
        n_layers=1,
        dropout=0.2,
        attention_type="dot",
        window_size=8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.n_experts = n_experts
        self.window_size = window_size
        self.attention_type = attention_type

        # -------------------------------
        # Hard-routed experts (GRUs)
        # -------------------------------
        self.experts = nn.ModuleList([
            nn.GRU(
                input_size=encoder_dim + 32,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True
            )
            for _ in range(n_experts)
        ])

        # -------------------------------
        # Scanpath embedding
        # -------------------------------
        self.scanpath_embedding = nn.Embedding(
            max_seq_len + 1,
            32,
            padding_idx=PAD_IDX
        )
        self.scanpath_dropout = nn.Dropout(dropout)

        # -------------------------------
        # Projection to encoder space
        # -------------------------------
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, encoder_dim),
            nn.LayerNorm(encoder_dim)
        )

        # -------------------------------
        # Duration head (stabilized)
        # -------------------------------
        self.duration_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # -------------------------------
        # Attention
        # -------------------------------
        if attention_type == "additive":
            self.attention = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
            self.attention_v = nn.Linear(hidden_dim, 1)
        else:
            self.attention = None

        self.scale = encoder_dim ** 0.5

    # -------------------------------
    # forward
    # -------------------------------
    def forward(self, inputs, fix_seq, full_word_embeddings, lengths, expert_id):
        """
        inputs: [B, T, 768]
        fix_seq: [B, T]
        full_word_embeddings: [B, W, 768]
        lengths: CPU tensor [B]
        expert_id: int
        """

        B, T, _ = inputs.shape

        # -------------------------------
        # Mask padding
        # -------------------------------
        valid_mask = fix_seq != PAD_IDX

        # -------------------------------
        # Embedding scanpath
        # -------------------------------
        scan_emb = self.scanpath_embedding(fix_seq)
        scan_emb = self.scanpath_dropout(scan_emb)

        rnn_inputs = torch.cat([inputs, scan_emb], dim=-1)

        # -------------------------------
        # Expert selection (hard routing)
        # -------------------------------
        expert_id = int(expert_id)
        expert = self.experts[expert_id]

        packed = pack_padded_sequence(
            rnn_inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = expert(packed)

        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T
        )

        # -------------------------------
        # Projection
        # -------------------------------
        proj = self.output_layer(outputs)  # [B, T, 768]

        # -------------------------------
        # Attention logits
        # -------------------------------
        W = full_word_embeddings.size(1)

        if self.attention_type == "dot":
            logits = torch.matmul(
                proj,
                full_word_embeddings.transpose(1, 2)
            ) / self.scale

        elif self.attention_type == "additive":
            B, T, H = outputs.shape

            out_exp = outputs.unsqueeze(2).expand(B, T, W, H)
            word_exp = full_word_embeddings.unsqueeze(1).expand(B, T, W, self.encoder_dim)

            energy = torch.tanh(self.attention(torch.cat([out_exp, word_exp], dim=-1)))
            logits = self.attention_v(energy).squeeze(-1)

        else:
            logits = torch.matmul(
                proj,
                full_word_embeddings.transpose(1, 2)
            ) / self.scale

        # -------------------------------
        # Saccade constraint (hard mask)
        # -------------------------------
        word_positions = torch.arange(W, device=logits.device).view(1, 1, W)
        cur_fix = fix_seq.unsqueeze(-1)

        distance = torch.abs(word_positions - cur_fix)
        saccade_mask = (distance <= self.window_size) & valid_mask.unsqueeze(-1)

        logits = logits.masked_fill(~saccade_mask, float("-inf"))

        # -------------------------------
        # Word padding mask
        # -------------------------------
        word_mask = (full_word_embeddings.abs().sum(-1) != 0)
        logits = logits.masked_fill(~word_mask.unsqueeze(1), float("-inf"))

        # -------------------------------
        # Duration prediction
        # -------------------------------
        dur_pred = self.duration_layer(outputs).squeeze(-1)

        return logits, dur_pred