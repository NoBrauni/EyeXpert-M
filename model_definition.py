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
def collate_batch(samples, embedding_cache, device="cpu"):
    """
    Collates a batch of samples for EyeExpertM, injecting word embeddings from a separate cache.

    Returns:
        padded_inputs       : [B, T, hidden_dim] -> embeddings at fixations
        padded_fix          : [B, T] -> padded scanpath indices
        padded_durs         : [B, T] -> padded durations
        padded_full_emb     : [B, max_words, hidden_dim] -> full sentence embeddings
        lengths_tensor      : [B] 1D CPU int64 tensor -> lengths of each sequence
        PAD_IDX             : int
    """
    batch_inputs = []
    batch_fix = []
    batch_dur = []
    batch_full_emb = []
    lengths = []

    for sample in samples:
        norm_sent = normalize_sentence(sample["sentence"])
        if norm_sent not in embedding_cache:
            word_embeddings = torch.zeros(len(sample["words"]), 768, device=device)
        else:
            word_embeddings = embedding_cache[norm_sent]["embeddings"].to(device)

        fix_seq = sample["scanpath"]
        dur_seq = sample["durations"]

        if len(fix_seq) <= 1:
            continue

        fix_indices = torch.tensor(fix_seq[1:], dtype=torch.long, device=device) + 1
        dur_values = torch.tensor(dur_seq[1:], dtype=torch.float, device=device)

        num_words = word_embeddings.size(0)
        mask_valid = fix_indices < num_words
        fix_indices = fix_indices[mask_valid]
        dur_values = dur_values[mask_valid]

        if len(fix_indices) == 0:
            continue

        batch_inputs.append(word_embeddings[fix_indices])
        batch_fix.append(fix_indices)
        batch_dur.append(dur_values)
        batch_full_emb.append(word_embeddings)
        lengths.append(len(fix_indices))

    if not batch_inputs:
        return None

    # Pad sequences
    padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=0.0)
    padded_fix = pad_sequence(batch_fix, batch_first=True, padding_value=PAD_IDX)
    padded_durs = pad_sequence(batch_dur, batch_first=True, padding_value=0.0)

    # Pad full sentence embeddings to max_words
    max_words = max(e.size(0) for e in batch_full_emb)
    hidden_dim = batch_full_emb[0].size(1)
    padded_full_emb = torch.stack([
        torch.cat([e, torch.zeros(max_words - e.size(0), hidden_dim, device=device)], dim=0)
        for e in batch_full_emb
    ], dim=0)

    # Convert lengths to a 1D CPU int64 tensor (required by pack_padded_sequence)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device='cpu')

    return padded_inputs, padded_fix, padded_durs, padded_full_emb, lengths_tensor, PAD_IDX


# -------------------------------
# EyeExpertM Model
# -------------------------------
class EyeExpertM(nn.Module):
    def __init__(
            self,
            hidden_dim=256,
            encoder_dim=768,
            n_experts=5,
            max_seq_len=200,
            n_layers=1,
            dropout=0.1,
            attention_type="dot",  # "dot", "additive", or None
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

        # Projection layer
        self.output_layer = nn.Linear(hidden_dim, encoder_dim)

        # Duration prediction
        self.duration_layer = nn.Linear(hidden_dim, 1)

        # Attention mechanism
        if attention_type == "additive":
            self.attention = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
            self.attention_v = nn.Linear(hidden_dim, 1)
        elif attention_type == "dot":
            self.attention = nn.Linear(hidden_dim, encoder_dim)
        else:
            self.attention = None

    def forward(self, inputs, fix_seq, full_word_embeddings, lengths, expert_id):
        # -------------------------------
        # Mask padding fixations
        # -------------------------------
        valid_mask = fix_seq != PAD_IDX
        scanpath_embeds = self.scanpath_embedding(fix_seq)  # PAD_IDX produces zero vector
        rnn_inputs = torch.cat([inputs, scanpath_embeds], dim=-1)

        # GRU expert
        expert = self.experts[expert_id]

        lengths_tensor = lengths  # Already 1D CPU int64 tensor

        packed = pack_padded_sequence(
            rnn_inputs,
            lengths_tensor,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, _ = expert(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=fix_seq.size(1))

        # -------------------------------
        # Attention / logits
        # -------------------------------
        if self.attention_type == "dot":
            proj = self.output_layer(outputs)
            logits = torch.matmul(proj, full_word_embeddings.transpose(1, 2))
        elif self.attention_type == "additive":
            B, T, H = outputs.shape
            W = full_word_embeddings.size(1)
            outputs_exp = outputs.unsqueeze(2).expand(B, T, W, H)
            words_exp = full_word_embeddings.unsqueeze(1).expand(B, T, W, self.encoder_dim)
            energy = torch.tanh(self.attention(torch.cat([outputs_exp, words_exp], dim=-1)))
            logits = self.attention_v(energy).squeeze(-1)
        else:
            proj = self.output_layer(outputs)
            logits = torch.matmul(proj, full_word_embeddings.transpose(1, 2))

        # -------------------------------
        # Saccade constraint mask
        # -------------------------------
        word_positions = torch.arange(
            full_word_embeddings.size(1),
            device=logits.device
        ).view(1, 1, -1)

        cur_fix_expanded = fix_seq.unsqueeze(-1)
        distance = torch.abs(word_positions - cur_fix_expanded)
        saccade_mask = distance <= self.window_size

        saccade_mask = saccade_mask & (valid_mask.unsqueeze(-1))
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