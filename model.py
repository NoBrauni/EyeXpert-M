import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel


# =========================================================
# LANGUAGE FAMILY
# =========================================================
LANG_FAMILY = {
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0,
    "sp": 1, "sp_ch": 1, "it": 1, "bp": 1,
    "no": 2, "da": 2, "ic": 2,
    "se": 3, "ru": 3, "ru_mo": 3,
    "fi": 4, "ee": 4,
}

# =========================================================
# ENCODER
# =========================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def forward(self, input_ids, attention_mask, word_ids):

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        hidden = out.last_hidden_state
        B, S, H = hidden.shape
        device = hidden.device

        word_ids = word_ids.to(device)

        flat_h = hidden.reshape(B * S, H)
        flat_w = word_ids.reshape(B * S)

        valid = flat_w >= 0
        flat_h = flat_h[valid]
        flat_w = flat_w[valid]

        max_words = word_ids.max(dim=1).values
        max_words = torch.nan_to_num(max_words, nan=0).long()

        offsets = torch.zeros(B, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(max_words[:-1] + 1, dim=0)

        batch_ids = torch.repeat_interleave(torch.arange(B, device=device), S)[valid]
        global_ids = flat_w + offsets[batch_ids]

        num_words = int(global_ids.max().item()) + 1

        word_repr = torch.zeros(num_words, H, device=device)
        counts = torch.zeros(num_words, 1, device=device)

        word_repr.index_add_(0, global_ids, flat_h)
        counts.index_add_(0, global_ids, torch.ones_like(global_ids, dtype=torch.float).unsqueeze(-1))

        word_repr = word_repr / counts.clamp(min=1.0)

        lengths = max_words + 1
        max_len = int(lengths.max())

        out = torch.zeros(B, max_len, H, device=device)

        start = 0
        for b in range(B):
            l = int(lengths[b])
            out[b, :l] = word_repr[start:start + l]
            start += l

        return out


# =========================================================
# BASE MODEL
# =========================================================
class BaseDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRUCell(hidden + 1, hidden)
        self.query = nn.Linear(hidden, hidden)

    def forward(self, memory, scanpath, durations):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]
        states = []

        for t in range(T):

            prev = scanpath[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device, dtype=torch.long)
            prev_word = memory[torch.arange(B, device=memory.device), prev]

            dur = torch.log1p(
                durations[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device)
            ).unsqueeze(-1)

            inp = torch.cat([prev_word, dur], dim=-1)

            state = self.gru(inp, state)
            states.append(state)

        states = torch.stack(states, dim=1)
        logits = torch.einsum("bth,bwh->btw", self.query(states), memory)

        return logits, states


# =========================================================
# BASE MODEL PLUS
# =========================================================
class BaseDecoderPlus(nn.Module):
    """Non-modular decoder with extra trainable layers to match hard/moe parameter count."""
    def __init__(self, hidden):
        super().__init__()
        
        # Stack 5 GRU cells to match hard/moe parameter count
        self.grus = nn.ModuleList([
            nn.GRUCell(hidden + 1, hidden) for _ in range(5)
        ])
        
        # Extra linear layers for capacity
        self.pre_processing = nn.Sequential(
            nn.Linear(hidden + 1, hidden + 1),
            nn.ReLU(),
            nn.Linear(hidden + 1, hidden + 1)
        )
        
        self.query = nn.Linear(hidden, hidden)

    def forward(self, memory, scanpath, durations):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]
        states = []

        for t in range(T):

            prev = scanpath[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device, dtype=torch.long)
            prev_word = memory[torch.arange(B, device=memory.device), prev]

            dur = torch.log1p(
                durations[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device)
            ).unsqueeze(-1)

            inp = torch.cat([prev_word, dur], dim=-1)
            
            # Pre-process input
            processed = self.pre_processing(inp)
            
            # Pass through all GRU cells sequentially (no routing, non-modular)
            for gru in self.grus:
                state = gru(processed, state)

            states.append(state)

        states = torch.stack(states, dim=1)
        logits = torch.einsum("bth,bwh->btw", self.query(states), memory)

        return logits, states


# =========================================================
# HARD ROUTING
# =========================================================
class HardDecoder(nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.GRUCell(hidden + 1, hidden) for _ in range(n_experts)
        ])

        self.query = nn.Linear(hidden, hidden)

    def forward(self, memory, scanpath, durations, family):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]
        states = []

        for t in range(T):

            prev = scanpath[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device, dtype=torch.long)
            prev_word = memory[torch.arange(B, device=memory.device), prev]

            dur = torch.log1p(
                durations[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device)
            ).unsqueeze(-1)

            inp = torch.cat([prev_word, dur], dim=-1)

            idx = family % len(self.experts)

            new_state = torch.stack([
                self.experts[idx[b].item()](inp[b], state[b])
                for b in range(B)
            ])

            state = new_state
            states.append(state)

        states = torch.stack(states, dim=1)
        logits = torch.einsum("bth,bwh->btw", self.query(states), memory)

        return logits, states


# =========================================================
# TRUE MoE
# =========================================================
class MoEDecoder(nn.Module):
    def __init__(self, hidden, n_experts):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.GRUCell(hidden + 1, hidden) for _ in range(n_experts)
        ])

        self.router = nn.Linear(hidden, n_experts)
        self.query = nn.Linear(hidden, hidden)

        self.last_probs = None

    def forward(self, memory, scanpath, durations):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]
        states = []

        all_probs = []

        for t in range(T):

            prev = scanpath[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device, dtype=torch.long)
            prev_word = memory[torch.arange(B, device=memory.device), prev]

            dur = torch.log1p(
                durations[:, t - 1] if t > 0 else torch.zeros(B, device=memory.device)
            ).unsqueeze(-1)

            inp = torch.cat([prev_word, dur], dim=-1)

            logits = self.router(state)
            probs = torch.softmax(logits, dim=-1)

            # Soft routing: weighted sum of expert outputs
            expert_outputs = torch.stack([
                self.experts[e](inp, state) for e in range(len(self.experts))
            ], dim=1)  # B, n_experts, H

            new_state = torch.einsum("be,bef->bf", probs, expert_outputs)

            state = new_state
            states.append(state)

            all_probs.append(probs)

        self.last_probs = torch.stack(all_probs, dim=1)

        states = torch.stack(states, dim=1)
        logits = torch.einsum("bth,bwh->btw", self.query(states), memory)

        return logits, states


# =========================================================
# MODEL WRAPPER
# =========================================================
class EyeXpertM(nn.Module):

    def __init__(self, mode="base", n_experts=5):
        super().__init__()

        self.encoder = Encoder()
        hidden = self.encoder.model.config.hidden_size

        self.mode = mode

        if mode == "base":
            self.decoder = BaseDecoder(hidden)
        elif mode == "base+":
            self.decoder = BaseDecoderPlus(hidden)
        elif mode == "hard":
            self.decoder = HardDecoder(hidden, n_experts)
        elif mode == "moe":
            self.decoder = MoEDecoder(hidden, n_experts)

        self.duration_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask, word_ids,
                scanpath, durations, family, lang_id=None):

        memory = self.encoder(input_ids, attention_mask, word_ids)

        if self.mode == "base" or self.mode == "base+":
            logits, states = self.decoder(memory, scanpath, durations)
        elif self.mode == "hard":
            logits, states = self.decoder(memory, scanpath, durations, family)
        else:
            logits, states = self.decoder(memory, scanpath, durations)

        dur_pred = self.duration_head(states)

        moe_metrics = {}
        if self.mode == "moe" and self.decoder.last_probs is not None:
            p = self.decoder.last_probs.mean(dim=(0, 1))
            moe_metrics["load_balance"] = (p ** 2).sum()

        return logits, dur_pred, moe_metrics