import torch
import torch.nn as nn
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
# ENCODER (unchanged)
# =========================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def forward(self, input_ids, attention_mask, word_ids_batch):

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        hidden = outputs.last_hidden_state
        B, S, H = hidden.shape
        device = hidden.device

        # FIX: avoid warning
        word_ids = torch.as_tensor(word_ids_batch, device=device, dtype=torch.long)

        hidden_flat = hidden.reshape(B * S, H)
        word_ids_flat = word_ids.reshape(B * S)

        valid_mask = word_ids_flat >= 0

        hidden_flat = hidden_flat[valid_mask]
        word_ids_flat = word_ids_flat[valid_mask]

        max_words = word_ids.max(dim=1).values
        max_words = torch.nan_to_num(max_words, nan=0).long()

        offset = torch.zeros(B, device=device, dtype=torch.long)
        offset[1:] = torch.cumsum(max_words[:-1] + 1, dim=0)

        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), S
        )[valid_mask]

        global_word_ids = word_ids_flat + offset[batch_ids]

        num_words = int(global_word_ids.max() + 1)

        word_reps = torch.zeros(num_words, H, device=device)
        counts = torch.zeros((num_words, 1), device=device)

        word_reps.index_add_(0, global_word_ids, hidden_flat)
        counts.index_add_(
            0,
            global_word_ids,
            torch.ones_like(global_word_ids, dtype=torch.float).unsqueeze(-1)
        )

        word_reps = word_reps / counts.clamp(min=1.0)

        lengths = max_words + 1
        max_len = int(lengths.max())

        out = torch.zeros(B, max_len, H, device=device)

        start = 0
        for b in range(B):
            l = int(lengths[b])
            out[b, :l] = word_reps[start:start + l]
            start += l

        return out


# =========================================================
# DECODER (NO DURATION ANYMORE)
# =========================================================
class Decoder(nn.Module):

    def __init__(self,
                 hidden,
                 n_experts=5,
                 use_experts=True,
                 routing="none",
                 use_lang=True,
                 use_rnn=True):

        super().__init__()

        self.use_experts = use_experts
        self.routing = routing
        self.use_lang = use_lang
        self.use_rnn = use_rnn

        self.input_proj = nn.Linear(hidden + 3, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)

        self.experts = nn.ModuleList([
            nn.GRUCell(hidden, hidden) for _ in range(n_experts)
        ])

        self.router = nn.Linear(hidden, n_experts)

        self.lang_emb = nn.Embedding(10, hidden)
        self.family_emb = nn.Embedding(10, hidden)

        self.query = nn.Linear(hidden, hidden)

        self.moe_loss = torch.tensor(0.0)

    def hard_route(self, family):
        return family % len(self.experts)

    def forward(self, memory, scanpath, durations, family, lang_id):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]

        context = 0
        if self.use_lang:
            context = self.lang_emb(lang_id) + self.family_emb(family)

        zero_sacc = torch.zeros(B, 2, device=memory.device)
        zero_dur = torch.zeros(B, 1, device=memory.device)

        states = []

        for t in range(T):

            prev_idx = scanpath[:, t - 1] if t > 0 else torch.zeros(
                B, dtype=torch.long, device=memory.device
            )

            prev_word = memory[torch.arange(B, device=memory.device), prev_idx]

            prev_dur = torch.log1p(durations[:, t - 1]).unsqueeze(-1) if t > 0 else zero_dur

            inp = self.input_proj(torch.cat([prev_word, zero_sacc, prev_dur], dim=-1))

            state = self.rnn(inp, state) if self.use_rnn else inp

            # -------- MoE --------
            if self.use_experts:
                if self.routing == "soft":
                    w = torch.softmax(self.router(state), dim=-1)
                    new_state = 0
                    for i, expert in enumerate(self.experts):
                        new_state += w[:, i].unsqueeze(-1) * expert(state, state)
                    state = new_state

                elif self.routing == "hard":
                    idx = self.hard_route(family)
                    new_state = torch.zeros_like(state)
                    for i, expert in enumerate(self.experts):
                        mask = (idx == i).float().unsqueeze(-1)
                        new_state += mask * expert(state, state)
                    state = new_state

            state = state + context
            states.append(state)

        states = torch.stack(states, dim=1)  # (B, T, H)

        # fixation logits
        q = self.query(states)
        logits = torch.einsum("bth,bwh->btw", q, memory)

        # MoE entropy loss
        if self.use_experts and self.routing == "soft":
            r = torch.softmax(self.router(states), dim=-1)
            p = r.mean(dim=(0, 1))
            self.moe_loss = -(p * torch.log(p + 1e-8)).sum()

        return logits, states


# =========================================================
# SEPARATE DURATION HEAD
# =========================================================
class DurationHead(nn.Module):

    def __init__(self, hidden, use_lang=True):
        super().__init__()

        self.use_lang = use_lang

        if use_lang:
            self.lang_emb = nn.Embedding(10, hidden)
            self.family_emb = nn.Embedding(10, hidden)

        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, states, family=None, lang_id=None):

        if self.use_lang:
            context = self.lang_emb(lang_id) + self.family_emb(family)
            states = states + context.unsqueeze(1)

        return self.net(states)


# =========================================================
# FULL MODEL (EyeXpert-M)
# =========================================================
class EyeXpertM(nn.Module):

    def __init__(self,
                 use_experts=True,
                 routing="none",
                 use_lang=True,
                 use_rnn=True,
                 use_duration=True,
                 duration_lang=True):

        super().__init__()

        self.use_duration = use_duration

        self.encoder = Encoder()
        hidden = self.encoder.model.config.hidden_size

        self.decoder = Decoder(
            hidden,
            use_experts=use_experts,
            routing=routing,
            use_lang=use_lang,
            use_rnn=use_rnn
        )

        if use_duration:
            self.duration_head = DurationHead(
                hidden,
                use_lang=duration_lang
            )

    def forward(self,
                input_ids,
                attention_mask,
                word_ids,
                scanpath,
                durations,
                family,
                lang_id):

        memory = self.encoder(input_ids, attention_mask, word_ids)

        logits, states = self.decoder(
            memory,
            scanpath,
            durations,
            family,
            lang_id
        )

        if self.use_duration:
            dur_pred = self.duration_head(states.detach(), family, lang_id)
        else:
            dur_pred = None

        return logits, dur_pred, self.decoder.moe_loss