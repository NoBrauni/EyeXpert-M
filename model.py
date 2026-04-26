import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# ENCODER (subword → word pooling)
# =========================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, word_ids_batch):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        subword_hidden = outputs.last_hidden_state  # (B, S, H)

        batch_word_reps = []

        for b in range(subword_hidden.size(0)):

            word_ids = word_ids_batch[b]
            hidden = subword_hidden[b]

            word_map = {}

            for i, w_id in enumerate(word_ids):
                if w_id is None:
                    continue
                word_map.setdefault(w_id, []).append(hidden[i])

            pooled = []
            for w_id in sorted(word_map.keys()):
                pooled.append(torch.stack(word_map[w_id]).mean(0))

            batch_word_reps.append(torch.stack(pooled))

        max_len = max(x.size(0) for x in batch_word_reps)
        H = subword_hidden.size(-1)

        out = torch.zeros(
            subword_hidden.size(0),
            max_len,
            H,
            device=subword_hidden.device
        )

        for i, x in enumerate(batch_word_reps):
            out[i, :x.size(0)] = x

        return out


# =========================================================
# DECODER (with duration modeling + feedback)
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

        # +3 now: saccade(2) + duration(1)
        self.input_proj = nn.Linear(hidden + 3, hidden)

        self.rnn = nn.GRUCell(hidden, hidden)

        self.experts = nn.ModuleList([
            nn.GRUCell(hidden, hidden) for _ in range(n_experts)
        ])

        self.router = nn.Linear(hidden, n_experts)

        self.lang_emb = nn.Embedding(10, hidden)
        self.family_emb = nn.Embedding(10, hidden)

        self.query = nn.Linear(hidden, hidden)

        # duration head (log-duration)
        self.duration_head = nn.Linear(hidden, 1)

        self.moe_loss = torch.tensor(0.0)

    def hard_route(self, family):
        return family % len(self.experts)

    def forward(self, memory, scanpath, durations, family, lang_id):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]

        outputs = []
        duration_outputs = []
        router_history = []

        context = 0
        if self.use_lang:
            context = self.lang_emb(lang_id) + self.family_emb(family)

        for t in range(T):

            prev_idx = scanpath[:, t-1] if t > 0 else torch.zeros(
                B, dtype=torch.long, device=memory.device
            )

            prev_word = memory[torch.arange(B), prev_idx]

            # ----------------------------------
            # LOG-DURATION INPUT (teacher forcing)
            # ----------------------------------
            if t > 0:
                prev_dur = torch.log1p(durations[:, t-1]).unsqueeze(-1)
            else:
                prev_dur = torch.zeros(B, 1, device=memory.device)

            sacc = torch.zeros(B, 2, device=memory.device)

            inp = self.input_proj(torch.cat([prev_word, sacc, prev_dur], dim=-1))

            state = self.rnn(inp, state) if self.use_rnn else inp

            # -----------------------------
            # MoE
            # -----------------------------
            if self.use_experts:

                if self.routing == "soft":
                    w = torch.softmax(self.router(state), dim=-1)
                    router_history.append(w)

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

            # -----------------------------
            # FIXATION PREDICTION
            # -----------------------------
            q = self.query(state)
            logits = torch.einsum("bh,bwh->bw", q, memory)
            outputs.append(logits.unsqueeze(1))

            # -----------------------------
            # DURATION PREDICTION (log-space)
            # -----------------------------
            dur_pred = self.duration_head(state)
            duration_outputs.append(dur_pred.unsqueeze(1))

        if self.use_experts and self.routing == "soft" and router_history:
            r = torch.stack(router_history, dim=1)
            p = r.mean(dim=(0, 1))
            self.moe_loss = -(p * torch.log(p + 1e-8)).sum()

        return (
            torch.cat(outputs, dim=1),           # (B, T, W)
            torch.cat(duration_outputs, dim=1),  # (B, T, 1)
        )


# =========================================================
# FULL MODEL
# =========================================================
class EyeXpertM(nn.Module):

    def __init__(self,
                 use_experts=True,
                 routing="none",
                 use_lang=True,
                 use_rnn=True):

        super().__init__()

        self.encoder = Encoder()
        hidden = self.encoder.model.config.hidden_size

        self.decoder = Decoder(
            hidden,
            use_experts=use_experts,
            routing=routing,
            use_lang=use_lang,
            use_rnn=use_rnn
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

        logits, dur_pred = self.decoder(
            memory,
            scanpath,
            durations,
            family,
            lang_id
        )

        return logits, dur_pred, self.decoder.moe_loss