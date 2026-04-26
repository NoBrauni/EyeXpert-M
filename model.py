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
import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()  # important for speed

    def forward(self, input_ids, attention_mask, word_ids_batch):

        # -------------------------------------------------
        # 1) XLM-R forward (frozen, no grad)
        # -------------------------------------------------
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        hidden = outputs.last_hidden_state  # (B, S, H)
        B, S, H = hidden.shape
        device = hidden.device

        # -------------------------------------------------
        # 2) Flatten batch + sequence
        # -------------------------------------------------
        hidden_flat = hidden.reshape(B * S, H)

        # -------------------------------------------------
        # 3) Build word indices (vectorized)
        # -------------------------------------------------
        # word_ids_batch: list of length B, each list length S
        word_ids = torch.full((B, S), -1, dtype=torch.long, device=device)

        for b in range(B):
            w = word_ids_batch[b]
            w = torch.tensor(w, dtype=torch.long, device=device)
            word_ids[b] = w

        word_ids_flat = word_ids.reshape(B * S)

        # -------------------------------------------------
        # 4) Remove special tokens (-1)
        # -------------------------------------------------
        valid_mask = word_ids_flat >= 0

        hidden_flat = hidden_flat[valid_mask]
        word_ids_flat = word_ids_flat[valid_mask]

        # -------------------------------------------------
        # 5) Build (batch_word_index)
        # -------------------------------------------------
        # we need unique word indices per sentence
        # so shift each batch by max word index

        offset = torch.zeros(B, device=device, dtype=torch.long)

        max_words = word_ids.max(dim=1).values
        max_words = torch.nan_to_num(max_words, nan=0).long()

        offset[1:] = torch.cumsum(max_words[:-1] + 1, dim=0)

        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), S
        )[valid_mask]

        global_word_ids = word_ids_flat + offset[batch_ids]

        # -------------------------------------------------
        # 6) Scatter mean pooling (CORE OPTIMIZATION)
        # -------------------------------------------------
        num_words = global_word_ids.max().item() + 1

        word_reps = torch.zeros(num_words, H, device=device)
        counts = torch.zeros(num_words, 1, device=device)

        word_reps = word_reps.index_add(0, global_word_ids, hidden_flat)
        counts = counts.index_add(0, global_word_ids, torch.ones_like(global_word_ids, dtype=torch.float).unsqueeze(-1))

        word_reps = word_reps / counts.clamp(min=1.0)

        # -------------------------------------------------
        # 7) Split back into batch sequences
        # -------------------------------------------------
        lengths = max_words + 1
        max_len = lengths.max().item()

        out = torch.zeros(B, max_len, H, device=device)

        start = 0
        for b in range(B):
            l = lengths[b].item()
            out[b, :l] = word_reps[start:start + l]
            start += l

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