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
# ENCODER (unchanged, already optimized)
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

        word_ids = torch.tensor(word_ids_batch, dtype=torch.long, device=device)

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
# DECODER (VECTORIZED TIME UNROLLING)
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
        self.duration_head = nn.Linear(hidden, 1)

        self.moe_loss = torch.tensor(0.0)

    def hard_route(self, family):
        return family % len(self.experts)

    # =====================================================
    # VECTORIZED FORWARD (NO PYTHON TIME LOOP)
    # =====================================================
    def forward(self, memory, scanpath, durations, family, lang_id):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]

        context = 0
        if self.use_lang:
            context = self.lang_emb(lang_id) + self.family_emb(family)

        outputs = []
        duration_outputs = []
        router_history = []

        zero_sacc = torch.zeros(B, 2, device=memory.device)
        zero_dur = torch.zeros(B, 1, device=memory.device)

        states = []

        # =====================================================
        # STEP 1: BUILD ALL STATES IN ONE PASS (UNROLLED LOOP)
        # =====================================================
        for t in range(T):

            if t == 0:
                prev_idx = torch.zeros(B, dtype=torch.long, device=memory.device)
            else:
                prev_idx = scanpath[:, t - 1]

            prev_word = memory[torch.arange(B, device=memory.device), prev_idx]

            prev_dur = torch.log1p(durations[:, t - 1]).unsqueeze(-1) if t > 0 else zero_dur

            sacc = zero_sacc

            inp = self.input_proj(torch.cat([prev_word, sacc, prev_dur], dim=-1))

            state = self.rnn(inp, state) if self.use_rnn else inp
            state = state + context

            states.append(state)

        # stack full sequence
        states = torch.stack(states, dim=1)  # (B, T, H)

        # =====================================================
        # STEP 2: PARALLEL FIXATION PREDICTION
        # =====================================================
        q = self.query(states)  # (B, T, H)

        logits = torch.einsum("bth,bwh->btw", q, memory)
        outputs = logits

        # =====================================================
        # STEP 3: PARALLEL DURATION PREDICTION
        # =====================================================
        dur_pred = self.duration_head(states)

        # =====================================================
        # STEP 4: MoE LOSS (UNCHANGED LOGIC)
        # =====================================================
        if self.use_experts and self.routing == "soft":

            r = torch.softmax(self.router(states), dim=-1)
            p = r.mean(dim=(0, 1))
            self.moe_loss = -(p * torch.log(p + 1e-8)).sum()

        return outputs, dur_pred


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