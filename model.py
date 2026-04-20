import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cpu"

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

LANG_FAMILY = {
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0,
    "sp": 1, "sp_ch": 1, "it": 1, "bp": 1,
    "no": 2, "da": 2, "ic": 2, "se": 2,
    "ru": 3, "ru_mo": 3,
    "fi": 4, "ee": 4,
}


# =========================================================
# ENCODER (Frozen XLM-R)
# =========================================================
class XLMREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, sentences):
        enc = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        out = self.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )

        return out.last_hidden_state, enc


# =========================================================
# WORD POOLING (unchanged but stable)
# =========================================================
def pool_to_words(hidden_states, encodings, sentences):

    batch_out = []

    for i in range(len(sentences)):

        word_ids = encodings.word_ids(batch_index=i)
        hs = hidden_states[i]

        groups = {}

        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            groups.setdefault(w_id, []).append(hs[idx])

        words = []

        for k in sorted(groups.keys()):
            words.append(torch.stack(groups[k]).mean(0))

        batch_out.append(torch.stack(words))

    return batch_out


# =========================================================
# DECODER (clean ablation-safe design)
# =========================================================
class Decoder(nn.Module):

    def __init__(
        self,
        hidden,
        n_experts=3,
        use_moe=True,
        use_saccade=True,
        use_lang=True,
        use_rnn=True
    ):
        super().__init__()

        self.hidden = hidden

        self.use_moe = use_moe
        self.use_saccade = use_saccade
        self.use_lang = use_lang
        self.use_rnn = use_rnn

        # -------------------------
        # INPUT projection (fixed size!)
        # -------------------------
        self.input_proj = nn.Linear(hidden + 2, hidden)

        # -------------------------
        # Recurrent state model (ablated cleanly)
        # -------------------------
        self.rnn = nn.GRUCell(hidden, hidden)

        # -------------------------
        # MoE experts (state update only)
        # -------------------------
        self.experts = nn.ModuleList([
            nn.GRUCell(hidden, hidden)
            for _ in range(n_experts)
        ])

        self.router = nn.Linear(hidden, n_experts)

        # -------------------------
        # Pointer query
        # -------------------------
        self.query = nn.Linear(hidden, hidden)

        # -------------------------
        # Language / family embeddings
        # -------------------------
        self.family_emb = nn.Embedding(6, hidden)
        self.lang_emb = nn.Embedding(len(LANG_FAMILY) + 1, hidden)

        # -------------------------
        # MoE diagnostics
        # -------------------------
        self.moe_loss = None

    # -----------------------------------------------------
    # SAFE SACCADE FEATURE (FIXED)
    # -----------------------------------------------------
    def compute_saccade(self, scanpath, t):

        if not self.use_saccade or t < 2:
            return torch.zeros(scanpath.size(0), 2, device=scanpath.device)

        delta = scanpath[:, t-1] - scanpath[:, t-2]

        return torch.stack([
            delta.float(),
            delta.abs().float()
        ], dim=-1)

    # -----------------------------------------------------
    def forward(self, memory, scanpath, durations, family, lang_id):

        B, W, H = memory.shape
        T = scanpath.shape[1]

        state = memory[:, 0]

        outputs = []
        router_hist = []

        # language context (clean switch)
        if self.use_lang:
            context = self.family_emb(family) + self.lang_emb(lang_id)
        else:
            context = torch.zeros_like(state)

        for t in range(T):

            # previous fixation
            prev_idx = scanpath[:, t-1] if t > 0 else torch.zeros(
                B, dtype=torch.long, device=memory.device
            )

            prev_word = memory[torch.arange(B), prev_idx]

            # saccade (clean ablation signal)
            sacc = self.compute_saccade(scanpath, t)

            # input construction (stable across ablations)
            inp = torch.cat([prev_word, sacc], dim=-1)
            inp = self.input_proj(inp)

            # optional recurrence ablation
            if self.use_rnn:
                state = self.rnn(inp, state)
            else:
                state = inp

            # MoE update (clean separation)
            if self.use_moe:

                weights = torch.softmax(self.router(state), dim=-1)
                router_hist.append(weights)

                new_state = 0

                for i, expert in enumerate(self.experts):
                    new_state = new_state + weights[:, i].unsqueeze(-1) * expert(state, state)

                state = new_state

            state = state + context

            q = self.query(state)

            logits = torch.einsum("bh,bwh->bw", q, memory)

            outputs.append(logits.unsqueeze(1))

        # -------------------------
        # MoE entropy penalty (stable)
        # -------------------------
        if self.use_moe and len(router_hist) > 0:

            r = torch.stack(router_hist, dim=1)  # [B,T,E]
            p = r.reshape(-1, r.size(-1)).mean(dim=0)

            self.moe_loss = -(p * torch.log(p + 1e-8)).sum()
        else:
            self.moe_loss = torch.tensor(0.0, device=memory.device)

        return torch.cat(outputs, dim=1), state


# =========================================================
# DURATION HEAD
# =========================================================
class DurationHead(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, state, word_emb):
        return self.mlp(torch.cat([state, word_emb], dim=-1)).squeeze(-1)


# =========================================================
# FULL MODEL (A0–A4 CLEAN SUPPORT)
# =========================================================
class EyeXpertXLMR(nn.Module):

    def __init__(
        self,
        use_moe=True,
        use_saccade=True,
        use_lang=True,
        use_rnn=True
    ):
        super().__init__()

        self.encoder = XLMREncoder()

        hidden = self.encoder.encoder.config.hidden_size

        self.decoder = Decoder(
            hidden,
            use_moe=use_moe,
            use_saccade=use_saccade,
            use_lang=use_lang,
            use_rnn=use_rnn
        )

        self.duration_head = DurationHead(hidden)

    def forward(self, sentences, scanpath, durations, family, lang_id):

        token_memory, enc = self.encoder(sentences)

        memory_list = pool_to_words(token_memory, enc, sentences)

        B = len(memory_list)
        W = max(x.size(0) for x in memory_list)
        H = memory_list[0].size(1)

        memory = torch.zeros(B, W, H, device=token_memory.device)

        for i, m in enumerate(memory_list):
            memory[i, :m.size(0)] = m

        logits, state = self.decoder(
            memory,
            scanpath,
            durations,
            family,
            lang_id
        )

        B, T = scanpath.shape

        word_embs = memory[torch.arange(B).unsqueeze(1), scanpath]

        dur_pred = self.duration_head(
            state.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1),
            word_embs.reshape(B * T, -1)
        ).view(B, T)

        return logits, dur_pred, self.decoder.moe_loss