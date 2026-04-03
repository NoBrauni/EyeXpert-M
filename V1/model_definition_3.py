import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel

class MecoDataset(Dataset):
    """Dataset wrapper for MECO eye-tracking CSVs with normalization of durations."""

    def __init__(self, csv_file, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trials = self.data.groupby('unique_sentence_id')
        self.trial_ids = list(self.trials.groups.keys())

        # Compute global mean/std for duration normalization
        self.dur_mean = self.data['dur'].mean()
        self.dur_std = self.data['dur'].std()

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        trial_id = self.trial_ids[idx]
        trial_df = self.trials.get_group(trial_id)

        words = trial_df['word'].tolist()
        fix_seq = [max(0, int(w) - 1) for w in trial_df['wordnum'].tolist()]

        # Normalize durations
        dur_seq = [(d - self.dur_mean) / self.dur_std for d in trial_df['dur'].tolist()]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        word_ids = encoding.word_ids()
        valid_word_ids = sorted(set([w for w in word_ids if w is not None]))

        fix_seq_filtered = [fix_seq[w] for w in valid_word_ids] or [0]
        dur_seq_filtered = [dur_seq[w] for w in valid_word_ids] or [0.0]

        word_map_tensor = torch.tensor([w for w in word_ids if w is not None], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'word_map': word_map_tensor,
            'fix_seq': torch.tensor(fix_seq_filtered, dtype=torch.long),
            'dur_seq': torch.tensor(dur_seq_filtered, dtype=torch.float)
        }


class EyeXpertModel(nn.Module):
    """Full EyeXpert model: XLM-R encoder + BiLSTM + autoregressive MoE decoder."""

    def __init__(self, hidden_dim=768, lstm_hidden=384, dropout=0.3,
                 num_languages=1, num_experts=4):
        super().__init__()
        # Encoder + BiLSTM
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Autoregressive decoder with MoE
        self.decoder = EyeXpertStep4(hidden_dim=hidden_dim,
                                     num_experts=num_experts,
                                     num_languages=num_languages)

    @staticmethod
    def aggregate_words(token_embs, word_map):
        if word_map.numel() == 0:
            return torch.zeros(1, token_embs.size(-1), device=token_embs.device)
        return torch.stack([token_embs[(word_map == w).nonzero(as_tuple=True)[0]].mean(dim=0)
                            for w in range(word_map.max().item() + 1)])

    def forward(self, input_ids, attention_mask, word_maps, lang_ids,
                target_fix_seq=None, target_dur_seq=None):
        B = input_ids.size(0)
        device = input_ids.device

        # Encode tokens
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        # Word-level aggregation + BiLSTM
        word_emb_list = []
        lengths = []
        max_len = 0
        for b in range(B):
            word_embs = self.aggregate_words(token_embs[b], word_maps[b])
            word_emb_list.append(word_embs)
            lengths.append(word_embs.size(0))
            max_len = max(max_len, word_embs.size(0))

        # Pad word embeddings
        padded = torch.zeros(B, max_len, token_embs.size(-1), device=device)
        for b, we in enumerate(word_emb_list):
            padded[b, :lengths[b]] = we

        # BiLSTM
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_len)
        seq_out = self.proj(lstm_out)
        seq_out = self.dropout(seq_out)

        # Autoregressive decoder with MoE
        fix_out, dur_mean_out, dur_logvar_out, gate_trace = self.decoder(
            seq_out, lengths, lang_ids, target_fix_seq, target_dur_seq
        )

        return fix_out, dur_mean_out, dur_logvar_out, gate_trace, lengths


class EyeXpertStep4(nn.Module):
    """Autoregressive decoder with mixture-of-experts (MoE) and language gating."""

    def __init__(self, hidden_dim=768, num_experts=4, num_languages=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # GRU decoder
        self.decoder = nn.GRU(input_size=hidden_dim+1, hidden_size=hidden_dim, batch_first=True)

        # Prediction heads
        self.next_fix_head = nn.Linear(hidden_dim, hidden_dim)
        self.dur_mean_head = nn.Linear(hidden_dim, 1)
        self.dur_logvar_head = nn.Linear(hidden_dim, 1)

        # Language embedding + gating
        self.lang_embedding = nn.Embedding(num_languages, 128)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # MoE experts
        self.decoder_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, seq_out, lengths, lang_ids=None, target_fix_seq=None, target_dur_seq=None):
        B, max_len, H = seq_out.size()
        device = seq_out.device

        all_fix_outputs = []
        all_dur_means = []
        all_dur_logvars = []
        all_gate_weights = []

        for b in range(B):
            W = lengths[b]
            hidden = None
            prev_fix = torch.tensor([0], device=device)
            prev_dur = torch.tensor([0.0], device=device)

            fix_outputs, dur_means, dur_logvars, gate_trace = [], [], [], []

            steps = W if target_fix_seq is None else target_fix_seq[b].size(0)
            lang_embed = self.lang_embedding(lang_ids[b]) if lang_ids is not None else torch.zeros(128, device=device)

            for t in range(steps):
                prev_emb = seq_out[b, prev_fix.item()].unsqueeze(0).unsqueeze(0)
                prev_dur_3d = prev_dur.view(1, 1, 1)
                decoder_in = torch.cat([prev_emb, prev_dur_3d], dim=-1)

                out, hidden = self.decoder(decoder_in, hidden)
                h_t = out.squeeze(0).squeeze(0)

                # MoE gating
                gate_input = torch.cat([h_t, lang_embed], dim=-1)
                gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
                gate_trace.append(gate_weights)

                # Expert combination
                expert_stack = torch.stack([expert(h_t) for expert in self.decoder_experts], dim=0)
                moe_hidden = torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=0)

                # Predictions
                fix_logits = torch.matmul(seq_out[b], self.next_fix_head(moe_hidden))
                dur_mean = self.dur_mean_head(moe_hidden).squeeze(-1)
                dur_logvar = self.dur_logvar_head(moe_hidden).squeeze(-1)

                fix_outputs.append(fix_logits)
                dur_means.append(dur_mean)
                dur_logvars.append(dur_logvar)

                # Teacher forcing
                if target_fix_seq is not None:
                    prev_fix = target_fix_seq[b][t].clamp(0, W-1)
                    prev_dur = target_dur_seq[b][t]
                else:
                    prev_fix = fix_logits.argmax().unsqueeze(0)
                    prev_dur = dur_mean.detach()

            all_fix_outputs.append(fix_outputs)
            all_dur_means.append(dur_means)
            all_dur_logvars.append(dur_logvars)
            all_gate_weights.append(gate_trace)

        return all_fix_outputs, all_dur_means, all_dur_logvars, all_gate_weights