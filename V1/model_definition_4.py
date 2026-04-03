import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from torch.utils.data import Dataset
import pandas as pd

# -----------------------------
# Dataset
# -----------------------------
class MecoDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Group by trial
        self.trials = self.data.groupby('unique_trial_id')
        self.trial_ids = list(self.trials.groups.keys())

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        trial_id = self.trial_ids[idx]
        trial_df = self.trials.get_group(trial_id)

        # Words in order
        words = trial_df['word'].tolist()

        # Original word indices (0-based)
        fix_seq_orig = [int(w) - 1 for w in trial_df['wordnum'].tolist()]
        dur_seq = trial_df['dur'].tolist()

        # Tokenize words
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        # Map subwords → words
        word_ids = encoding.word_ids()
        word_map = [wid for wid in word_ids if wid is not None]
        word_map = torch.tensor(word_map, dtype=torch.long)

        # Keep only fixations that survived truncation
        valid_words = set(word_map.tolist())
        fix_seq_filtered = [w for w in fix_seq_orig if w in valid_words]
        dur_seq_filtered = [dur_seq[i] for i, w in enumerate(fix_seq_orig) if w in valid_words]

        # Convert to tensors
        fix_seq_tensor = torch.tensor(fix_seq_filtered, dtype=torch.long)
        dur_seq_tensor = torch.tensor(dur_seq_filtered, dtype=torch.float).unsqueeze(-1)  # [num_fix, 1]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'word_map': word_map,
            'fix_seq': fix_seq_tensor,
            'dur_seq': dur_seq_tensor
        }


# -----------------------------
# EyeXpertMModel
# -----------------------------
class EyeXpertMModel(nn.Module):
    def __init__(self, num_languages, num_experts=4, pretrained_model="xlm-roberta-base",
                 lstm_hidden=384, dropout=0.3, freeze_bottom_layers=6):
        super().__init__()

        # Encoder
        self.encoder = XLMRobertaModel.from_pretrained(pretrained_model)
        hidden_dim = self.encoder.config.hidden_size
        self.num_experts = num_experts

        # Freeze bottom layers
        if freeze_bottom_layers > 0:
            for layer in self.encoder.encoder.layer[:freeze_bottom_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Word-level BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.sequence_proj = nn.Linear(lstm_hidden * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Language embedding + gating
        self.lang_embedding = nn.Embedding(num_languages, 128)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # MoE expert transformations
        self.decoder_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_experts)
        ])

        # Autoregressive decoder
        self.decoder = nn.GRU(
            input_size=hidden_dim + 1,  # prev embedding + duration
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.next_fix_head = nn.Linear(hidden_dim, hidden_dim)
        self.dur_mean_head = nn.Linear(hidden_dim, 1)
        self.dur_logvar_head = nn.Linear(hidden_dim, 1)

    # Subword → word aggregation
    def aggregate_words(self, token_embs, word_map):
        word_embs = []
        max_word = word_map.max().item()
        for w in range(max_word + 1):
            idxs = (word_map == w).nonzero(as_tuple=True)[0]
            word_embs.append(token_embs[idxs].mean(dim=0))
        return torch.stack(word_embs)  # [W, H]

    # Forward
    def forward(self, input_ids, attention_mask, word_maps, lang_ids,
                target_fix_seq=None, target_dur_seq=None):

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # [B, L, H]
        batch_size = input_ids.size(0)

        all_fix_outputs, all_dur_means, all_dur_logvars, all_gate_weights = [], [], [], []

        for b in range(batch_size):
            # Word aggregation + BiLSTM
            word_embs = self.aggregate_words(token_embs[b], word_maps[b])  # [W, H]
            seq_out, _ = self.bilstm(word_embs.unsqueeze(0))
            seq_out = self.sequence_proj(seq_out.squeeze(0))
            seq_out = self.dropout(seq_out)
            W = seq_out.size(0)

            # Autoregressive decoder
            prev_fix = torch.zeros(1, dtype=torch.long, device=seq_out.device)
            prev_dur = torch.zeros(1, 1, device=seq_out.device)
            hidden = None

            fix_outputs, dur_means, dur_logvars, gate_trace = [], [], [], []

            steps = target_fix_seq[b].size(0) if target_fix_seq is not None else W

            for t in range(steps):
                # Fix tensor shapes
                prev_emb = seq_out[prev_fix]               # [1, hidden_dim]
                if prev_emb.dim() == 3:                   # handle batch dimension from seq_out[prev_fix]
                    prev_emb = prev_emb.squeeze(0)
                prev_dur = prev_dur.view(1, 1)           # [1,1]

                decoder_input = torch.cat([prev_emb, prev_dur], dim=-1).unsqueeze(0)  # [1,1,H+1]

                decoder_out, hidden = self.decoder(decoder_input, hidden)
                h_t = decoder_out.squeeze(0).squeeze(0)  # [H]

                # Dynamic gating
                lang_embed = self.lang_embedding(lang_ids[b])
                gate_input = torch.cat([h_t, lang_embed], dim=-1)
                gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
                gate_trace.append(gate_weights)

                # MoE
                expert_stack = torch.stack([expert(h_t) for expert in self.decoder_experts], dim=0)
                moe_hidden = torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=0)

                # Predict next fixation
                fix_logits = torch.matmul(seq_out, self.next_fix_head(moe_hidden))  # [W]

                # Predict duration
                dur_mean = self.dur_mean_head(moe_hidden)
                dur_logvar = torch.clamp(self.dur_logvar_head(moe_hidden), -10, 10)

                fix_outputs.append(fix_logits)
                dur_means.append(dur_mean.squeeze(-1))
                dur_logvars.append(dur_logvar.squeeze(-1))

                # Teacher forcing
                if target_fix_seq is not None:
                    prev_fix = target_fix_seq[b][t].unsqueeze(0)
                    prev_dur = target_dur_seq[b][t].view(1, 1)
                else:
                    prev_fix = fix_logits.argmax().unsqueeze(0)
                    prev_dur = dur_mean.detach()

            all_fix_outputs.append(fix_outputs)
            all_dur_means.append(dur_means)
            all_dur_logvars.append(dur_logvars)
            all_gate_weights.append(gate_trace)

        return all_fix_outputs, all_dur_means, all_dur_logvars, all_gate_weights