import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# -----------------------------
# Dataset
# -----------------------------
class MecoDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Group by trial for simplicity
        self.trials = self.data.groupby('unique_trial_id')

        self.trial_ids = list(self.trials.groups.keys())

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        trial_id = self.trial_ids[idx]
        trial_df = self.trials.get_group(trial_id)
        words = trial_df['word'].tolist()
        fix_durs = trial_df['dur'].tolist()  # target fixation durations

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        # Map subwords to words
        word_map = []
        for i, word_id in enumerate(encoding.word_ids()):
            if word_id is not None:
                word_map.append(word_id)
        word_map = torch.tensor(word_map)

        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(fix_durs,
                                                                                                     dtype=torch.float), word_map



class EyeXpertMModel(nn.Module):
    def __init__(
        self,
        num_languages,
        num_experts=4,
        pretrained_model="xlm-roberta-base",
        lstm_hidden=384,
        dropout=0.3,
        freeze_bottom_layers=6,
    ):
        super().__init__()

        # ---------------------------
        # Encoder
        # ---------------------------
        self.encoder = XLMRobertaModel.from_pretrained(pretrained_model)
        hidden_dim = self.encoder.config.hidden_size
        self.num_experts = num_experts

        # Freeze bottom layers (helps generalization on MECO)
        if freeze_bottom_layers > 0:
            for layer in self.encoder.encoder.layer[:freeze_bottom_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # ---------------------------
        # Word-level sequential modeling
        # ---------------------------
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.sequence_proj = nn.Linear(lstm_hidden * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ---------------------------
        # Language Embedding for Gating
        # ---------------------------
        self.lang_embedding = nn.Embedding(num_languages, 128)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # ---------------------------
        # Experts
        # ---------------------------
        self.expert_fix = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])

        self.expert_dur = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])

    # -----------------------------------------------------
    # Subword → Word Aggregation
    # -----------------------------------------------------
    def aggregate_words(self, token_embs, word_map):
        """
        token_embs: [L, H]
        word_map: [num_valid_tokens] (mapping subword index → word index)
        """
        word_embs = []
        max_word = word_map.max().item()

        for w in range(max_word + 1):
            idxs = (word_map == w).nonzero(as_tuple=True)[0]
            emb = token_embs[idxs].mean(dim=0)
            word_embs.append(emb)

        return torch.stack(word_embs)  # [W, H]

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, input_ids, attention_mask, word_maps, lang_ids):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embs = outputs.last_hidden_state  # [B, L, H]
        sentence_repr = token_embs[:, 0]        # CLS token [B, H]

        batch_size = input_ids.size(0)

        all_fix_preds = []
        all_dur_preds = []

        for b in range(batch_size):

            # ----- Word aggregation -----
            word_embs = self.aggregate_words(
                token_embs[b],
                word_maps[b]
            )  # [W, H]

            # ----- Sequential modeling -----
            word_embs = word_embs.unsqueeze(0)  # [1, W, H]
            seq_out, _ = self.bilstm(word_embs)
            seq_out = self.sequence_proj(seq_out.squeeze(0))
            seq_out = self.dropout(seq_out)  # [W, H]

            # ----- Gating -----
            lang_embed = self.lang_embedding(lang_ids[b])
            gate_input = torch.cat(
                [sentence_repr[b], lang_embed],
                dim=-1
            )

            gate_weights = torch.softmax(
                self.gate(gate_input),
                dim=-1
            )  # [num_experts]

            # ----- Expert Combination -----
            fix_pred = 0
            dur_pred = 0

            for i in range(self.num_experts):
                fix_pred += gate_weights[i] * self.expert_fix[i](seq_out)
                dur_pred += gate_weights[i] * self.expert_dur[i](seq_out)

            all_fix_preds.append(fix_pred.squeeze(-1))
            all_dur_preds.append(dur_pred.squeeze(-1))

        return all_fix_preds, all_dur_preds
