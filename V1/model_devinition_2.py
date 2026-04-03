import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# ----------------------
# Language → Cluster mapping
# ----------------------
LANG_TO_CLUSTER = {
    "en": 0, "en_uk": 0,
    "ge": 0, "ge_po": 0, "ge_zu": 0,
    "du": 0,
    "da": 1, "no": 1, "se": 1,
    "sp": 2, "sp_ch": 2, "it": 2,
    "fi": 3, "ee": 3,
    "ru": 4, "ru_mo": 4, "ic": 4,
}


# ----------------------
# Dataset
# ----------------------
class MecoDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.trials = self.data.groupby('unique_sentence_id')
        self.trial_ids = list(self.trials.groups.keys())
        self.dur_mean = self.data['dur'].mean()
        self.dur_std = self.data['dur'].std()

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        trial_id = self.trial_ids[idx]
        trial_df = self.trials.get_group(trial_id)
        words = trial_df['word'].tolist()
        fix_seq = [max(0, int(w) - 1) for w in trial_df['wordnum'].tolist()]
        dur_seq = [(d - self.dur_mean) / self.dur_std for d in trial_df['dur'].tolist()]
        lang = trial_df['lang'].iloc[0]
        cluster_id = LANG_TO_CLUSTER.get(lang, 0)

        encoding = self.tokenizer(
            words, is_split_into_words=True, return_tensors='pt',
            padding='max_length', truncation=True, max_length=self.max_len
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
            'dur_seq': torch.tensor(dur_seq_filtered, dtype=torch.float),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long)
        }


# ----------------------
# Collate
# ----------------------
def collate_fn(batch):
    max_input_len = max([b['input_ids'].size(0) for b in batch])
    max_word_len = max([b['word_map'].size(0) for b in batch])
    max_dur_len = max([b['dur_seq'].size(0) for b in batch])

    input_ids = torch.stack(
        [torch.cat([b['input_ids'], torch.zeros(max_input_len - b['input_ids'].size(0), dtype=torch.long)]) for b in
         batch])
    attention_mask = torch.stack(
        [torch.cat([b['attention_mask'], torch.zeros(max_input_len - b['attention_mask'].size(0), dtype=torch.long)])
         for b in batch])
    word_maps = torch.stack(
        [torch.cat([b['word_map'], torch.zeros(max_word_len - b['word_map'].size(0), dtype=torch.long)]) for b in
         batch])
    dur_seq = torch.stack([torch.cat([b['dur_seq'], torch.zeros(max_dur_len - b['dur_seq'].size(0))]) for b in batch])
    cluster_ids = torch.stack([b['cluster_id'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'word_map': word_maps, 'dur_seq': dur_seq,
            'cluster_id': cluster_ids}


# ----------------------
# Vectorized Cross-lingual MoE EyeXpert
# ----------------------
class CrossLingualMoEVectorized(nn.Module):
    def __init__(self, hidden_dim=768, lstm_hidden=384, num_clusters=5,
                 experts_per_cluster=2, lang_embedding_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.bilstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.cluster_embedding = nn.Embedding(num_clusters, lang_embedding_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim + lang_embedding_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, experts_per_cluster))
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.decoder_experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                            nn.ReLU(),
                                                            nn.Linear(hidden_dim, hidden_dim))
                                              for _ in range(num_clusters * experts_per_cluster)])
        self.next_fix_head = nn.Linear(hidden_dim, hidden_dim)
        self.dur_mean_head = nn.Linear(hidden_dim, 1)
        self.dur_logvar_head = nn.Linear(hidden_dim, 1)
        self.decoder_gru = nn.GRU(hidden_dim + 1, hidden_dim, batch_first=True)

    @staticmethod
    def aggregate_words(token_embs, word_map):
        return torch.stack([token_embs[(word_map == w).nonzero(as_tuple=True)[0]].mean(dim=0)
                            if (word_map == w).any() else torch.zeros(token_embs.size(-1), device=token_embs.device)
                            for w in range(word_map.max().item() + 1)])

    def forward(self, input_ids, attention_mask, word_maps, cluster_ids, target_dur_seq=None):
        B = input_ids.size(0)
        device = input_ids.device
        token_embs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Aggregate words
        word_emb_list = []
        lengths = []
        max_len = 0
        for b in range(B):
            word_embs = self.aggregate_words(token_embs[b], word_maps[b])
            word_emb_list.append(word_embs)
            lengths.append(word_embs.size(0))
            max_len = max(max_len, word_embs.size(0))
        padded = torch.zeros(B, max_len, token_embs.size(-1), device=device)
        for b, we in enumerate(word_emb_list):
            padded[b, :lengths[b]] = we

        # BiLSTM
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_len)
        seq_out = self.dropout(self.proj(lstm_out))

        # -----------------------
        # Vectorized autoregressive decoding
        # -----------------------
        max_steps = max(lengths)
        hidden = torch.zeros(1, B, seq_out.size(-1), device=device)
        prev_dur = torch.zeros(B, 1, 1, device=device)
        all_dur_means = []

        cluster_embed = self.cluster_embedding(cluster_ids)  # [B, emb]
        experts_per_batch = []
        for i, cid in enumerate(cluster_ids):
            start = cid * self.experts_per_cluster
            end = start + self.experts_per_cluster
            experts_per_batch.append(self.decoder_experts[start:end])

        for t in range(max_steps):
            # Gather previous embeddings
            idxs = torch.clamp(torch.tensor([min(t, l - 1) for l in lengths], device=device), 0, max_steps - 1)
            prev_emb = seq_out[torch.arange(B, device=device), idxs].unsqueeze(1)  # [B,1,H]
            decoder_in = torch.cat([prev_emb, prev_dur], dim=-1)
            out, hidden = self.decoder_gru(decoder_in, hidden)
            h_t = out.squeeze(1)

            # MoE gating & expert combination
            gate_weights = torch.stack(
                [torch.softmax(self.gate(torch.cat([h_t[i], cluster_embed[i]])), 0) for i in range(B)])
            moe_hidden = torch.stack([torch.sum(
                torch.stack([experts_per_batch[i][k](h_t[i]) for k in range(self.experts_per_cluster)]) * gate_weights[
                    i].unsqueeze(-1), 0)
                                      for i in range(B)])

            dur_mean = self.dur_mean_head(moe_hidden).squeeze(-1)
            all_dur_means.append(dur_mean)
            prev_dur = dur_mean.view(B, 1, 1) if target_dur_seq is None else target_dur_seq[:, t].view(B, 1, 1)

        all_dur_means = torch.stack(all_dur_means, dim=1)  # [B, max_len]
        return all_dur_means, lengths