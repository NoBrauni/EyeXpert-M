import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer

class MultilingualMecoDataset(Dataset):
    """
    Combines multiple per-language MECO CSV files into a single dataset.
    Each trial has a language ID for model embeddings.
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Get all CSV files
        self.csv_files = [f for f in os.listdir(data_dir) if f.endswith("_processed.csv")]
        self.csv_files.sort()  # consistent ordering

        # Assign language IDs
        self.lang2id = {name: idx for idx, name in enumerate(self.csv_files)}
        self.trials = []  # list of (trial_df, lang_id)

        # Load all CSVs
        for csv_file in self.csv_files:
            lang_id = self.lang2id[csv_file]
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            grouped = df.groupby('unique_trial_id')
            for trial_id, trial_df in grouped:
                self.trials.append((trial_df, lang_id))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial_df, lang_id = self.trials[idx]

        words = trial_df['word'].tolist()
        fix_seq = trial_df['fix_index'].tolist()
        dur_seq = trial_df['dur'].tolist()

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        word_map = encoding.word_ids()
        word_map = [w if w is not None else -1 for w in word_map]
        word_map = torch.tensor(word_map, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'word_map': word_map,
            'fix_seq': torch.tensor(fix_seq, dtype=torch.long),
            'dur_seq': torch.tensor(dur_seq, dtype=torch.float),
            'lang_id': torch.tensor(lang_id, dtype=torch.long)
        }