import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaTokenizerFast
from model import LANG_FAMILY

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


# =========================================================
# ENCODING
# =========================================================
def encode(sentence):
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=False
    )

    word_ids = enc.word_ids(batch_index=0)
    if word_ids is None:
        word_ids = []

    word_ids = torch.tensor(
        [-1 if w is None else w for w in word_ids],
        dtype=torch.long
    )

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "word_ids": word_ids,
    }


# =========================================================
# DATASET
# =========================================================
class MECO(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        enc = encode(item["sentence"])

        scanpath = torch.tensor(item["scanpath"], dtype=torch.long)
        durations = torch.tensor(item["durations"], dtype=torch.float)

        return {
            **enc,
            "scanpath": scanpath,
            "durations": durations,

            # language conditioning
            "family": torch.tensor(LANG_FAMILY[item["lang"]], dtype=torch.long),
            "lang_id": torch.tensor(0, dtype=torch.long),

            # IMPORTANT: mask = valid timesteps
            "scanpath_mask": torch.ones(len(scanpath), dtype=torch.float)
        }


# =========================================================
# COLLATE
# =========================================================
def collate(batch):

    def pad(key, val):
        return pad_sequence(
            [b[key] for b in batch],
            batch_first=True,
            padding_value=val
        )

    return {
        "input_ids": pad("input_ids", 0),
        "attention_mask": pad("attention_mask", 0),
        "word_ids": pad("word_ids", -1),

        "scanpath": pad("scanpath", 0),
        "durations": pad("durations", 0.0),
        "scanpath_mask": pad("scanpath_mask", 0.0),

        "family": torch.stack([b["family"] for b in batch]),
        "lang_id": torch.stack([b["lang_id"] for b in batch]),
    }