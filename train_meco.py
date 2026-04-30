import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast

from model import LANG_FAMILY

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


# =========================================================
# CACHE
# =========================================================
GLOBAL_CACHE = {}


def encode_sentence(tokenizer, sentence):
    enc = tokenizer(
        sentence,
        truncation=True,
        padding=False,
        return_tensors="pt"
    )

    word_ids = enc.word_ids(batch_index=0)
    word_ids = [-1 if x is None else x for x in word_ids]

    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "word_ids": torch.tensor(word_ids, dtype=torch.long),
    }


# =========================================================
# DATASET
# =========================================================
class MECO(Dataset):
    def __init__(self, data, tokenizer, cache=GLOBAL_CACHE):
        self.data = data
        self.tokenizer = tokenizer
        self.cache = cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        item = self.data[i]
        sentence = item["sentence"]

        # -------------------------
        # CACHE HIT / MISS
        # -------------------------
        if sentence in self.cache:
            cached = self.cache[sentence]
        else:
            cached = encode_sentence(self.tokenizer, sentence)
            self.cache[sentence] = cached

        scanpath = torch.tensor(item["scanpath"], dtype=torch.long)
        durations = torch.tensor(item["durations"], dtype=torch.float)

        return {
            **cached,

            "scanpath": scanpath,
            "durations": durations,

            "family": torch.tensor(LANG_FAMILY[item["lang"]], dtype=torch.long),
            "lang_id": torch.tensor(0, dtype=torch.long)
        }


# =========================================================
# COLLATE
# =========================================================
def collate(batch):

    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]

    input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    word_ids = [b["word_ids"] for b in batch]
    word_ids = rnn.pad_sequence(word_ids, batch_first=True, padding_value=-1)

    scanpath = [b["scanpath"] for b in batch]
    scanpath = rnn.pad_sequence(scanpath, batch_first=True, padding_value=0)

    durations = [b["durations"] for b in batch]
    durations = rnn.pad_sequence(durations, batch_first=True, padding_value=0.0)

    mask = (scanpath != 0).float()

    family = torch.tensor([b["family"] for b in batch], dtype=torch.long)
    lang_id = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": word_ids,

        "scanpath": scanpath,
        "durations": durations,
        "mask": mask,

        "family": family,
        "lang_id": lang_id
    }