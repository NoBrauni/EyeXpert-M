import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast

from model import LANG_FAMILY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


# =========================================================
# DATASET
# =========================================================
class MECO(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        item = self.data[i]

        # ONE tokenizer call (important)
        enc = tokenizer(
            item["sentence"],
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        return {
            # NLP
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "word_ids": enc.word_ids(batch_index=0),  # critical

            # eye-tracking
            "scanpath": torch.tensor(item["scanpath"], dtype=torch.long),
            "durations": torch.tensor(item["durations"], dtype=torch.float),

            # meta
            "family": torch.tensor(LANG_FAMILY[item["lang"]], dtype=torch.long),
            "lang_id": torch.tensor(0)
        }


# =========================================================
# COLLATE
# =========================================================
def collate(batch):

    out = {}

    # -------------------------
    # TOKEN SEQUENCES
    # -------------------------
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]

    input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attn = rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    out["input_ids"] = input_ids
    out["attention_mask"] = attn

    # -------------------------
    # WORD IDS (pad with None)
    # -------------------------
    max_len = input_ids.size(1)

    word_ids = []
    for b in batch:
        w = b["word_ids"]
        w = w + [None] * (max_len - len(w))
        word_ids.append(w)

    out["word_ids"] = word_ids

    # -------------------------
    # SCANPATH + DURATIONS
    # -------------------------
    scan = [b["scanpath"] for b in batch]
    dur = [b["durations"] for b in batch]

    out["scanpath"] = rnn.pad_sequence(scan, batch_first=True, padding_value=0)
    out["durations"] = rnn.pad_sequence(dur, batch_first=True, padding_value=0)

    # -------------------------
    # MASK (for loss)
    # -------------------------
    lengths = [len(b["scanpath"]) for b in batch]
    max_len = max(lengths)

    mask = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    out["mask"] = mask

    # -------------------------
    # META
    # -------------------------
    out["family"] = torch.tensor([b["family"] for b in batch], dtype=torch.long)
    out["lang_id"] = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    return out