import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast

from model import LANG_FAMILY

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

        enc = tokenizer(
            item["sentence"],
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        # -------------------------
        # WORD IDS (CLEAN)
        # -------------------------
        word_ids = enc.word_ids(batch_index=0)
        word_ids = [-1 if x is None else x for x in word_ids]

        # -------------------------
        # REQUIRED SIGNALS (NO OPTIONAL MODES)
        # -------------------------
        scanpath = torch.tensor(item["scanpath"], dtype=torch.long)
        durations = torch.tensor(item["durations"], dtype=torch.float)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),

            "word_ids": torch.tensor(word_ids, dtype=torch.long),

            "scanpath": scanpath,
            "durations": durations,

            "family": torch.tensor(LANG_FAMILY[item["lang"]], dtype=torch.long),
            "lang_id": torch.tensor(0)
        }


# =========================================================
# COLLATE FUNCTION (CLEAN + GPU SAFE)
# =========================================================
def collate(batch):

    out = {}

    # -------------------------
    # TOKEN INPUTS
    # -------------------------
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]

    out["input_ids"] = rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    out["attention_mask"] = rnn.pad_sequence(attn, batch_first=True, padding_value=0)

    # -------------------------
    # WORD IDS
    # -------------------------
    word_ids = [b["word_ids"] for b in batch]
    out["word_ids"] = rnn.pad_sequence(word_ids, batch_first=True, padding_value=-1)

    # -------------------------
    # SCANPATH
    # -------------------------
    scan = [b["scanpath"] for b in batch]
    out["scanpath"] = rnn.pad_sequence(scan, batch_first=True, padding_value=0)

    # -------------------------
    # DURATIONS (ALWAYS PRESENT)
    # -------------------------
    dur = [b["durations"] for b in batch]
    out["durations"] = rnn.pad_sequence(dur, batch_first=True, padding_value=0.0)

    # -------------------------
    # MASK (for loss weighting)
    # -------------------------
    lengths = [len(b["scanpath"]) for b in batch]
    max_len = max(lengths)

    mask = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    out["mask"] = mask

    # -------------------------
    # META DATA
    # -------------------------
    out["family"] = torch.tensor([b["family"] for b in batch], dtype=torch.long)
    out["lang_id"] = torch.tensor([b["lang_id"] for b in batch], dtype=torch.long)

    return out