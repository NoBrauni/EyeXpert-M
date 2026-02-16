import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from model_definition import EyeXpertMModel
from model_definition import MecoDataset


# -------------------------------------------------
# 1. Load small subset
# -------------------------------------------------

csv_path = "processed_data/meco_l1_en_processed.csv"

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
dataset = MecoDataset(csv_path, tokenizer)

# Take only first 10 trials
small_indices = list(range(10))
small_dataset = Subset(dataset, small_indices)

def collate_fn_small(batch):
    input_ids_list, attention_mask_list, fix_durs_list, word_map_list = zip(*batch)

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)

    # Dummy language IDs (single language test)
    lang_ids = torch.zeros(len(batch), dtype=torch.long)

    return input_ids, attention_mask, fix_durs_list, word_map_list, lang_ids


dataloader = DataLoader(
    small_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn_small
)

# -------------------------------------------------
# 2. Initialize model (small + frozen)
# -------------------------------------------------

model = EyeXpertMModel(
    num_languages=1,
    num_experts=2,
    freeze_bottom_layers=12  # freeze everything
)

# Freeze entire encoder for speed
for param in model.encoder.parameters():
    param.requires_grad = False

device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

mse_loss = nn.MSELoss()

# -------------------------------------------------
# 3. Tiny Training Loop
# -------------------------------------------------

model.train()

for epoch in range(20):
    total_loss = 0

    for input_ids, attention_mask, fix_durs_list, word_map_list, lang_ids in dataloader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        lang_ids = lang_ids.to(device)

        optimizer.zero_grad()

        _, dur_preds = model(
            input_ids,
            attention_mask,
            word_map_list,
            lang_ids
        )

        loss = 0

        for pred_dur, target_dur in zip(dur_preds, fix_durs_list):

            target_dur = target_dur.to(device)

            # Trim to smallest length (important!)
            min_len = min(len(pred_dur), len(target_dur))
            pred_dur = pred_dur[:min_len]
            target_dur = target_dur[:min_len]

            # Log transform improves numerical stability
            target_dur = torch.log1p(target_dur)

            loss += mse_loss(pred_dur, target_dur)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Test Epoch Loss: {total_loss:.4f}")

