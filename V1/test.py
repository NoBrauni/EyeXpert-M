# diagnostic_plot.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_definition_old import EyeTrackingDatasetMultilingual, TokenizerWrapper, CrossLingualExpertETModel
import numpy as np
from transformers import AutoTokenizer


#tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#df = pd.read_csv("processed_data/meco_l1_ge_processed.csv")
#print(df.keys().tolist())

import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv("processed_data/meco_l1_ge_processed.csv")
print(df.keys().tolist())
print("Max wordnum:", df["wordnum"].max())
print(df.head())
"""
lengths = []

for trial_id, trial_df in df.groupby("unique_trial_id"):
    words = trial_df["word"].tolist()
    enc = tokenizer(words, is_split_into_words=True)
    lengths.append(len(enc["input_ids"]))

print("Max token length:", max(lengths))
print("95th percentile:", sorted(lengths)[int(0.95*len(lengths))])



df = pd.read_csv("processed_data/meco_l1_en_processed.csv")

trial_id = df['unique_trial_id'].iloc[0]
trial_df = df[df['unique_trial_id'] == trial_id]

print(trial_df[['fix_index', 'wordnum', 'ianum', 'word', 'dur']].head(20))

# ------------------------
# 1. Files and language clusters
# ------------------------
csv_files = [
    "processed_data/meco_l1_ba_processed.csv",
    "processed_data/meco_l1_bp_processed.csv",
    "processed_data/meco_l1_da_processed.csv",
    "processed_data/meco_l1_du_processed.csv",
    "processed_data/meco_l1_ee_processed.csv",
    "processed_data/meco_l1_en_processed.csv",
    "processed_data/meco_l1_en_uk_processed.csv",
    "processed_data/meco_l1_fi_processed.csv",
    "processed_data/meco_l1_ge_po_processed.csv",
    "processed_data/meco_l1_ge_processed.csv",
    "processed_data/meco_l1_ge_zu_processed.csv",
    "processed_data/meco_l1_gr_processed.csv",
    "processed_data/meco_l1_he_processed.csv",
    "processed_data/meco_l1_hi_iiith_processed.csv",
    "processed_data/meco_l1_hi_iitk_processed.csv",
    "processed_data/meco_l1_ic_processed.csv",
    "processed_data/meco_l1_it_processed.csv",
    "processed_data/meco_l1_ko_processed.csv",
    "processed_data/meco_l1_no_processed.csv",
    "processed_data/meco_l1_ru_mo_processed.csv",
    "processed_data/meco_l1_ru_processed.csv",
    "processed_data/meco_l1_se_processed.csv",
    "processed_data/meco_l1_sp_ch_processed.csv",
    "processed_data/meco_l1_sp_processed.csv",
    "processed_data/meco_l1_tr_processed.csv"
]


import pandas as pd
import torch
from tokenizer import EyeTrackingTokenizer  # <-- replace with actual file

# ------------------------------
# Debug function
# ------------------------------
def debug_alignment(sentence, df_sentence, tokenizer_obj):
    encoding = tokenizer_obj.tokenize(sentence)

    input_ids = encoding["input_ids"]
    word_map = encoding["word_map"]
    words = encoding["words"]

    tokens = tokenizer_obj.tokenizer.convert_ids_to_tokens(input_ids)

    print("\n===================================================")
    print("Sentence:")
    print(sentence)

    print("\nCSV word count:", df_sentence["wordnum"].nunique())
    print("Tokenizer word count:", len(words))

    print("\nWords:")
    for i, w in enumerate(words):
        print(f"{i}: {w}")

    print("\nSubword Alignment:")
    for i, (tok, widx) in enumerate(zip(tokens, word_map.tolist())):
        print(f"{i:3d} | {tok:15s} | word_id = {widx}")

    used_word_ids = sorted(set([w for w in word_map.tolist() if w >= 0]))
    expected = list(range(len(words)))

    print("\nWord Coverage:")
    print("Used word IDs:", used_word_ids)
    print("Expected:", expected)

    missing = set(expected) - set(used_word_ids)
    if len(missing) == 0:
        print("✅ All words covered")
    else:
        print("❌ Missing word IDs:", missing)


# ------------------------------
# Main test
# ------------------------------
if __name__ == "__main__":

    csv_path = "processed_data/meco_l1_en_processed.csv"  # <-- change to your CSV
    df = pd.read_csv(csv_path)

    tokenizer = EyeTrackingTokenizer(model_name="xlm-roberta-base", max_len=128)

    # Pick first 3 unique sentences
    unique_sentences = df["sent"].unique()[:3]

    for sentence in unique_sentences:
        df_sentence = df[df["sent"] == sentence]
        debug_alignment(sentence, df_sentence, tokenizer)



lang2cluster = {
    "ru": 0, "ru_mo": 0, "ba": 0, "bp": 0,
    "en": 1, "en_uk": 1, "du": 1,
    "da": 2, "no": 2, "se": 2,
    "it": 3, "sp": 3, "sp_ch": 3,
    "ge": 4, "ge_po": 4, "ge_zu": 4,
    "hi_iiith": 5, "hi_iitk": 5,
    "he": 6, "gr": 6,
    "ko": 7, "ic": 7,
    "tr": 8,
    "ee": 9, "fi": 9,
    "unknown": 10
}

# ------------------------
# 2. Tokenizer and dataset
# ------------------------
tokenizer_wrapper = TokenizerWrapper(max_len=128)
dataset = EyeTrackingDatasetMultilingual(csv_files, tokenizer_wrapper, lang2cluster)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ------------------------
# 3. Load model
# ------------------------
num_experts = max(lang2cluster.values()) + 1
model = CrossLingualExpertETModel(num_experts=num_experts, hidden_dim=768, max_len=128)
device = torch.device("cpu")
model.to(device)
model.eval()

# ------------------------
# 4. Run one batch
# ------------------------
batch = next(iter(dataloader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
lang_id = batch["lang_id"].to(device)
fix_seq_target = batch["fix_seq"]
fix_dur_target = batch["fix_dur"]

with torch.no_grad():
    fix_seq_pred, fix_dur_pred = model(input_ids, attention_mask, lang_id)

# ------------------------
# 5. Plot predictions vs targets
# ------------------------
num_tokens_to_plot = min(5, fix_seq_pred.shape[1])
num_fixations_to_plot = 10


def to_numpy(tensor):
    return tensor.cpu().numpy() if tensor.dim() > 0 else np.array([tensor.item()])


for i in range(num_tokens_to_plot):
    pred_seq = to_numpy(fix_seq_pred[0][i])[:num_fixations_to_plot]
    target_seq = to_numpy(fix_seq_target[0][i])[:num_fixations_to_plot]

    pred_dur = to_numpy(fix_dur_pred[0][i])[:num_fixations_to_plot]
    target_dur = to_numpy(fix_dur_target[0][i])[:num_fixations_to_plot]

    plt.figure(figsize=(10, 4))

    # Sequence
    plt.subplot(1, 2, 1)
    plt.plot(pred_seq, 'o-', label='Predicted')
    plt.plot(target_seq, 'x-', label='Target')
    plt.title(f'Token {i + 1} - Fixation sequence')
    plt.xlabel('Fixation index')
    plt.ylabel('Value')
    plt.legend()

    # Duration
    plt.subplot(1, 2, 2)
    plt.plot(pred_dur, 'o-', label='Predicted')
    plt.plot(target_dur, 'x-', label='Target')
    plt.title(f'Token {i + 1} - Fixation durations')
    plt.xlabel('Fixation index')
    plt.ylabel('Duration')
    plt.legend()

    plt.tight_layout()
    plt.show()

"""