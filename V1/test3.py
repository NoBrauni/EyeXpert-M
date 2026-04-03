import random
import torch
from model_definition import MECODataset
from test_model import collate_batch
from model_definition import EyeExpertM, LANG_TO_EXPERT, batch_precompute_embeddings

import os
import pickle
import pandas as pd
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # for progress bar
from model_definition import  MECODataset

# ===============================
# 1️⃣ Configuration
# ===============================
csv_paths = [
    "processed_data/meco_l1_ge_processed.csv",
    "processed_data/meco_l1_en_processed.csv",
    "processed_data/meco_l1_no_processed.csv",
    "processed_data/meco_l1_se_processed.csv",
    "processed_data/meco_l1_da_processed.csv",
    "processed_data/meco_l1_sp_processed.csv",
    "processed_data/meco_l1_sp_ch_processed.csv",
    "processed_data/meco_l1_it_processed.csv",
    "processed_data/meco_l1_bp_processed.csv",
    "processed_data/meco_l1_ru_processed.csv",
    "processed_data/meco_l1_ru_mo_processed.csv",
    "processed_data/meco_l1_fi_processed.csv",
    "processed_data/meco_l1_ee_processed.csv",
]

cache_path = "embeddings_parallel.pkl"
batch_size = 16
n_workers = max(1, cpu_count() - 1)  # leave one core free

# ===============================
# 2️⃣ Load all samples
# ===============================

all_samples = []
for path in csv_paths:
    ds = MECODataset(path)
    all_samples.extend(ds.samples)

print(f"Total samples loaded: {len(all_samples)}")

# ===============================
# 3️⃣ Load XLM-R
# ===============================
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# ===============================
# 4️⃣ Embedding worker
# ===============================
def compute_batch_embeddings(batch_sentences):
    encoding = tokenizer(batch_sentences, return_tensors="pt", padding=True,
                         truncation=True, return_offsets_mapping=True)
    with torch.no_grad():
        outputs = encoder(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
    hidden_states = outputs.last_hidden_state
    embeddings = []

    for i, sentence in enumerate(batch_sentences):
        offsets = encoding["offset_mapping"][i]
        words = sentence.split()  # simple tokenization by space
        word_embeddings = []
        for w in words:
            vecs = []
            for j, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue
                token_text = sentence[start:end]
                if token_text.strip() == w.strip():
                    vecs.append(hidden_states[i, j])
            if vecs:
                word_embeddings.append(torch.stack(vecs).mean(dim=0))
            else:
                word_embeddings.append(torch.zeros(hidden_states.size(-1)))
        embeddings.append(torch.stack(word_embeddings))
    return embeddings

# ===============================
# 5️⃣ Parallel precompute with progress bar
# ===============================
def batch_precompute_embeddings_parallel(samples, batch_size=16, cache_path=None, n_workers=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    all_sentences = [s["sentence"] for s in samples]
    batches = [all_sentences[i:i+batch_size] for i in range(0, len(all_sentences), batch_size)]

    print(f"Computing embeddings using {n_workers} workers...")

    embeddings_list = []
    with Pool(n_workers) as pool:
        for batch_emb in tqdm(pool.imap(compute_batch_embeddings, batches), total=len(batches)):
            embeddings_list.extend(batch_emb)

    # Assign embeddings back to samples
    for s, emb in zip(samples, embeddings_list):
        s["embeddings"] = emb

    if cache_path:
        print(f"Saving embeddings cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)

    return samples

# ===============================
# 6️⃣ Run
# ===============================
all_samples = batch_precompute_embeddings_parallel(
    all_samples,
    batch_size=batch_size,
    cache_path=cache_path,
    n_workers=n_workers
)

print("✅ All embeddings precomputed and cached!")