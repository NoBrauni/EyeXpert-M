import pyreadr
import pandas as pd
import numpy as np
import re
import json
from rapidfuzz import process, fuzz
from collections import Counter, defaultdict

# -----------------------------
# CONFIG
# -----------------------------

RDA_FILE_1 = "joint_l1_fixation_version2.0_w1.rda"
RDA_FILE_2 = "joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA"
SENTENCE_CSV = "sentences.csv"

TEST_RATIO = 0.2
RANDOM_SEED = 42

LANGUAGES = [
    "en", "en_uk", "du", "ge", "ge_po", "ge_zu",
    "no", "da", "ic",
    "sp", "sp_ch", "it", "bp",
    "ru", "ru_mo", "se",
    "fi", "ee",
]

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------

def norm_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).lower().strip())

def norm_word(w):
    return re.sub(r"[^\wåæøÅÆØ]", "", str(w).lower())

# -----------------------------
# LOAD
# -----------------------------

def load_rda(path):
    return next(iter(pyreadr.read_r(path).values()))

print("Loading data...")

df = pd.concat([
    load_rda(RDA_FILE_1),
    load_rda(RDA_FILE_2)
], ignore_index=True)

df = df[df["lang"].isin(LANGUAGES)].copy()

# sentence bank
sentences = pd.read_csv(SENTENCE_CSV)["sentence"].dropna()
sentence_list = sentences.map(norm_text).tolist()

# -----------------------------
# MATCH SENTENCES (FAST CACHE)
# -----------------------------

print("Matching sentences (fast)...")

df["sent_norm"] = df["sent"].map(norm_text)
unique_prefixes = df["sent_norm"].dropna().unique()

mapping = {}

for i, p in enumerate(unique_prefixes):
    if i % 1000 == 0:
        print(f"{i}/{len(unique_prefixes)}")

    if not p:
        continue

    match = process.extractOne(p, sentence_list, scorer=fuzz.partial_ratio)
    mapping[p] = match[0] if match and match[1] >= 85 else None

df["full_sentence"] = df["sent_norm"].map(mapping)

print("Missing:", df["full_sentence"].isna().mean())

df = df.dropna(subset=["full_sentence"])

# -----------------------------
# UID
# -----------------------------

df["uid"] = (
    df["subid"].astype(str) + "_" +
    df["trialid"].astype(str) + "_" +
    df["sentnum"].astype(str)
)

# -----------------------------
# GROUP META TABLE (IMPORTANT)
# -----------------------------

meta = df.groupby("uid").first().reset_index()[["uid", "subid", "lang", "full_sentence"]]

# -----------------------------
# SPLIT (BALANCED TEST)
# -----------------------------

print("Creating balanced split...")

np.random.seed(RANDOM_SEED)

test_uids = []

for lang, g in meta.groupby("lang"):
    uids = g["uid"].tolist()
    np.random.shuffle(uids)

    n_test = int(len(uids) * TEST_RATIO)
    test_uids.extend(uids[:n_test])

test_uids = set(test_uids)

train_uids = set(meta["uid"]) - test_uids

# sanity: no overlap
assert len(train_uids & test_uids) == 0

print("Train UIDs:", len(train_uids))
print("Test UIDs:", len(test_uids))

# -----------------------------
# ALIGNMENT
# -----------------------------

def build_alignment(g):

    full_sent = g["full_sentence"].iloc[0]
    tokens_raw = full_sent.split()
    tokens = [norm_word(w) for w in tokens_raw]

    fix_words_raw = g["word"].astype(str).tolist()
    fix_words = [norm_word(w) for w in fix_words_raw]

    ia = pd.to_numeric(g["ianum"], errors="coerce").tolist()
    dur = pd.to_numeric(g["dur"], errors="coerce").fillna(0).tolist()

    # -----------------------------
    # Anchor using unique words
    # -----------------------------
    counts = Counter(tokens)
    unique_words = {w for w, c in counts.items() if c == 1}

    anchor_fix = None
    anchor_sent = None

    for i, w in enumerate(fix_words):
        if w in unique_words and not np.isnan(ia[i]):
            anchor_fix = i
            anchor_sent = tokens.index(w)
            break

    if anchor_fix is None:
        return None

    offset = anchor_sent - int(ia[anchor_fix])

    # -----------------------------
    # Build aligned scanpath
    # -----------------------------
    scanpath = []
    scanpath_words = []
    durations = []

    for i, x in enumerate(ia):

        if np.isnan(x):
            continue

        shifted = int(x) + offset

        if 0 <= shifted < len(tokens):

            token_word = tokens[shifted]
            fix_word = fix_words[i]

            # allow mismatch but track it
            if fix_word != "" and fix_word != token_word:
                # optional: skip noisy mismatch
                # continue

                pass

            scanpath.append(shifted)
            scanpath_words.append(tokens_raw[shifted])  # keep original casing
            durations.append(dur[i])

    if len(scanpath) == 0:
        return None

    return {
        "uid": g["uid"].iloc[0],
        "lang": g["lang"].iloc[0],
        "sentence": full_sent,
        "words": tokens_raw,

        "scanpath": scanpath,
        "scanpath_words": scanpath_words,
        "durations": durations,

        "n_words": len(tokens),
        "n_fixations": len(scanpath),
    }

# -----------------------------
# BUILD SAMPLES
# -----------------------------

print("Building dataset...")

train, test = [], []

for uid, g in df.groupby("uid"):

    sample = build_alignment(g)
    if sample is None:
        continue

    if uid in test_uids:
        test.append(sample)
    else:
        train.append(sample)

print("Train samples:", len(train))
print("Test samples:", len(test))

# -----------------------------
# LANGUAGE BALANCE CHECK
# -----------------------------

def lang_dist(data):
    d = defaultdict(int)
    for s in data:
        d[s["lang"]] += 1
    return dict(d)

print("\nLANGUAGE DISTRIBUTION")
print("Train:", lang_dist(train))
print("Test:", lang_dist(test))

# -----------------------------
# SAVE JSON
# -----------------------------

print("Saving JSON...")

with open("meco_train.json", "w") as f:
    json.dump(train, f)

with open("meco_test.json", "w") as f:
    json.dump(test, f)

print("Saved:")
print(" - meco_train.json")
print(" - meco_test.json")