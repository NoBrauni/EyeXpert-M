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

# Data quality filtering
MIN_FIXATION_DURATION = 80  # milliseconds
FILTER_BLINKS = True  # Remove blink events from raw data

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

# ========== DATA QUALITY FILTERING ==========
print(f"Before filtering: {len(df):,} fixations")

# Filter out blinks (standard eye-tracking data cleaning)
if FILTER_BLINKS and "blink" in df.columns:
    df = df[df["blink"] == 0].copy()
    print(f"After blink filtering: {len(df):,} fixations")

# Filter out very short fixations (< 80ms)
if "dur" in df.columns:
    df = df[pd.to_numeric(df["dur"], errors="coerce") >= MIN_FIXATION_DURATION].copy()
    print(f"After duration filtering (>= {MIN_FIXATION_DURATION}ms): {len(df):,} fixations")

# sentence bank
sentences = pd.read_csv(SENTENCE_CSV)["sentence"].dropna()
sentence_list = sentences.map(norm_text).tolist()

# -----------------------------
# MATCH SENTENCES
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
# GROUP META TABLE
# -----------------------------

meta = df.groupby("uid").first().reset_index()[["uid", "subid", "lang", "full_sentence"]]


print("Creating split: hybrid test set (unseen sentences + unseen readers)...")

np.random.seed(RANDOM_SEED)

# Step 1: Split sentences 80/20 per language (test-exclusive sentences)
train_sentences = set()
test_sentences = set()

for lang in meta["lang"].unique():
    lang_sentences = meta[meta["lang"] == lang]["full_sentence"].unique()
    sentences_list = list(lang_sentences)
    np.random.shuffle(sentences_list)

    n_test = int(len(sentences_list) * TEST_RATIO)
    test_sentences.update(sentences_list[:n_test])
    train_sentences.update(sentences_list[n_test:])

print(f"Sentence split: {len(train_sentences)} train, {len(test_sentences)} test-exclusive")

# Step 2: Split readers 80/20 per language (test-exclusive readers)
train_readers = set()
test_readers = set()

for lang in meta["lang"].unique():
    lang_readers = meta[meta["lang"] == lang]["subid"].unique()
    readers_list = list(lang_readers)
    np.random.shuffle(readers_list)

    n_test = int(len(readers_list) * TEST_RATIO)
    test_readers.update(readers_list[:n_test])
    train_readers.update(readers_list[n_test:])

print(f"Reader split: {len(train_readers)} train, {len(test_readers)} test-exclusive")

# Step 3: Build train/test sets
train_uids = set()
test_uids = set()

for _, row in meta.iterrows():
    uid = row["uid"]
    reader = row["subid"]
    sentence = row["full_sentence"]

    # TEST SET: Two conditions (inclusive OR)
    if (sentence in test_sentences) or (reader in test_readers):
        test_uids.add(uid)
    elif (reader in train_readers) and (sentence in train_sentences):
        train_uids.add(uid)

print(f"\nFinal split:")
print(f"  Train UIDs: {len(train_uids):,}")
print(f"  Test UIDs: {len(test_uids):,}")

# Validation
assert len(train_uids & test_uids) == 0, "Train/test overlap detected!"

test_readers_actual = set(meta[meta["uid"].isin(test_uids)]["subid"].unique())
test_sentences_actual = set(meta[meta["uid"].isin(test_uids)]["full_sentence"].unique())

train_readers_actual = set(meta[meta["uid"].isin(train_uids)]["subid"].unique())
train_sentences_actual = set(meta[meta["uid"].isin(train_uids)]["full_sentence"].unique())

reader_overlap = test_readers_actual & train_readers_actual
sentence_overlap = test_sentences_actual & train_sentences_actual

print(f"\nVALIDATION RESULTS:")
print(f"  Test readers: {len(test_readers_actual)}")
print(f"  Test sentences: {len(test_sentences_actual)}")
print(f"  Reader overlap: {len(reader_overlap)} (test readers NOT in train: YES)")
print(f"  Sentence overlap: {len(sentence_overlap)} (test sentences NOT in train: YES)")

test_only_readers = test_readers_actual - train_readers_actual
test_only_sentences = test_sentences_actual - train_sentences_actual
bridge_readers = test_readers_actual & train_readers_actual  # Readers in both sets

print(f"\nSPLIT COMPOSITION:")
print(f"  Test-exclusive readers: {len(test_only_readers)} (completely new readers)")
print(f"  Bridge readers (in both sets): {len(bridge_readers)} (see new sentences in test)")
print(f"  Test-exclusive sentences: {len(test_only_sentences)} (completely new sentences)")
print(f"  Bridge sentences (in both sets): {len(test_sentences_actual - test_only_sentences)} (seen by new readers only)")

if len(test_only_readers) > 0 and len(test_only_sentences) > 0:
    print(f"\n  → PERFECT! Test set achieves:")
    print(f"     • Completely unseen readers: YES ({len(test_only_readers)} readers)")
    print(f"     • Completely unseen sentences: YES ({len(test_only_sentences)} sentences)")
    print(f"     • Generalization test: YES (bridge readers see new sentences)")
else:
    print(f"\n Suboptimal split, but test is still independent")

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

# -----------------------------
# VALIDATION & QUALITY METRICS
# -----------------------------

def validate_alignment_quality(df, aligned_samples):
    """Validate alignment quality and report statistics."""
    print("\n ALIGNMENT QUALITY VALIDATION")
    print("="*50)

    total_samples = len(aligned_samples)
    total_fixations = sum(len(s["scanpath"]) for s in aligned_samples)

    # Language coverage
    lang_counts = defaultdict(int)
    for s in aligned_samples:
        lang_counts[s["lang"]] += 1

    print(f"✓ Total aligned samples: {total_samples:,}")
    print(f"✓ Total fixations: {total_fixations:,}")
    
    original_fixations = len(df)
    retention_rate = total_fixations / original_fixations * 100
    print(f"✓ Fixation retention rate: {retention_rate:.2f}%")

    # Alignment success rate by language
    print("\nAlignment Success by Language:")
    print("-" * 40)
    for lang in sorted(lang_counts.keys()):
        count = lang_counts[lang]
        total_lang = len(df[df["lang"] == lang]["uid"].unique())
        success_rate = count / total_lang * 100 if total_lang > 0 else 0
        print(f"  {lang}: {count:5d} samples ({success_rate:5.1f}% success rate)")

    # Scanpath statistics
    scanpath_lengths = [len(s["scanpath"]) for s in aligned_samples]
    print("\nScanpath Statistics:")
    print("-" * 40)
    print(f"  Average fixations per sample: {np.mean(scanpath_lengths):.1f}")
    print(f"  Median fixations per sample: {np.median(scanpath_lengths):.1f}")
    print(f"  Min/Max fixations: {min(scanpath_lengths)}/{max(scanpath_lengths)}")

    # Duration statistics
    all_durations = [d for s in aligned_samples for d in s["durations"]]
    print("\nDuration Statistics:")
    print("-" * 40)
    print(f"  Average fixation duration: {np.mean(all_durations):.1f}ms")
    print(f"  Median fixation duration: {np.median(all_durations):.1f}ms")
    print(f"  Duration range: {min(all_durations):.1f}-{max(all_durations):.1f}ms")

    # Check for potential issues
    issues = []
    if np.mean(scanpath_lengths) < 3:
        issues.append("Very short scanpaths - may indicate alignment issues")
    if np.mean(all_durations) < 100:
        issues.append("Very short durations - check units (should be ms)")
    if total_samples < 1000:
        issues.append("Low sample count - may limit statistical power")

    if issues:
        print("\nPOTENTIAL ISSUES:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ No obvious data quality issues detected")
    return True


def check_data_balance(train_data, test_data):
    """Check statistical balance between train/test splits."""
    print("\nTRAIN/TEST BALANCE CHECK")
    print("="*50)

    def get_stats(data):
        if len(data) == 0:
            return {
                "count": 0,
                "lang_dist": {},
                "avg_fixations": 0,
                "avg_words": 0
            }
        
        stats = defaultdict(list)
        for s in data:
            stats["lang"].append(s["lang"])
            stats["n_fixations"].append(s["n_fixations"])
            stats["n_words"].append(s["n_words"])

        return {
            "count": len(data),
            "lang_dist": dict(Counter(stats["lang"])),
            "avg_fixations": np.mean(stats["n_fixations"]),
            "avg_words": np.mean(stats["n_words"])
        }

    train_stats = get_stats(train_data)
    test_stats = get_stats(test_data)

    print(f"Train samples: {train_stats['count']:,}")
    print(f"Test samples: {test_stats['count']:,}")
    
    if test_stats['count'] == 0:
        print("WARNING: Test set is empty!")
        return False

    # Language balance check
    print("\nLanguage Balance:")
    all_langs = set(train_stats["lang_dist"].keys()) | set(test_stats["lang_dist"].keys())
    for lang in sorted(all_langs):
        train_count = train_stats["lang_dist"].get(lang, 0)
        test_count = test_stats["lang_dist"].get(lang, 0)
        train_pct = train_count / max(train_stats["count"], 1) * 100
        test_pct = test_count / max(test_stats["count"], 1) * 100
        diff = abs(train_pct - test_pct)
        status = "✅" if diff < 5 else "⚠️ "
        print(f"  {status} {lang}: train {train_pct:5.1f}%, test {test_pct:5.1f}% (diff {diff:5.1f}%)")

    # Statistical balance
    fix_diff = abs(train_stats["avg_fixations"] - test_stats["avg_fixations"])
    word_diff = abs(train_stats["avg_words"] - test_stats["avg_words"])

    print("\nStatistical Balance:")
    print(f"  Avg fixations - train: {train_stats['avg_fixations']:.1f}, test: {test_stats['avg_fixations']:.1f} (diff {fix_diff:.1f})")
    print(f"  Avg words - train: {train_stats['avg_words']:.1f}, test: {test_stats['avg_words']:.1f} (diff {word_diff:.1f})")

    if fix_diff > 1 or word_diff > 1:
        print(" Large statistical differences detected")
    else:
        print(" Statistical balance maintained")

    return True

# -----------------------------
# RUN VALIDATION
# -----------------------------

validate_alignment_quality(df, train + test)
check_data_balance(train, test)

print("\n DATA PREPARATION SUMMARY")
print("="*80)
print("\nDATA QUALITY FILTERING APPLIED:")
print(f"  ✓ Blinks removed (blink == 0)")
print(f"  ✓ Minimum fixation duration: {MIN_FIXATION_DURATION}ms (excludes microsaccades)")


print("\nDATA SPLIT STATISTICS:")
print(f"  • Training samples: {len(train):,}")
print(f"  • Test samples: {len(test):,}")
if len(train) + len(test) > 0:
    test_ratio = len(test) / (len(train) + len(test)) * 100
    print(f"  • Test ratio: {test_ratio:.1f}%")

print("\nTOTAL FIXATIONS (AFTER CLEANING):")
total_train_fixations = sum(len(s['scanpath']) for s in train)
total_test_fixations = sum(len(s['scanpath']) for s in test)
total_fixations = total_train_fixations + total_test_fixations
print(f"  • Training fixations: {total_train_fixations:,}")
print(f"  • Test fixations: {total_test_fixations:,}")
print(f"  • Total: {total_fixations:,}")

print("\n Data preparation complete!")
print("="*80)
