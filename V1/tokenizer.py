import pandas as pd
import numpy as np
import pyreadr
from pathlib import Path
from rapidfuzz import process, fuzz
import pickle


class MECOPipeline:
    def __init__(
        self,
        w1_fix_file,
        w2_fix_file,
        sentences_file,
        duration_stats_path="duration_stats.npy",
        min_dur=60,
    ):
        self.w1_fix_file = w1_fix_file
        self.w2_fix_file = w2_fix_file
        self.sentences_file = sentences_file
        self.min_dur = min_dur

        # Load duration stats once
        self.dur_mean, self.dur_std = np.load(duration_stats_path)

        # Load sentence bank
        sentences_df = pd.read_csv(self.sentences_file)
        self.sentences_list = (
            sentences_df["sentence"].dropna().astype(str).tolist()
        )

        self.samples = []

    # -------------------------
    # Step 1: Fuzzy Matching
    # -------------------------
    def fuzzy_match(self, sent, score_cutoff=80):
        if pd.isna(sent) or not str(sent).strip():
            return None

        match = process.extractOne(
            str(sent).strip(),
            self.sentences_list,
            scorer=fuzz.partial_ratio,
        )

        if match and match[1] >= score_cutoff:
            return match[0]

        return None

    # -------------------------
    # Step 2: Load + Merge
    # -------------------------
    def load_fixations(self):
        print("Loading fixation files...")

        # Wave 1
        w1_fix = pyreadr.read_r(self.w1_fix_file)["joint.fix"]

        # Wave 2
        w2_fix = pyreadr.read_r(self.w2_fix_file)["joint.fix"]

        df = pd.concat([w1_fix, w2_fix], ignore_index=True)

        print(f"Loaded total rows: {len(df)}")

        if "lang" not in df.columns:
            df["lang"] = "unknown"

        print("Running fuzzy sentence matching...")
        df["sentence"] = df["sent"].apply(self.fuzzy_match)

        match_rate = df["sentence"].notna().mean()
        print(f"Fuzzy match success rate: {match_rate:.2%}")

        df["unique_sentence_id"] = (
            df["subid"].astype(str)
            + "_"
            + df["trialid"].astype(str)
            + "_"
            + df["sentnum"].astype(str)
        )

        return df

    # -------------------------
    # Step 3: Feature Engineering
    # -------------------------
    def compute_features(self, df):
        print("Computing features...")

        df = df.sort_values(["unique_sentence_id", "start"]).reset_index(drop=True)
        grouped = df.groupby("unique_sentence_id")

        df["fix_index"] = grouped.cumcount() + 1

        if "ianum" in df.columns:
            df["ianum_local"] = grouped["ianum"].transform(
                lambda x: pd.factorize(x)[0] + 1
            )

        print("Feature computation done.")
        return df

    # -------------------------
    # Step 4: Dataset Build
    # -------------------------
    def build_dataset_full(self, df):
        print("Building dataset (FULL sentence version)...")

        # -------------------------
        # Step 1: Filter
        # -------------------------
        df = df[(df["blink"] == 0) & (df["dur"] >= self.min_dur)].copy()
        print(f"Rows after filtering: {len(df)}")

        # -------------------------
        # Step 2: Build FULL word table (no dropping!)
        # -------------------------
        word_table = (
            df[["subid", "unique_sentence_id", "ianum", "word"]]
            .drop_duplicates()
            .sort_values(["subid", "unique_sentence_id", "ianum"])
        )

        # FULL sentence index (based on ianum order)
        word_table["full_idx"] = (
            word_table.groupby(["subid", "unique_sentence_id"])
            .cumcount()
        )

        # -------------------------
        # Step 3: Map fixations → full indices
        # -------------------------
        df = df.merge(
            word_table,
            on=["subid", "unique_sentence_id", "ianum", "word"],
            how="left",
        )

        missing = df["full_idx"].isna().sum()
        if missing > 0:
            print(f"⚠️ WARNING: {missing} fixations could not be mapped!")

        df = df.dropna(subset=["full_idx"])

        # -------------------------
        # Step 4: Normalize durations
        # -------------------------
        df["dur_norm"] = (df["dur"] - self.dur_mean) / (self.dur_std + 1e-6)

        # -------------------------
        # Step 5: Build scanpaths
        # -------------------------
        grouped = df.sort_values("fix_index").groupby(
            ["subid", "unique_sentence_id"]
        )

        scanpaths = grouped["full_idx"].agg(list)
        durations = grouped["dur_norm"].agg(list)

        # -------------------------
        # Step 6: FULL sentence words
        # -------------------------
        words_full = (
            word_table.sort_values("full_idx")
            .groupby(["subid", "unique_sentence_id"])["word"]
            .agg(list)
        )

        # -------------------------
        # Step 7: Metadata
        # -------------------------
        meta = grouped.first()[["sentence", "lang"]]

        # -------------------------
        # Step 8: Combine
        # -------------------------
        combined = pd.concat(
            [meta, words_full, scanpaths, durations],
            axis=1
        ).reset_index()

        combined.columns = [
            "subid",
            "sentence_id",
            "sentence",
            "lang",
            "words",
            "scanpath",
            "durations",
        ]

        combined["sentence_len"] = combined["words"].apply(len)

        self.samples = combined.to_dict(orient="records")

        print(f"Total samples built: {len(self.samples)}")

        # -------------------------
        # Debug sanity check
        # -------------------------
        bad = 0
        for s in self.samples[:100]:  # check first 100
            if max(s["scanpath"]) >= len(s["words"]):
                bad += 1

        if bad > 0:
            print(f"⚠️ Found {bad} misaligned samples!")
        else:
            print("✅ Alignment check passed!")

        print("\nExample sample:")
        print(self.samples[0] if self.samples else "No samples!")

        with open("meco_dataset.pkl", "wb") as f:
            pickle.dump(self.samples, f)

    # -------------------------
    # Run Everything
    # -------------------------
    def run(self):
        df = self.load_fixations()
        df = self.compute_features(df)

        print("\nSample rows after preprocessing:")
        print(df.head(3))

        self.build_dataset_full(df)

        print("\nExample sample:")
        print(self.samples[0] if self.samples else "No samples built.")

        print("\nPipeline complete!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]





pipeline = MECOPipeline(
    w1_fix_file="joint_l1_fixation_version2.0_w1.rda",
    w2_fix_file="joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA",
    sentences_file="sentences.csv",
)

pipeline.run()

print(len(pipeline))
sample = pipeline[0]