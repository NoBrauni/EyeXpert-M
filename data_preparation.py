import os

import pandas as pd
import numpy as np
import pyreadr
from pathlib import Path
from rapidfuzz import process, fuzz
import pickle
from transformers import XLMRobertaModel, XLMRobertaTokenizer

class MECOPipeline:
    def __init__(
        self,
        w1_fix_file,
        w2_fix_file,
        sentences_file,
        duration_stats_path="duration_stats.npy",
        min_dur=60,
        output_dir="datasets",
        languages=None,  # List of languages to process; None = all
    ):
        self.w1_fix_file = w1_fix_file
        self.w2_fix_file = w2_fix_file
        self.sentences_file = sentences_file
        self.min_dur = min_dur
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.languages = languages

        # Load duration stats once
        self.dur_mean, self.dur_std = np.load(duration_stats_path)

        # Load sentence bank
        sentences_df = pd.read_csv(self.sentences_file)
        self.sentences_list = (
            sentences_df["sentence"].dropna().astype(str).tolist()
        )

        self.samples = []

    # -------------------------
    # Fuzzy matching
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
    # Load fixations
    # -------------------------
    def load_fixations(self):
        print("Loading fixation files...")
        w1_fix = pyreadr.read_r(self.w1_fix_file)["joint.fix"]
        w2_fix = pyreadr.read_r(self.w2_fix_file)["joint.fix"]
        df = pd.concat([w1_fix, w2_fix], ignore_index=True)
        if "lang" not in df.columns:
            df["lang"] = "unknown"
        df["unique_sentence_id"] = (
            df["subid"].astype(str)
            + "_"
            + df["trialid"].astype(str)
            + "_"
            + df["sentnum"].astype(str)
        )
        return df

    # -------------------------
    # Compute features
    # -------------------------
    def compute_features(self, df):
        df = df.sort_values(["unique_sentence_id", "start"]).reset_index(drop=True)
        grouped = df.groupby("unique_sentence_id")
        df["fix_index"] = grouped.cumcount() + 1
        if "ianum" in df.columns:
            df["ianum_local"] = grouped["ianum"].transform(lambda x: pd.factorize(x)[0] + 1)
        return df

    # -------------------------
    # Build dataset
    # -------------------------
    def build_dataset_full(self, df):
        df = df[(df["blink"] == 0) & (df["dur"] >= self.min_dur)].copy()
        word_table = df[["subid", "unique_sentence_id", "ianum", "word"]].drop_duplicates()
        word_table["full_idx"] = word_table.groupby(["subid", "unique_sentence_id"]).cumcount()
        df = df.merge(word_table, on=["subid", "unique_sentence_id", "ianum", "word"], how="left")
        df = df.dropna(subset=["full_idx"])
        df["dur_norm"] = (df["dur"] - self.dur_mean) / (self.dur_std + 1e-6)

        grouped = df.sort_values("fix_index").groupby(["subid", "unique_sentence_id"])
        scanpaths = grouped["full_idx"].agg(list)
        durations = grouped["dur_norm"].agg(list)
        words_full = word_table.sort_values("full_idx").groupby(["subid", "unique_sentence_id"])["word"].agg(list)
        meta = grouped.first()[["sentence", "lang"]]

        combined = pd.concat([meta, words_full, scanpaths, durations], axis=1).reset_index()
        combined.columns = ["subid", "sentence_id", "sentence", "lang", "words", "scanpath", "durations"]
        combined["sentence_len"] = combined["words"].apply(len)

        self.samples = combined.to_dict(orient="records")

    # -------------------------
    # Run pipeline
    # -------------------------
    def run(self):
        df = self.load_fixations()
        df = self.compute_features(df)

        # Filter languages if provided
        langs_to_process = df["lang"].unique()
        if self.languages:
            langs_to_process = [lang for lang in langs_to_process if lang in self.languages]

        for lang in langs_to_process:
            df_lang = df[df["lang"] == lang]
            out_file = self.output_dir / f"meco_dataset_{lang}.pkl"
            if out_file.exists():
                print(f"✅ Found existing dataset for {lang}, skipping fuzzy matching.")
                continue

            print(f"\nProcessing language: {lang} ({len(df_lang)} rows)")
            df_lang["sentence"] = df_lang["sent"].apply(self.fuzzy_match)
            match_rate = df_lang["sentence"].notna().mean()
            print(f"Fuzzy match success rate for {lang}: {match_rate:.2%}")

            self.build_dataset_full(df_lang)

            with open(out_file, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"Saved {len(self.samples)} samples to {out_file}")

        print("\nPipeline complete!")

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

pipeline = MECOPipeline(
    w1_fix_file="joint_l1_fixation_version2.0_w1.rda",
    w2_fix_file="joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA",
    sentences_file="sentences.csv",
    languages= [
    "en", "en_uk", "du", "ge", "ge_po", "ge_zu",
    "no", "da", "ic",
    "sp", "sp_ch", "it", "bp",
    "ru", "ru_mo", "se",
    "fi", "ee",
]
)
pipeline.run()