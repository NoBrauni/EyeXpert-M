import pyreadr
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

class MECO_L1_Preprocessor:
    def __init__(self, w1_fix_file, w2_fix_file, sentences_file, output_dir="processed_data"):
        self.w1_fix_file = w1_fix_file
        self.w2_fix_file = w2_fix_file
        self.sentences_file = sentences_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sentences_df = pd.read_csv(self.sentences_file)
        self.sentences_list = sentences_df['sentence'].dropna().astype(str).tolist()

    def fuzzy_match(self, sent, score_cutoff=80):
        if pd.isna(sent) or not str(sent).strip():
            return None

        match = process.extractOne(
            str(sent).strip(),
            self.sentences_list,
            scorer=fuzz.partial_ratio
        )

        if match and match[1] >= score_cutoff:
            return match[0]

        return None

    def load_fixations(self):
        # Wave 1
        w1_fix = pd.read_csv(self.w1_fix_file) if self.w1_fix_file.endswith(".csv") else pyreadr.read_r(self.w1_fix_file)['joint.fix']

        # Wave 2
        w2_fix = pyreadr.read_r(self.w2_fix_file)['joint.fix']

        # Combine
        df = pd.concat([w1_fix, w2_fix], ignore_index=True)

        # Ensure lang exists
        if 'lang' not in df.columns:
            df['lang'] = 'unknown'

        # Add fuzzy matched sentence
        df['sentence'] = df['sent'].apply(self.fuzzy_match)

        # Unique sentence identifier (kept for downstream grouping)
        df['unique_sentence_id'] = (
            df['subid'].astype(str)
            + '_'
            + df['trialid'].astype(str)
            + '_'
            + df['sentnum'].astype(str)
        )
        return df

    def compute_features(self, df):
        df = df.sort_values(['unique_sentence_id', 'start']).reset_index(drop=True)
        grouped = df.groupby('unique_sentence_id')

        df['fix_index'] = grouped.cumcount() + 1

        if 'ianum' in df.columns:
            df['ianum_local'] = grouped['ianum'].transform(lambda x: pd.factorize(x)[0] + 1)
            df['ianum_norm'] = grouped['ianum_local'].transform(lambda x: x / x.max())

        df['sentence_fix_count'] = grouped['dur'].transform('count')
        df['sentence_rt'] = grouped['dur'].transform('sum')
        df['ia_dwell_pct'] = grouped['dur'].transform(lambda x: x / x.sum())

        return df

    def save_per_language(self, df):
        for lang, lang_df in df.groupby('lang'):
            out_path = self.output_dir / f"meco_l1_{lang}_processed.csv"
            lang_df.to_csv(out_path, index=False)
            print(f"Saved {lang_df.shape[0]} rows to {out_path}")

    def run(self):
        df = self.load_fixations()
        df = self.compute_features(df)
        self.save_per_language(df)
        print("Preprocessing complete!")

preprocessor = MECO_L1_Preprocessor(
    w1_fix_file="joint_l1_fixation_version2.0_w1.rda",
    w2_fix_file="joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA",
    sentences_file="sentences.csv"
)
preprocessor.run()