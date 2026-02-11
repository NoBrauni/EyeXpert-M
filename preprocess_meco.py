import pyreadr
import pandas as pd
from pathlib import Path

class MECO_L1_Preprocessor:
    def __init__(self, w1_fix_file, w1_sent_file, w2_fix_file, w2_sent_file, output_dir="processed_data"):
        self.w1_fix_file = w1_fix_file
        self.w1_sent_file = w1_sent_file
        self.w2_fix_file = w2_fix_file
        self.w2_sent_file = w2_sent_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_trial_by_trial(self, fix_df, sent_df, wave_name):
        merged_trials = []
        for trial_id in fix_df['trialid'].unique():
            fix_trial = fix_df[fix_df['trialid'] == trial_id].copy()
            sent_trial = sent_df[sent_df['trialid'] == trial_id]
            if sent_trial.empty:
                continue
            for col in sent_trial.columns:
                if col not in fix_trial.columns:
                    fix_trial[col] = sent_trial.iloc[0][col]
            # Only keep fixations on words
            if 'wordnum' in fix_trial.columns:
                fix_trial = fix_trial[fix_trial['wordnum'] > 0]
            merged_trials.append(fix_trial)
        merged_df = pd.concat(merged_trials, ignore_index=True)
        return merged_df

    def load_fixations(self):
        # Wave 1
        w1_fix = pd.read_csv(self.w1_fix_file) if self.w1_fix_file.endswith(".csv") else pyreadr.read_r(self.w1_fix_file)['joint.fix']
        w1_sent = pd.read_csv(self.w1_sent_file)
        w1 = self.merge_trial_by_trial(w1_fix, w1_sent, "W1")

        # Wave 2
        w2_fix = pyreadr.read_r(self.w2_fix_file)['joint.fix']
        w2_sent_rda = pyreadr.read_r(self.w2_sent_file)
        rda_key = list(w2_sent_rda.keys())[0]
        w2_sent = w2_sent_rda[rda_key]
        w2 = self.merge_trial_by_trial(w2_fix, w2_sent, "W2")

        # Keep only columns common to both waves
        common_cols = w1.columns.intersection(w2.columns)
        w1 = w1[common_cols]
        w2 = w2[common_cols]
        df = pd.concat([w1, w2], ignore_index=True)
        df['unique_trial_id'] = df['subid'].astype(str) + '_' + df['trialid'].astype(str)
        return df

    def compute_features(self, df):
        # Sort by time
        df = df.sort_values(['unique_trial_id', 'start']).reset_index(drop=True)
        grouped = df.groupby('unique_trial_id')

        # Fixation index
        df['fix_index'] = grouped.cumcount() + 1

        # Local IA index and normalized IA
        if 'ianum' in df.columns:
            df['ianum_local'] = grouped['ianum'].transform(lambda x: pd.factorize(x)[0] + 1)
            df['ianum_norm'] = grouped['ianum_local'].transform(lambda x: x / x.max())

        # Trial-level
        df['trial_ia_count'] = grouped['unique_trial_id'].transform('count')
        df['paragraph_rt'] = grouped['dur'].transform('sum')
        df['ia_dwell_pct'] = grouped['dur'].transform(lambda x: x / x.sum())

        return df

    def save_per_language(self, df):
        if 'lang' not in df.columns:
            df['lang'] = 'unknown'
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
    w1_sent_file="sentence_data_version2.0_w1.csv",
    w2_fix_file="joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA",
    w2_sent_file="joint_sent_trimmed_wave2.rda"
)
preprocessor.run()
