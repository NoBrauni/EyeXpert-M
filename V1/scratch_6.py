import pandas as pd

df = pd.read_csv('processed_data/meco_l1_ge_processed.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.head())