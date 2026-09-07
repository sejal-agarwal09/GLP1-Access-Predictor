import pandas as pd

DATA_PATH = "data/raw/cdc_round7.csv"

df = pd.read_csv(DATA_PATH)

print("GLP-1 target distribution:")
print(df["GLP_MED12M"].value_counts().sort_index())

print("\nPercent distribution:")
print(df["GLP_MED12M"].value_counts(normalize=True).sort_index() * 100)