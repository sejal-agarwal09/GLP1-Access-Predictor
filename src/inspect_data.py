import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/cdc_round7.csv")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))