import pandas as pd

DATA_PATH = "data/processed/modeling_data.csv"

df = pd.read_csv(DATA_PATH)

print("Special CDC codes by variable:\n")

for column in df.columns:
    special_codes = df[column][df[column] < 0].value_counts().sort_index()

    if not special_codes.empty:
        print(f"{column}:")
        print(special_codes)
        print()