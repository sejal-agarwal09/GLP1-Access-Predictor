import pandas as pd

DATA_PATH = "data/processed/modeling_data.csv"

df = pd.read_csv(DATA_PATH)

predictors = [column for column in df.columns if column != "GLP_MED12M"]

print("Predictor distributions:\n")

for column in predictors:
    print("=" * 60)
    print(column)
    print("-" * 60)
    print(df[column].value_counts(dropna=False).sort_index())
    print()