import pandas as pd

from predictor_selection import PREDICTOR_COLUMNS
from target_definition import TARGET_COLUMN, VALID_TARGET_VALUES


RAW_DATA_PATH = "data/raw/cdc_round7.csv"
OUTPUT_PATH = "data/processed/modeling_data.csv"


# Load the CDC dataset
df = pd.read_csv(RAW_DATA_PATH)

# Keep only our target and selected predictors
columns_to_keep = PREDICTOR_COLUMNS + [TARGET_COLUMN]
modeling_df = df[columns_to_keep].copy()

# Remove respondents with invalid target values
modeling_df = modeling_df[
    modeling_df[TARGET_COLUMN].isin(VALID_TARGET_VALUES)
].copy()


# CDC uses negative values for non-substantive responses.
# Convert these codes to missing values rather than treating them
# as real categories.
for column in PREDICTOR_COLUMNS:
    modeling_df.loc[modeling_df[column] < 0, column] = pd.NA


# Save the cleaned modeling dataset
modeling_df.to_csv(OUTPUT_PATH, index=False)


print("Cleaned modeling dataset created.")
print(f"Rows: {len(modeling_df):,}")
print(f"Columns: {len(modeling_df.columns)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nMissing values after cleaning:")
print(modeling_df.isna().sum())