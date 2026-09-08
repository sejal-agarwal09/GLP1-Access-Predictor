import pandas as pd


RAW_DATA_PATH = "data/raw/cdc_round7.csv"
OUTPUT_PATH = "data/processed/weighted_glp1_prevalence.csv"

TARGET = "GLP_MED12M"
WEIGHT = "WEIGHT"


# Load the original CDC Round 7 data
df = pd.read_csv(RAW_DATA_PATH)

# Keep valid target responses
df = df[df[TARGET].isin([0, 1])].copy()

# Weighted counts
weighted_no = df.loc[df[TARGET] == 0, WEIGHT].sum()
weighted_yes = df.loc[df[TARGET] == 1, WEIGHT].sum()

weighted_total = weighted_no + weighted_yes

weighted_prevalence = weighted_yes / weighted_total

# Unweighted prevalence for comparison
unweighted_prevalence = df[TARGET].mean()

print("CDC ROUND 7 WEIGHTED GLP-1 PREVALENCE")
print("=" * 80)

print(f"Valid respondents: {len(df):,}")

print("\nUnweighted prevalence:")
print(f"GLP-1 use: {unweighted_prevalence:.4%}")

print("\nWeighted prevalence:")
print(f"GLP-1 use: {weighted_prevalence:.4%}")

print("\nWeighted totals:")
print(f"No GLP-1 use: {weighted_no:,.2f}")
print(f"GLP-1 use:    {weighted_yes:,.2f}")
print(f"Total:        {weighted_total:,.2f}")


# Save results
results = pd.DataFrame(
    {
        "Measure": [
            "Unweighted GLP-1 prevalence",
            "Weighted GLP-1 prevalence",
        ],
        "Estimate": [
            unweighted_prevalence,
            weighted_prevalence,
        ],
    }
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nSaved:")
print(OUTPUT_PATH)