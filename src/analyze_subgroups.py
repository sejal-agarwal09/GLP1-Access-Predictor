import pandas as pd

DATA_PATH = "data/processed/final_test_predictions.csv"

df = pd.read_csv(DATA_PATH)

# Identify prediction errors
df["Error_Type"] = "Correct"

df.loc[
    (df["Actual"] == 0) & (df["Predicted_Class"] == 1),
    "Error_Type"
] = "False Positive"

df.loc[
    (df["Actual"] == 1) & (df["Predicted_Class"] == 0),
    "Error_Type"
] = "False Negative"

df.loc[
    (df["Actual"] == 0) & (df["Predicted_Class"] == 0),
    "Error_Type"
] = "True Negative"

df.loc[
    (df["Actual"] == 1) & (df["Predicted_Class"] == 1),
    "Error_Type"
] = "True Positive"


print("FINAL TEST ERROR ANALYSIS")
print("=" * 80)

print("\nPrediction outcomes:")
print(df["Error_Type"].value_counts())

print("\nPrediction outcome percentages:")
print(
    (df["Error_Type"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str)
    + "%"
)

print("\nAverage predicted probability by outcome:")
print(
    df.groupby("Error_Type")["Predicted_Probability"]
    .mean()
    .round(3)
)

print("\nFalse positive cases:")
false_positives = df[df["Error_Type"] == "False Positive"]

print(f"Count: {len(false_positives):,}")
print(
    f"Mean predicted probability: "
    f"{false_positives['Predicted_Probability'].mean():.3f}"
)

print("\nFalse negative cases:")
false_negatives = df[df["Error_Type"] == "False Negative"]

print(f"Count: {len(false_negatives):,}")
print(
    f"Mean predicted probability: "
    f"{false_negatives['Predicted_Probability'].mean():.3f}"
)

# Save error analysis
df.to_csv(
    "data/processed/final_test_error_analysis.csv",
    index=False,
)

print("\nSaved:")
print("data/processed/final_test_error_analysis.csv")