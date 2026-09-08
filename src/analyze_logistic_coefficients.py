import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "models/final_logistic_model.joblib"
OUTPUT_PATH = "data/processed/logistic_odds_ratios.csv"


# Load the trained model
model = joblib.load(MODEL_PATH)

# Get the preprocessing pipeline and logistic regression model
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get logistic regression coefficients
coefficients = classifier.coef_[0]

# Calculate odds ratios
odds_ratios = np.exp(coefficients)

results = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Odds_Ratio": odds_ratios,
})

# Add direction
results["Direction"] = np.where(
    results["Coefficient"] > 0,
    "Higher odds",
    "Lower odds"
)

# Sort by absolute coefficient magnitude
results["Absolute_Coefficient"] = results["Coefficient"].abs()
results = results.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

# Save results
results.to_csv(OUTPUT_PATH, index=False)

print("Logistic regression odds ratios:")
print("-" * 80)
print(results[
    ["Feature", "Coefficient", "Odds_Ratio", "Direction"]
].head(25).to_string(index=False))

print("\nOdds-ratio file saved:")
print(OUTPUT_PATH)

print("\nTop predictors by absolute coefficient:")
print("-" * 80)

for _, row in results.head(10).iterrows():
    print(
        f"{row['Feature']}: "
        f"OR = {row['Odds_Ratio']:.2f} "
        f"({row['Direction']})"
    )