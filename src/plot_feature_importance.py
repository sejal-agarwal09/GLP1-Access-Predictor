import pandas as pd
import matplotlib.pyplot as plt


LOGISTIC_PATH = "data/processed/logistic_feature_importance.csv"
RF_PATH = "data/processed/random_forest_feature_importance.csv"


# ============================================================
# LOAD FEATURE IMPORTANCE DATA
# ============================================================

logistic_df = pd.read_csv(LOGISTIC_PATH)
rf_df = pd.read_csv(RF_PATH)


# ============================================================
# MAP ONE-HOT FEATURES TO ORIGINAL PREDICTORS
# ============================================================

def get_original_predictor(feature_name):
    """
    Convert a one-hot encoded feature name back to
    its original predictor variable.
    """

    feature_name = feature_name.replace(
        "categorical__",
        ""
    )

    predictor_names = [
        "CHR_OBEV",
        "DIB_DIBEV",
        "INSURED",
        "DOCVIS_P12M",
        "ACC_HTHUSUAL",
        "P_AGE5YRS_R",
        "P_SEX",
        "DEM_RACEETH",
        "DEM_REGION",
        "P_EDUCATION_I_R",
        "P_POVERTY4_R",
        "BMICAT6",
    ]

    for predictor in predictor_names:
        if feature_name.startswith(predictor + "_"):
            return predictor

    return feature_name


# Apply mapping
logistic_df["Original_Predictor"] = (
    logistic_df["Feature"]
    .apply(get_original_predictor)
)

rf_df["Original_Predictor"] = (
    rf_df["Feature"]
    .apply(get_original_predictor)
)


# ============================================================
# AGGREGATE LOGISTIC REGRESSION IMPORTANCE
# ============================================================

# Sum absolute coefficients across categories
logistic_grouped = (
    logistic_df
    .groupby("Original_Predictor")["Absolute_Coefficient"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# ============================================================
# AGGREGATE RANDOM FOREST IMPORTANCE
# ============================================================

# Sum importance across one-hot encoded categories
rf_grouped = (
    rf_df
    .groupby("Original_Predictor")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# ============================================================
# SAVE AGGREGATED RESULTS
# ============================================================

logistic_grouped.to_csv(
    "data/processed/logistic_predictor_importance.csv",
    index=False
)

rf_grouped.to_csv(
    "data/processed/random_forest_predictor_importance.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nAggregated Logistic Regression importance:")
print("-" * 60)
print(logistic_grouped.to_string(index=False))


print("\nAggregated Random Forest importance:")
print("-" * 60)
print(rf_grouped.to_string(index=False))


# ============================================================
# CLEAN DISPLAY NAMES
# ============================================================

display_names = {
    "CHR_OBEV": "Obesity history",
    "DIB_DIBEV": "Diabetes history",
    "INSURED": "Health insurance",
    "DOCVIS_P12M": "Doctor visit in past 12 months",
    "ACC_HTHUSUAL": "Usual healthcare location",
    "P_AGE5YRS_R": "Age",
    "P_SEX": "Sex",
    "DEM_RACEETH": "Race/ethnicity",
    "DEM_REGION": "Census region",
    "P_EDUCATION_I_R": "Education",
    "P_POVERTY4_R": "Poverty status",
    "BMICAT6": "BMI category",
}


logistic_grouped["Display_Name"] = (
    logistic_grouped["Original_Predictor"]
    .map(display_names)
)


rf_grouped["Display_Name"] = (
    rf_grouped["Original_Predictor"]
    .map(display_names)
)


# ============================================================
# LOGISTIC REGRESSION PLOT
# ============================================================

logistic_plot = logistic_grouped.head(10).sort_values(
    "Absolute_Coefficient"
)

plt.figure(figsize=(10, 6))

plt.barh(
    logistic_plot["Display_Name"],
    logistic_plot["Absolute_Coefficient"]
)

plt.xlabel("Aggregated Absolute Coefficient")
plt.ylabel("Predictor")
plt.title(
    "Predictor Importance — Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    "data/processed/logistic_predictor_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# RANDOM FOREST PLOT
# ============================================================

rf_plot = rf_grouped.head(10).sort_values(
    "Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    rf_plot["Display_Name"],
    rf_plot["Importance"]
)

plt.xlabel("Aggregated Feature Importance")
plt.ylabel("Predictor")
plt.title(
    "Predictor Importance — Random Forest"
)

plt.tight_layout()

plt.savefig(
    "data/processed/random_forest_predictor_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nAggregated feature importance files saved.")
print(
    "data/processed/logistic_predictor_importance.csv"
)
print(
    "data/processed/random_forest_predictor_importance.csv"
)