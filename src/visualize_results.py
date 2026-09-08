import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


# ============================================================
# Paths
# ============================================================

MODEL_COMPARISON_PATH = "data/processed/validation_model_comparison.csv"
THRESHOLD_PATH = "data/processed/validation_threshold_analysis.csv"
FINAL_PREDICTIONS_PATH = "data/processed/final_test_predictions.csv"

LOGISTIC_IMPORTANCE_PATH = (
    "data/processed/logistic_predictor_importance.csv"
)

RF_IMPORTANCE_PATH = (
    "data/processed/random_forest_predictor_importance.csv"
)

OUTPUT_DIR = "data/processed"


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# Load validation model comparison
# ============================================================

comparison = pd.read_csv(
    MODEL_COMPARISON_PATH
)

print("Loaded validation model comparison.")


# ============================================================
# 1. Validation ROC-AUC comparison
# ============================================================

models = comparison["Model"]
roc_auc = comparison["ROC-AUC"]

plt.figure()

bars = plt.bar(
    models,
    roc_auc
)

plt.axhline(
    0.50,
    linestyle="--",
    linewidth=1.2,
    label="No-skill baseline"
)

plt.ylabel("ROC-AUC")
plt.title("Validation ROC-AUC by Model")

plt.ylim(
    0.45,
    max(0.90, roc_auc.max() + 0.05)
)

plt.xticks(
    rotation=15
)

for bar, value in zip(bars, roc_auc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.008,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.legend()

save_plot(
    "validation_model_roc_auc.png"
)


# ============================================================
# 2. Validation PR-AUC comparison
# ============================================================

pr_auc = comparison["PR-AUC"]

plt.figure()

bars = plt.bar(
    models,
    pr_auc
)

# GLP-1 prevalence in the validation set
validation_prevalence = 195 / 1618

plt.axhline(
    validation_prevalence,
    linestyle="--",
    linewidth=1.2,
    label=f"No-skill baseline ({validation_prevalence:.3f})"
)

plt.ylabel("PR-AUC")
plt.title(
    "Validation Precision-Recall AUC by Model"
)

plt.ylim(
    0,
    pr_auc.max() + 0.10
)

plt.xticks(
    rotation=15
)

for bar, value in zip(bars, pr_auc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.008,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.legend()

save_plot(
    "validation_model_pr_auc.png"
)


# ============================================================
# 3. Validation threshold analysis
# ============================================================

thresholds = pd.read_csv(
    THRESHOLD_PATH
)

plt.figure()

plt.plot(
    thresholds["Threshold"],
    thresholds["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    thresholds["Threshold"],
    thresholds["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    thresholds["Threshold"],
    thresholds["F1"],
    marker="o",
    linewidth=2,
    label="F1 score"
)

best_row = thresholds.loc[
    thresholds["F1"].idxmax()
]

plt.axvline(
    best_row["Threshold"],
    linestyle="--",
    linewidth=1.2,
    label=(
        f"Selected threshold = "
        f"{best_row['Threshold']:.2f}"
    )
)

plt.xlabel(
    "Classification threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Validation Precision, Recall, and F1 "
    "Across Classification Thresholds"
)

plt.ylim(
    0,
    1
)

plt.legend()

save_plot(
    "validation_threshold_tradeoff.png"
)


# ============================================================
# Load final test predictions
# ============================================================

final_predictions = pd.read_csv(
    FINAL_PREDICTIONS_PATH
)

print("\nLoaded final test predictions.")

print(
    "Columns:",
    final_predictions.columns.tolist()
)


# These are the exact column names produced by train_model.py
y_true = final_predictions["Actual"]

y_prob = final_predictions[
    "Predicted_Probability"
]


# ============================================================
# 4. Final-test ROC curve
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

baseline_model = comparison[
    comparison["Model"] == "Baseline Logistic"
]

if not baseline_model.empty:

    final_roc_auc = baseline_model[
        "ROC-AUC"
    ].iloc[0]

else:

    from sklearn.metrics import roc_auc_score

    final_roc_auc = roc_auc_score(
        y_true,
        y_prob
    )


plt.figure()

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=(
        f"Baseline Logistic "
        f"(AUC = {final_roc_auc:.3f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    label="No-skill baseline"
)

plt.xlabel(
    "False positive rate"
)

plt.ylabel(
    "True positive rate"
)

plt.title(
    "Final Test ROC Curve"
)

plt.legend(
    loc="lower right"
)

save_plot(
    "final_test_roc_curve.png"
)


# ============================================================
# 5. Final-test precision-recall curve
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_true,
    y_prob
)

final_prevalence = y_true.mean()

plt.figure()

plt.plot(
    recall,
    precision,
    linewidth=2,
    label="Baseline Logistic"
)

plt.axhline(
    final_prevalence,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"No-skill baseline "
        f"({final_prevalence:.3f})"
    )
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final Test Precision-Recall Curve"
)

plt.legend()

save_plot(
    "final_test_precision_recall_curve.png"
)


# ============================================================
# 6. Final-test calibration curve
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    strategy="quantile"
)

plt.figure()

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    linewidth=2,
    label="Baseline Logistic"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    label="Perfect calibration"
)

plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed frequency"
)

plt.title(
    "Final Test Calibration Curve"
)

plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1
)

plt.legend()

save_plot(
    "final_test_calibration_curve.png"
)


# ============================================================
# 7. Logistic regression predictor importance
# ============================================================

print(
    "\nLoading logistic predictor importance..."
)

logistic = pd.read_csv(
    LOGISTIC_IMPORTANCE_PATH
)

print(
    "Columns:",
    logistic.columns.tolist()
)


# The file generated by plot_feature_importance.py
# contains:
#
# Original_Predictor
# Absolute_Coefficient

logistic = logistic.rename(
    columns={
        "Original_Predictor": "Predictor",
        "Absolute_Coefficient": "Importance"
    }
)

logistic = logistic.sort_values(
    "Importance",
    ascending=True
).tail(12)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    logistic["Predictor"],
    logistic["Importance"]
)

plt.xlabel(
    "Aggregate absolute coefficient magnitude"
)

plt.ylabel(
    "Predictor"
)

plt.title(
    "Logistic Regression Predictor Importance"
)

save_plot(
    "logistic_predictor_importance_formal.png"
)


# ============================================================
# 8. Random Forest predictor importance
# ============================================================

print(
    "\nLoading Random Forest predictor importance..."
)

rf = pd.read_csv(
    RF_IMPORTANCE_PATH
)

print(
    "Columns:",
    rf.columns.tolist()
)


# The Random Forest aggregation file should use
# the same structure as the logistic file.
if "Original_Predictor" in rf.columns:

    rf = rf.rename(
        columns={
            "Original_Predictor": "Predictor"
        }
    )

if "Absolute_Importance" in rf.columns:

    rf = rf.rename(
        columns={
            "Absolute_Importance": "Importance"
        }
    )

elif "Importance" not in rf.columns:

    # Fall back to the raw Random Forest importance
    # column if needed.
    numeric_columns = rf.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        rf = rf.rename(
            columns={
                numeric_columns[-1]: "Importance"
            }
        )

    else:

        raise ValueError(
            "Could not find a numeric Random Forest "
            "importance column."
        )


rf = rf.sort_values(
    "Importance",
    ascending=True
).tail(12)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    rf["Predictor"],
    rf["Importance"]
)

plt.xlabel(
    "Aggregate feature importance"
)

plt.ylabel(
    "Predictor"
)

plt.title(
    "Random Forest Predictor Importance"
)

save_plot(
    "random_forest_predictor_importance_formal.png"
)


# ============================================================
# 9. Final-test target distribution
# ============================================================

target_counts = (
    y_true
    .value_counts()
    .sort_index()
)

labels = [
    "No GLP-1 use",
    "GLP-1 use"
]

values = [
    target_counts.get(0, 0),
    target_counts.get(1, 0)
]


plt.figure()

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Number of respondents"
)

plt.title(
    "GLP-1 Medication Use in the Final Test Set"
)

for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 20,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

save_plot(
    "final_test_target_distribution.png"
)


# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 60)

print(
    "VISUALIZATION COMPLETE"
)

print("=" * 60)

print(
    "\nAll evaluation figures were generated successfully."
)