import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from predictor_selection import PREDICTOR_COLUMNS
from target_definition import TARGET_COLUMN


DATA_PATH = "data/processed/modeling_data.csv"

VALIDATION_RESULTS_PATH = "data/processed/validation_model_comparison.csv"
THRESHOLD_RESULTS_PATH = "data/processed/validation_threshold_analysis.csv"
FINAL_TEST_PATH = "data/processed/final_test_predictions.csv"
FINAL_SUMMARY_PATH = "data/processed/final_model_summary.csv"
VALIDATION_PREDICTIONS_PATH = "data/processed/validation_model_predictions.csv"
MODEL_PREDICTIONS_PATH = "data/processed/model_predictions.csv"
MODEL_OUTPUT_PATH = "models/final_logistic_model.joblib"


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[PREDICTOR_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

print("Dataset loaded.")
print(f"Rows: {len(df):,}")
print(f"Predictors: {len(PREDICTOR_COLUMNS)}")
print(f"Target: {TARGET_COLUMN}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# -------------------------------------------------------------------
# Train / validation / final test split
#
# 60% training
# 20% validation
# 20% final test
# -------------------------------------------------------------------

X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=0.25,
    stratify=y_development,
    random_state=42,
)

print("\nTrain/validation/final test split:")
print(f"Training rows:      {len(X_train):,}")
print(f"Validation rows:    {len(X_validation):,}")
print(f"Final test rows:    {len(X_test):,}")

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation target distribution:")
print(y_validation.value_counts().sort_index())

print("\nFinal test target distribution:")
print(y_test.value_counts().sort_index())


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="first",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            PREDICTOR_COLUMNS,
        ),
    ]
)


# -------------------------------------------------------------------
# Candidate models
# -------------------------------------------------------------------

models = {
    "Baseline Logistic": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Balanced Logistic": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    ),
}


# -------------------------------------------------------------------
# Train candidate models
# -------------------------------------------------------------------

trained_models = {}
validation_results = []
validation_prediction_rows = []

print("\nTraining candidate models...")

for model_name, model in models.items():

    print(f"Training {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[model_name] = pipeline

    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = (
        validation_probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    validation_results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }
    )

    validation_prediction_rows.append(
        pd.DataFrame(
            {
                "Model": model_name,
                "Actual": y_validation.values,
                "Predicted_Probability": validation_probabilities,
                "Predicted_Class": validation_predictions,
            }
        )
    )

    print(f"{model_name} training complete.")


# -------------------------------------------------------------------
# Compare models on validation set
# -------------------------------------------------------------------

validation_results_df = pd.DataFrame(validation_results)

validation_results_df.to_csv(
    VALIDATION_RESULTS_PATH,
    index=False,
)

print("\nValidation-set model comparison:")
print("-" * 90)

for _, row in validation_results_df.iterrows():
    print(
        f"{row['Model']:<22}"
        f" ROC-AUC: {row['ROC-AUC']:.3f}"
        f" | PR-AUC: {row['PR-AUC']:.3f}"
        f" | F1: {row['F1']:.3f}"
    )


# -------------------------------------------------------------------
# Select model using validation PR-AUC
# -------------------------------------------------------------------

best_model_name = validation_results_df.loc[
    validation_results_df["PR-AUC"].idxmax(),
    "Model",
]

best_validation_pr_auc = validation_results_df.loc[
    validation_results_df["Model"] == best_model_name,
    "PR-AUC",
].iloc[0]

best_model = trained_models[best_model_name]

print("\nSelected model:")
print(best_model_name)
print(f"Validation PR-AUC: {best_validation_pr_auc:.3f}")


# -------------------------------------------------------------------
# Threshold analysis on validation set
# -------------------------------------------------------------------

validation_probabilities = best_model.predict_proba(
    X_validation
)[:, 1]

threshold_results = []

for threshold in np.arange(0.10, 0.51, 0.05):

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    threshold_results.append(
        {
            "Threshold": round(threshold, 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )

threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df.to_csv(
    THRESHOLD_RESULTS_PATH,
    index=False,
)

print("\nValidation threshold analysis:")
print("-" * 80)

for _, row in threshold_results_df.iterrows():
    print(
        f"Threshold {row['Threshold']:.2f}"
        f" | Precision: {row['Precision']:.3f}"
        f" | Recall: {row['Recall']:.3f}"
        f" | F1: {row['F1']:.3f}"
    )


# -------------------------------------------------------------------
# Select threshold using validation F1
# -------------------------------------------------------------------

best_threshold = threshold_results_df.loc[
    threshold_results_df["F1"].idxmax(),
    "Threshold",
]

best_validation_f1 = threshold_results_df.loc[
    threshold_results_df["Threshold"] == best_threshold,
    "F1",
].iloc[0]

print("\nSelected classification threshold:")
print(f"{best_threshold:.2f}")
print(f"Validation F1: {best_validation_f1:.3f}")


# -------------------------------------------------------------------
# Final test evaluation
#
# The final test set has not been used for model selection.
# -------------------------------------------------------------------

final_test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

final_test_predictions_class = (
    final_test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    final_test_predictions_class,
)

test_precision = precision_score(
    y_test,
    final_test_predictions_class,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    final_test_predictions_class,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    final_test_predictions_class,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    final_test_probabilities,
)

test_pr_auc = average_precision_score(
    y_test,
    final_test_probabilities,
)

confusion = confusion_matrix(
    y_test,
    final_test_predictions_class,
)

tn, fp, fn, tp = confusion.ravel()


print("\nFINAL TEST EVALUATION")
print("=" * 80)
print(f"Model: {best_model_name}")
print(f"Threshold: {best_threshold:.2f}")

print("\nFinal test performance:")
print(f"Accuracy:  {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1 Score:  {test_f1:.3f}")
print(f"ROC-AUC:   {test_roc_auc:.3f}")
print(f"PR-AUC:    {test_pr_auc:.3f}")

print("\nFinal confusion matrix:")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")


# -------------------------------------------------------------------
# Save final test predictions
#
# IMPORTANT:
# Keep the original predictor values alongside predictions.
# This allows subgroup/error analysis later.
# -------------------------------------------------------------------

final_test_predictions = X_test.copy()

final_test_predictions["Actual"] = y_test.values

final_test_predictions["Predicted_Probability"] = (
    final_test_probabilities
)

final_test_predictions["Predicted_Class"] = (
    final_test_predictions_class
)

final_test_predictions.to_csv(
    FINAL_TEST_PATH,
    index=False,
)

print("\nFinal evaluation files saved:")
print(FINAL_TEST_PATH)


# -------------------------------------------------------------------
# Save final model summary
# -------------------------------------------------------------------

final_summary = pd.DataFrame(
    [
        {
            "Model": best_model_name,
            "Threshold": best_threshold,
            "Accuracy": test_accuracy,
            "Precision": test_precision,
            "Recall": test_recall,
            "F1": test_f1,
            "ROC-AUC": test_roc_auc,
            "PR-AUC": test_pr_auc,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
            "Validation_PR-AUC": best_validation_pr_auc,
            "Validation_F1": best_validation_f1,
        }
    ]
)

final_summary.to_csv(
    FINAL_SUMMARY_PATH,
    index=False,
)


# -------------------------------------------------------------------
# 5-fold cross-validation on training data
# -------------------------------------------------------------------

print("\n5-fold cross-validation on training data:")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
)

print(f"Fold ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")


# -------------------------------------------------------------------
# Save logistic feature importance
# -------------------------------------------------------------------

if best_model_name == "Baseline Logistic":

    preprocessor_fitted = best_model.named_steps[
        "preprocessor"
    ]

    classifier = best_model.named_steps["model"]

    feature_names = (
        preprocessor_fitted
        .get_feature_names_out()
    )

    coefficients = classifier.coef_[0]

    logistic_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Absolute_Coefficient": np.abs(coefficients),
        }
    )

    logistic_importance = logistic_importance.sort_values(
        "Absolute_Coefficient",
        ascending=False,
    )

    logistic_importance.to_csv(
        "data/processed/logistic_feature_importance.csv",
        index=False,
    )

    print("\nLogistic feature importance saved.")


# -------------------------------------------------------------------
# Save Random Forest feature importance
# -------------------------------------------------------------------

rf_model = trained_models["Random Forest"]

rf_preprocessor = rf_model.named_steps[
    "preprocessor"
]

rf_classifier = rf_model.named_steps["model"]

rf_feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)

rf_importances = rf_classifier.feature_importances_

rf_importance = pd.DataFrame(
    {
        "Feature": rf_feature_names,
        "Importance": rf_importances,
    }
)

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False,
)

rf_importance.to_csv(
    "data/processed/random_forest_feature_importance.csv",
    index=False,
)

print("Random Forest feature importance saved.")


# -------------------------------------------------------------------
# Save validation predictions
# -------------------------------------------------------------------

validation_predictions_df = pd.concat(
    validation_prediction_rows,
    ignore_index=True,
)

validation_predictions_df.to_csv(
    VALIDATION_PREDICTIONS_PATH,
    index=False,
)


# -------------------------------------------------------------------
# Save model predictions
# -------------------------------------------------------------------

model_predictions = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted_Probability": final_test_probabilities,
        "Predicted_Class": final_test_predictions_class,
    }
)

model_predictions.to_csv(
    MODEL_PREDICTIONS_PATH,
    index=False,
)

print("\nModel prediction files saved:")
print(VALIDATION_PREDICTIONS_PATH)
print(MODEL_PREDICTIONS_PATH)


# -------------------------------------------------------------------
# Save final model
# -------------------------------------------------------------------

joblib.dump(
    best_model,
    MODEL_OUTPUT_PATH,
)

print(f"Final model saved to: {MODEL_OUTPUT_PATH}")

print("\nTraining and evaluation pipeline complete.")