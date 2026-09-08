import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from predictor_selection import PREDICTOR_COLUMNS
from target_definition import TARGET_COLUMN


DATA_PATH = "data/processed/modeling_data.csv"
OUTPUT_PATH = "data/processed/sensitivity_analysis.csv"


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[PREDICTOR_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# ---------------------------------------------------------------
# Same train/test split as the main analysis
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


# ---------------------------------------------------------------
# Method 1: Main analysis
# Most-frequent imputation
# ---------------------------------------------------------------

imputed_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
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
            ),
            PREDICTOR_COLUMNS,
        )
    ]
)

imputed_model = Pipeline(
    steps=[
        ("preprocessor", imputed_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

imputed_model.fit(X_train, y_train)

imputed_probabilities = imputed_model.predict_proba(
    X_test
)[:, 1]

imputed_roc_auc = roc_auc_score(
    y_test,
    imputed_probabilities,
)

imputed_pr_auc = average_precision_score(
    y_test,
    imputed_probabilities,
)


# ---------------------------------------------------------------
# Method 2: Complete-case analysis
# Remove rows with missing predictors
# ---------------------------------------------------------------

train_complete = X_train.notna().all(axis=1)
test_complete = X_test.notna().all(axis=1)

X_train_complete = X_train.loc[train_complete]
y_train_complete = y_train.loc[train_complete]

X_test_complete = X_test.loc[test_complete]
y_test_complete = y_test.loc[test_complete]


complete_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            drop="first",
                        ),
                    ),
                ]
            ),
            PREDICTOR_COLUMNS,
        )
    ]
)

complete_model = Pipeline(
    steps=[
        ("preprocessor", complete_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

complete_model.fit(
    X_train_complete,
    y_train_complete,
)

complete_probabilities = complete_model.predict_proba(
    X_test_complete
)[:, 1]

complete_roc_auc = roc_auc_score(
    y_test_complete,
    complete_probabilities,
)

complete_pr_auc = average_precision_score(
    y_test_complete,
    complete_probabilities,
)


# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------

results = pd.DataFrame(
    [
        {
            "Method": "Most-frequent imputation",
            "Training_N": len(X_train),
            "Test_N": len(X_test),
            "ROC-AUC": imputed_roc_auc,
            "PR-AUC": imputed_pr_auc,
        },
        {
            "Method": "Complete-case analysis",
            "Training_N": len(X_train_complete),
            "Test_N": len(X_test_complete),
            "ROC-AUC": complete_roc_auc,
            "PR-AUC": complete_pr_auc,
        },
    ]
)

print("\nSENSITIVITY ANALYSIS")
print("=" * 80)
print(results.to_string(index=False))

print("\nDifferences:")
print(
    f"ROC-AUC difference: "
    f"{complete_roc_auc - imputed_roc_auc:+.3f}"
)

print(
    f"PR-AUC difference: "
    f"{complete_pr_auc - imputed_pr_auc:+.3f}"
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"\nSaved: {OUTPUT_PATH}")