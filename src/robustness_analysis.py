import pandas as pd
import numpy as np

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

df = pd.read_csv(DATA_PATH)

X = df[PREDICTOR_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()


# Same preprocessing used by the final model
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
        )
    ]
)


seeds = [1, 21, 42, 84, 168]

results = []

print("ROBUSTNESS ANALYSIS")
print("=" * 80)

for seed in seeds:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    results.append(
        {
            "Random_State": seed,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }
    )

    print(
        f"Seed {seed}: "
        f"ROC-AUC = {roc_auc:.3f} | "
        f"PR-AUC = {pr_auc:.3f}"
    )


results_df = pd.DataFrame(results)

print("\nSummary:")
print(
    f"Mean ROC-AUC: "
    f"{results_df['ROC-AUC'].mean():.3f}"
)

print(
    f"ROC-AUC SD: "
    f"{results_df['ROC-AUC'].std():.3f}"
)

print(
    f"Mean PR-AUC: "
    f"{results_df['PR-AUC'].mean():.3f}"
)

print(
    f"PR-AUC SD: "
    f"{results_df['PR-AUC'].std():.3f}"
)


results_df.to_csv(
    "data/processed/robustness_results.csv",
    index=False,
)

print("\nSaved:")
print("data/processed/robustness_results.csv")