import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier


DATA_PATH = "data/processed/modeling_data.csv"
TARGET_COLUMN = "GLP_MED12M"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


print("Dataset loaded.")
print(f"Rows: {len(df):,}")
print(f"Predictors: {X.shape[1]}")
print(f"Target: {TARGET_COLUMN}")

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain/test split:")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# PREPROCESSING
# ============================================================

categorical_features = [
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


print("\nCategorical predictors:")
print(categorical_features)


# Fill missing values and one-hot encode categorical variables.
#
# sparse_output=False is important because
# HistGradientBoostingClassifier requires dense input.
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nPreprocessing pipeline created.")


# ============================================================
# BASELINE LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

print("\nBaseline model created.")


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\nComplete ML pipeline created.")


pipeline.fit(X_train, y_train)

print("\nModel training complete.")


y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()


print("\nConfusion matrix:")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")


print("\nModel evaluation:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC-AUC:   {roc_auc:.3f}")
print(f"PR-AUC:    {pr_auc:.3f}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds_to_test = [
    0.20,
    0.30,
    0.40,
    0.50
]

print("\nThreshold analysis:")

for threshold in thresholds_to_test:

    threshold_predictions = (
        y_prob >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {threshold_precision:.3f} | "
        f"Recall: {threshold_recall:.3f} | "
        f"F1: {threshold_f1:.3f}"
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)


print("\n5-fold cross-validation:")
print(f"Fold ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")


# ============================================================
# NO-SKILL BASELINE
# ============================================================

dummy_model = DummyClassifier(
    strategy="prior"
)

dummy_model.fit(
    X_train,
    y_train
)

dummy_probabilities = dummy_model.predict_proba(
    X_test
)[:, 1]


dummy_roc_auc = roc_auc_score(
    y_test,
    dummy_probabilities
)

dummy_pr_auc = average_precision_score(
    y_test,
    dummy_probabilities
)


print("\nNo-skill baseline:")
print(f"ROC-AUC: {dummy_roc_auc:.3f}")
print(f"PR-AUC:  {dummy_pr_auc:.3f}")


# ============================================================
# CLASS-WEIGHTED LOGISTIC REGRESSION
# ============================================================

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nClass-weighted model created.")


balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", balanced_model)
    ]
)


print("\nClass-weighted ML pipeline created.")


balanced_pipeline.fit(
    X_train,
    y_train
)

print("\nClass-weighted model training complete.")


balanced_pred = balanced_pipeline.predict(
    X_test
)

balanced_prob = balanced_pipeline.predict_proba(
    X_test
)[:, 1]


print("\nClass-weighted predictions generated.")


balanced_accuracy = accuracy_score(
    y_test,
    balanced_pred
)

balanced_precision = precision_score(
    y_test,
    balanced_pred,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_pred,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_pred,
    zero_division=0
)

balanced_roc_auc = roc_auc_score(
    y_test,
    balanced_prob
)

balanced_pr_auc = average_precision_score(
    y_test,
    balanced_prob
)


print("\nClass-weighted model evaluation:")
print(f"Accuracy:  {balanced_accuracy:.3f}")
print(f"Precision: {balanced_precision:.3f}")
print(f"Recall:    {balanced_recall:.3f}")
print(f"F1 Score:  {balanced_f1:.3f}")
print(f"ROC-AUC:   {balanced_roc_auc:.3f}")
print(f"PR-AUC:    {balanced_pr_auc:.3f}")


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nRandom Forest model created.")


random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest_model)
    ]
)


print("\nRandom Forest pipeline created.")


random_forest_pipeline.fit(
    X_train,
    y_train
)

print("\nRandom Forest training complete.")


random_forest_pred = random_forest_pipeline.predict(
    X_test
)

random_forest_prob = (
    random_forest_pipeline.predict_proba(X_test)[:, 1]
)


print("\nRandom Forest predictions generated.")


random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

random_forest_precision = precision_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

random_forest_recall = recall_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

random_forest_roc_auc = roc_auc_score(
    y_test,
    random_forest_prob
)

random_forest_pr_auc = average_precision_score(
    y_test,
    random_forest_prob
)


print("\nRandom Forest evaluation:")
print(f"Accuracy:  {random_forest_accuracy:.3f}")
print(f"Precision: {random_forest_precision:.3f}")
print(f"Recall:    {random_forest_recall:.3f}")
print(f"F1 Score:  {random_forest_f1:.3f}")
print(f"ROC-AUC:   {random_forest_roc_auc:.3f}")
print(f"PR-AUC:    {random_forest_pr_auc:.3f}")


# ============================================================
# HISTOGRAM GRADIENT BOOSTING
# ============================================================

gradient_boosting_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    random_state=42
)

print("\nGradient Boosting model created.")


gradient_boosting_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", gradient_boosting_model)
    ]
)


print("\nGradient Boosting pipeline created.")


gradient_boosting_pipeline.fit(
    X_train,
    y_train
)

print("\nGradient Boosting training complete.")


gradient_boosting_pred = (
    gradient_boosting_pipeline.predict(X_test)
)

gradient_boosting_prob = (
    gradient_boosting_pipeline.predict_proba(X_test)[:, 1]
)


print("\nGradient Boosting predictions generated.")


gradient_boosting_accuracy = accuracy_score(
    y_test,
    gradient_boosting_pred
)

gradient_boosting_precision = precision_score(
    y_test,
    gradient_boosting_pred,
    zero_division=0
)

gradient_boosting_recall = recall_score(
    y_test,
    gradient_boosting_pred,
    zero_division=0
)

gradient_boosting_f1 = f1_score(
    y_test,
    gradient_boosting_pred,
    zero_division=0
)

gradient_boosting_roc_auc = roc_auc_score(
    y_test,
    gradient_boosting_prob
)

gradient_boosting_pr_auc = average_precision_score(
    y_test,
    gradient_boosting_prob
)


print("\nGradient Boosting evaluation:")
print(f"Accuracy:  {gradient_boosting_accuracy:.3f}")
print(f"Precision: {gradient_boosting_precision:.3f}")
print(f"Recall:    {gradient_boosting_recall:.3f}")
print(f"F1 Score:  {gradient_boosting_f1:.3f}")
print(f"ROC-AUC:   {gradient_boosting_roc_auc:.3f}")
print(f"PR-AUC:    {gradient_boosting_pr_auc:.3f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\nModel comparison:")
print("-" * 100)

print(
    f"{'Metric':<15}"
    f"{'Baseline':>15}"
    f"{'Balanced':>15}"
    f"{'Random Forest':>20}"
    f"{'Gradient Boosting':>22}"
)

print("-" * 100)


print(
    f"{'Accuracy':<15}"
    f"{accuracy:>15.3f}"
    f"{balanced_accuracy:>15.3f}"
    f"{random_forest_accuracy:>20.3f}"
    f"{gradient_boosting_accuracy:>22.3f}"
)


print(
    f"{'Precision':<15}"
    f"{precision:>15.3f}"
    f"{balanced_precision:>15.3f}"
    f"{random_forest_precision:>20.3f}"
    f"{gradient_boosting_precision:>22.3f}"
)


print(
    f"{'Recall':<15}"
    f"{recall:>15.3f}"
    f"{balanced_recall:>15.3f}"
    f"{random_forest_recall:>20.3f}"
    f"{gradient_boosting_recall:>22.3f}"
)


print(
    f"{'F1 Score':<15}"
    f"{f1:>15.3f}"
    f"{balanced_f1:>15.3f}"
    f"{random_forest_f1:>20.3f}"
    f"{gradient_boosting_f1:>22.3f}"
)


print(
    f"{'ROC-AUC':<15}"
    f"{roc_auc:>15.3f}"
    f"{balanced_roc_auc:>15.3f}"
    f"{random_forest_roc_auc:>20.3f}"
    f"{gradient_boosting_roc_auc:>22.3f}"
)


print(
    f"{'PR-AUC':<15}"
    f"{pr_auc:>15.3f}"
    f"{balanced_pr_auc:>15.3f}"
    f"{random_forest_pr_auc:>20.3f}"
    f"{gradient_boosting_pr_auc:>22.3f}"
)


print("-" * 100)