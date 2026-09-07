import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


DATA_PATH = "data/processed/modeling_data.csv"
TARGET_COLUMN = "GLP_MED12M"


# Load the cleaned modeling dataset
df = pd.read_csv(DATA_PATH)

# Separate predictors from target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


print("Dataset loaded.")
print(f"Rows: {len(df):,}")
print(f"Predictors: {X.shape[1]}")
print(f"Target: {TARGET_COLUMN}")

print("\nTarget distribution:")
print(y.value_counts())


# Split the data into training and testing sets
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


# Define the categorical predictors
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


# Preprocess categorical predictors:
# 1. Fill missing values with the most common category
# 2. Convert categorical values into one-hot encoded features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# Apply the categorical preprocessing to all predictor variables
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features)
    ]
)


print("\nPreprocessing pipeline created.")


# Define the baseline logistic regression model
model = LogisticRegression(
    max_iter=1000
)


print("\nBaseline model created.")


# Combine preprocessing and the machine learning model
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\nComplete ML pipeline created.")


# Train the complete pipeline using only the training data
pipeline.fit(X_train, y_train)

print("\nModel training complete.")

# Generate predictions on the unseen test set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


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

# Examine how different probability thresholds affect precision and recall
thresholds_to_test = [0.20, 0.30, 0.40, 0.50]

print("\nThreshold analysis:")

for threshold in thresholds_to_test:
    threshold_predictions = (y_prob >= threshold).astype(int)

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
    
    # Perform 5-fold stratified cross-validation on the training data
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