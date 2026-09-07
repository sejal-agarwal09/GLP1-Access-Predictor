import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

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

print("\nPreprocessing pipeline created.")

print("\nCategorical predictors:")
print(categorical_features)

