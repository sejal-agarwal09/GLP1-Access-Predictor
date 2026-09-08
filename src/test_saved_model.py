import joblib
import pandas as pd

from predictor_selection import PREDICTOR_COLUMNS


MODEL_PATH = "models/final_logistic_model.joblib"

# Load the complete trained pipeline
model = joblib.load(MODEL_PATH)

print("Saved model loaded successfully.")
print(f"Model type: {type(model).__name__}")

# Create one example observation
example = pd.DataFrame(
    {
        "CHR_OBEV": [1],
        "DIB_DIBEV": [1],
        "INSURED": [1],
        "DOCVIS_P12M": [1],
        "ACC_HTHUSUAL": [1],
        "P_AGE5YRS_R": [6],
        "P_SEX": [2],
        "DEM_RACEETH": [7],
        "DEM_REGION": [4],
        "P_EDUCATION_I_R": [3],
        "P_POVERTY4_R": [4],
        "BMICAT6": [5],
    }
)

# Make sure all expected predictors are present
example = example[PREDICTOR_COLUMNS]

probability = model.predict_proba(example)[0, 1]

print(f"Example predicted probability: {probability:.4f}")
print(f"Example predicted probability: {probability * 100:.2f}%")