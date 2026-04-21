# =============================================================================
# src/data_loader.py
# Zweck: Datensatz und Modell laden
# Wird von app/main.py importiert
# =============================================================================

import pandas as pd
import joblib
from pathlib import Path

# --- Pfade ---
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_lgbm_full_v1_final.joblib"
DATA_PATH  = BASE_DIR / "data" / "processed" / "loans_processed.csv"

def load_model():
    """Champion Modell laden."""
    return joblib.load(MODEL_PATH)

def load_processed_data():
    """Verarbeiteten Datensatz laden."""
    return pd.read_csv(DATA_PATH)

def load_test_data():
    """Train/Test Split laden."""
    X_test  = pd.read_csv(BASE_DIR / "data" / "processed" / "X_test.csv")
    y_test  = pd.read_csv(BASE_DIR / "data" / "processed" / "y_test.csv").squeeze()
    X_train = pd.read_csv(BASE_DIR / "data" / "processed" / "X_train.csv")
    y_train = pd.read_csv(BASE_DIR / "data" / "processed" / "y_train.csv").squeeze()
    return X_train, X_test, y_train, y_test

def get_feature_names():
    """Feature-Liste zurückgeben."""
    return [
        "gender", "married", "dependents", "education",
        "self_employed", "applicant_income", "coapplicant_income",
        "loan_amount", "loan_amount_term", "credit_history",
        "has_coapplicant", "is_standard_term", "total_income",
        "property_area_Semiurban", "property_area_Urban"
    ]