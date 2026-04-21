# =============================================================================
# src/preprocessing.py
# Zweck: Einzelantrag preprocessing für App
# Transformiert Roheingabe → Modell-Input
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import yeojohnson
from src.data_loader import get_feature_names

# --- Yeo-Johnson Lambda Werte (aus 02_preprocessing.ipynb) ---
# Werden benötigt um neue Eingaben konsistent zu transformieren
YJ_LAMBDAS = {
    "applicant_income":   -0.09,
    "coapplicant_income":  0.07,
    "loan_amount":         0.07,
    "total_income":       -0.04,
}

def preprocess_single(input_dict: dict) -> pd.DataFrame:
    """
    Einzelnen Kreditantrag vorverarbeiten.

    Parameters:
        input_dict: Roheingabe aus Streamlit Formular

    Returns:
        DataFrame mit 15 Features – bereit für Modell-Prediction
    """

    # --- Feature Engineering ---
    applicant_income   = float(input_dict["applicant_income"])
    coapplicant_income = float(input_dict["coapplicant_income"])
    loan_amount_term   = float(input_dict["loan_amount_term"])

    has_coapplicant  = 1 if coapplicant_income > 0 else 0
    is_standard_term = 1 if loan_amount_term == 360 else 0
    total_income     = applicant_income + coapplicant_income

    # --- Encoding ---
    gender        = 1 if input_dict["gender"] == "Male" else 0
    married       = 1 if input_dict["married"] == "Yes" else 0
    education     = 1 if input_dict["education"] == "Not Graduate" else 0
    self_employed = 1 if input_dict["self_employed"] == "Yes" else 0

    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents = dependents_map[str(input_dict["dependents"])]

    property_area = input_dict["property_area"]
    property_area_Semiurban = 1 if property_area == "Semiurban" else 0
    property_area_Urban     = 1 if property_area == "Urban" else 0

    credit_history = int(input_dict["credit_history"])

    # --- Yeo-Johnson Transformation ---
    def yj_transform(value: float, lam: float) -> float:
        """Yeo-Johnson Transformation für einzelnen Wert."""
        if value >= 0:
            if abs(lam) < 1e-10:
                return float(np.log1p(value))
            else:
                return float(((value + 1) ** lam - 1) / lam)
        else:
            if abs(2 - lam) < 1e-10:
                return float(-np.log1p(-value))
            else:
                return float(-((-value + 1) ** (2 - lam) - 1) / (2 - lam))

    applicant_income_t   = yj_transform(applicant_income,
                                         YJ_LAMBDAS["applicant_income"])
    coapplicant_income_t = yj_transform(coapplicant_income,
                                         YJ_LAMBDAS["coapplicant_income"])
    loan_amount_t        = yj_transform(float(input_dict["loan_amount"]),
                                         YJ_LAMBDAS["loan_amount"])
    total_income_t       = yj_transform(total_income,
                                         YJ_LAMBDAS["total_income"])

    # --- Feature DataFrame erstellen ---
    features = {
        "gender":                  gender,
        "married":                 married,
        "dependents":              dependents,
        "education":               education,
        "self_employed":           self_employed,
        "applicant_income":        applicant_income_t,
        "coapplicant_income":      coapplicant_income_t,
        "loan_amount":             loan_amount_t,
        "loan_amount_term":        loan_amount_term,
        "credit_history":          credit_history,
        "has_coapplicant":         has_coapplicant,
        "is_standard_term":        is_standard_term,
        "total_income":            total_income_t,
        "property_area_Semiurban": property_area_Semiurban,
        "property_area_Urban":     property_area_Urban,
    }

    return pd.DataFrame([features])[get_feature_names()]