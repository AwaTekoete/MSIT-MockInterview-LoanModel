# src/evaluate.py
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)

def predict_single(model, X, threshold=0.5):
    proba    = model.predict_proba(X)[0][1]
    decision = 1 if proba >= threshold else 0
    return decision, proba

def compute_test_metrics(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= threshold).astype(int)
    cm           = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_proba), 4),
        "pr_auc":    round(average_precision_score(y_test, y_pred_proba), 4),
        "mcc":       round(matthews_corrcoef(y_test, y_pred), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def compute_shap_single(model, X_single):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)
    if isinstance(shap_values, list):
        sv             = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        sv             = shap_values
        expected_value = explainer.expected_value
    return shap.Explanation(
        values=sv[0],
        base_values=expected_value,
        data=X_single.iloc[0],
        feature_names=X_single.columns.tolist()
    )

def get_threshold_analysis(model, X_test, y_test):
    y_proba    = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    results    = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        results.append({
            "Threshold": round(thr, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "FP":        int(((y_pred == 1) & (y_test == 0)).sum()),
            "FN":        int(((y_pred == 0) & (y_test == 1)).sum()),
        })
    return pd.DataFrame(results)