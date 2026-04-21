# MLflow Experiment Tracking Summary
**Projekt:** MSIT Mock Interview – Loan Status Klassifikation  
**Experiment:** MSIT_MockInterview_LoanStatus  
**Tracking URI:** ../mlruns  
**MLflow Version:** 3.11.1  

---

## 1. Registrierte Modelle

| Modell | Version | Status | Run ID |
|--------|---------|--------|--------|
| LoanStatus_Champion | 1 | **Production** | 0e355454 |
| LoanStatus_GapPenalty | 1 | Archiviert | 0fae8f9d |

---

## 2. Experiment Runs – Vergleich

| Run | Precision | Recall | F1 | PR-AUC | MCC | Nested CV PR-AUC | Gap PR-AUC |
|-----|-----------|--------|-----|--------|-----|-----------------|-----------|
| LGBM_Full_v1_Champion | **0.8667** | **0.8904** | **0.8784** | **0.9146** | **0.5578** | **0.8693** | 0.1307 |
| LGBM_Full_v2_GapPenalty | 0.8400 | 0.8630 | 0.8514 | 0.9077 | 0.4593 | 0.8553 | 0.1324 |

---

## 3. Geloggte Artefakte – Champion v1

| Kategorie | Inhalt |
|-----------|--------|
| `plots/` | Alle Visualisierungen (EDA, Modeling, Evaluation) |
| `metrics_csv/` | Alle CSV Auswertungen |
| `summaries/` | EDA, Preprocessing, Modeling, Tuning, Evaluation Summaries |
| `model/` | LightGBM Modell (MLflow Format) |

---

## 4. Geloggte Parameter – Champion v1

| Parameter | Wert |
|-----------|------|
| n_estimators | 680 |
| max_depth | 14 |
| learning_rate | 0.0314 |
| num_leaves | 58 |
| min_child_samples | 6 |
| subsample | 0.830 |
| colsample_bytree | 0.797 |
| reg_alpha | 7.71e-05 |
| reg_lambda | 9.08 |
| is_unbalance | True |

---

## 5. Geloggte Metriken – Champion v1

### Test-Set Metriken
| Metrik | Wert |
|--------|------|
| precision | 0.8667 |
| recall | 0.8904 |
| f1 | 0.8784 |
| pr_auc | 0.9146 |
| roc_auc | 0.8295 |
| mcc | 0.5578 |

### Robustheit & Stabilität
| Metrik | Wert |
|--------|------|
| nested_cv_pr_auc | 0.8693 |
| nested_cv_precision | 0.8295 |
| overfitting_gap_pr_auc | 0.1307 |
| overfitting_gap_precision | 0.1705 |
| bootstrap_ci_precision_low | 0.7808 |
| bootstrap_ci_precision_high | 0.9390 |
| bootstrap_ci_pr_auc_low | 0.8453 |
| bootstrap_ci_pr_auc_high | 0.9672 |

### Operativer Einsatz
| Metrik | Wert |
|--------|------|
| optimal_threshold_mcc | 0.41 |
| optimal_threshold_bank | 0.90 |

---

## 6. MLflow UI – Anleitung

```bash
# Terminal im Projektordner
mlflow ui --backend-store-uri ../mlruns

# Browser öffnen
http://127.0.0.1:5000
```

Im UI verfügbar:
- Experiment Run Vergleich
- Parameter & Metriken parallel
- Artefakte direkt ansehen
- Model Registry mit Production Tag

---

## 7. Reproduzierbarkeit

| Aspekt | Wert |
|--------|------|
| random_state | 42 |
| CV Strategie | StratifiedKFold (5-Fold) |
| Train/Test Split | 80/20 stratified |
| Datensatz | loans_modified.csv (563 Zeilen original) |
| Preprocessing | 02_preprocessing.ipynb |
| Tuning | Optuna TPESampler (seed=42) |

---

*Erstellt: 06_experiment_tracking.ipynb – MSIT Mock Interview | 2026*
