# Modeling Summary – Loan Status Klassifikation
**Projekt:** MSIT Mock Interview  
**Notebook:** 03_modeling.ipynb  
**Basis:** preprocessing_summary.md  

---

## 1. Modeling Strategie

| Parameter | Entscheidung | Begründung |
|-----------|-------------|-----------|
| Modelle | 6 (LR, SVM, RF, XGB, LGBM, CatBoost) | Breiter Vergleich linear vs. nicht-linear |
| Klassenungleichgewicht | class_weight='balanced' | 2.54:1 – kein extremes Ungleichgewicht |
| Primärmetrik | Precision (Klasse 1) | Bankperspektive – FP teuerster Fehler |
| Sekundärmetrik | F1-Score, PR-AUC | Balance Precision/Recall |
| Validierung | 5-Fold Stratified CV + Test-Set | Robuste Schätzung |
| Skalierung | StandardScaler nur LR + SVM | Baumbasierte Modelle skalierungsunabhängig |

---

## 2. Cross-Validation Ergebnisse (5-Fold, sortiert nach Precision)

| Modell | Precision | Std | Recall | F1 | ROC-AUC | PR-AUC |
|--------|-----------|-----|--------|-----|---------|--------|
| SVM | 0.8088 | 0.0314 | 0.7966 | 0.7984 | 0.7271 | 0.8425 |
| LightGBM | 0.8063 | 0.0170 | 0.8724 | 0.8366 | 0.7189 | 0.8333 |
| Random Forest | 0.8051 | 0.0275 | 0.9621 | 0.8762 | 0.7254 | 0.8426 |
| Logistic Regression | 0.8026 | 0.0157 | 0.7724 | 0.7857 | 0.6946 | 0.8250 |
| CatBoost | 0.8012 | 0.0332 | 0.8517 | 0.8229 | 0.7101 | 0.8353 |
| XGBoost | 0.7979 | 0.0156 | 0.8414 | 0.8169 | 0.7180 | 0.8387 |

---

## 3. Test-Set Ergebnisse (sortiert nach Precision)

| Modell | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC | FP |
|--------|-----------|--------|-----|---------|--------|-----|----|
| Logistic Regression | 0.8649 | 0.8767 | 0.8707 | 0.7454 | 0.8209 | 0.5376 | 10 |
| SVM | 0.8630 | 0.8630 | 0.8630 | 0.7898 | 0.8858 | 0.5182 | 10 |
| LightGBM | 0.8571 | 0.9041 | 0.8800 | 0.8162 | 0.9068 | 0.5503 | 11 |
| Random Forest | 0.8537 | 0.9589 | 0.9032 | 0.8880 | 0.9501 | 0.6193 | 12 |
| CatBoost | 0.8481 | 0.9178 | 0.8816 | 0.8026 | 0.8711 | 0.5440 | 12 |
| XGBoost | 0.8400 | 0.8630 | 0.8514 | 0.7917 | 0.8753 | 0.4593 | 12 |

---

## 4. Confusion Matrix Analyse (Bankperspektive)

| Modell | TN | FP | FN | TP | FP Bewertung |
|--------|----|----|----|----|-------------|
| Logistic Regression | 19 | 10 | 9 | 64 | Niedrigste FP |
| SVM | 19 | 10 | 10 | 63 | Niedrigste FP |
| LightGBM | 18 | 11 | 7 | 66 | Gut |
| Random Forest | 17 | 12 | 3 | 70 | FP zu hoch |
| XGBoost | 17 | 12 | 10 | 63 | FP zu hoch |
| CatBoost | 17 | 12 | 6 | 67 | FP zu hoch |

---

## 5. Feature Importance – Permutation (PR-AUC Loss)

| Rang | Feature | Mean Loss | Bewertung |
|------|---------|-----------|-----------|
| 1 | credit_history | 0.1268 | Dominant |
| 2 | applicant_income | 0.0240 | Relevant |
| 3 | coapplicant_income | 0.0230 | Relevant |
| 4 | property_area_Semiurban | 0.0176 | Relevant |
| 5 | total_income | 0.0140 | Moderat |
| 6 | married | 0.0089 | Schwach |
| 7 | education | 0.0035 | Minimal |
| 8 | is_standard_term | 0.0034 | Minimal |
| 9 | gender | 0.0033 | Minimal |
| 10 | has_coapplicant | 0.0020 | Minimal |
| 11 | self_employed | -0.0003 | Irrelevant |
| 12 | dependents | -0.0005 | Irrelevant |
| 13 | loan_amount | -0.0016 | Schaedlich |
| 14 | property_area_Urban | -0.0032 | Schaedlich |
| 15 | loan_amount_term | -0.0082 | Schaedlich |

---

## 6. ROC-AUC & PR-AUC Kurven

| Modell | ROC-AUC | PR-AUC |
|--------|---------|--------|
| Random Forest | 0.888 | 0.950 |
| LightGBM | 0.816 | 0.907 |
| CatBoost | 0.803 | 0.871 |
| XGBoost | 0.792 | 0.875 |
| SVM | 0.790 | 0.886 |
| Logistic Regression | 0.745 | 0.821 |

---

## 7. Theoretische Grenzen

| Grenze | Detail |
|--------|--------|
| Test-Set Groesse | 102 Samples – hohe Metrik-Varianz |
| Feature Dominance | credit_history dominiert – andere Features werden unterschaetzt |
| Native Feature Importance | Ueberschaetzt kontinuierliche Features – Permutation robuster |
| CV Instabilitaet | Hohe Std bei SVM Recall (0.0895) – kleine Folds |
| Default-Parameter | Alle Modelle ungetunt – Baseline-Vergleich |

---

## 8. Entscheidungen fuer 04_hyperparameter_tuning.ipynb

### Tuning-Kandidaten
| Rang | Modell | Begruendung |
|------|--------|------------|
| 1 | Random Forest | Beste PR-AUC (0.950) + ROC-AUC (0.888) |
| 2 | LightGBM | Beste Balance aller Metriken |
| 3 | SVM | Niedrigste FP – Bankperspektive |

### Feature-Entscheidungen fuer Tuning
| Feature | Entscheidung | Begruendung |
|---------|-------------|------------|
| loan_amount_term | Entfernen | Schaedlich (-0.0082) |
| property_area_Urban | Entfernen | Schaedlich (-0.0032) |
| loan_amount | Evaluieren | Schaedlich aber kontextuell relevant |
| self_employed, dependents | Evaluieren | Nahe null |

### Tuning-Methoden
- Random Forest → RandomizedSearch + HyperOpt
- LightGBM → HyperOpt
- SVM → GridSearch

### Threshold-Tuning
- Default-Schwellenwert 0.5 wird in Evaluation optimiert
- Bankperspektive: Schwellenwert erhoehen → Precision steigt, Recall sinkt

---

*Erstellt: 03_modeling.ipynb – MSIT Mock Interview | 2026*
