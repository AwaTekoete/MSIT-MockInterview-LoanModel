# Hyperparameter Tuning Summary – Loan Status Klassifikation
**Projekt:** MSIT Mock Interview  
**Notebook:** 04_hyperparameter_tuning.ipynb  
**Basis:** modeling_summary.md  

---

## 1. Tuning Strategie

| Parameter | Entscheidung | Begruendung |
|-----------|-------------|------------|
| Kandidaten | RF, LightGBM, SVM | Top 3 aus Baseline Modeling |
| Feature Sets | Full (15) vs. Reduced (12) | Ansatz B – datengetrieben |
| Entfernte Features (Reduced) | loan_amount, loan_amount_term, property_area_Urban | Permutation Importance < 0 |
| RF Methode | RandomizedSearchCV (n_iter=100) | Grosser Parameterraum |
| LightGBM Methode | Optuna (n_trials=100) | Bayesianisch – effizienter |
| SVM Methode | GridSearchCV | Kleiner Parameterraum |
| Optimierungsziel | PR-AUC (5-Fold CV) | Robust bei Klassenungleichgewicht |

---

## 2. Tuning Ergebnisse – Alle Modelle

| Modell | CV PR-AUC | Precision | Recall | F1 | PR-AUC | MCC | N Features |
|--------|-----------|-----------|--------|-----|--------|-----|-----------|
| LGBM_Full | 0.8539 | **0.8667** | **0.8904** | **0.8784** | 0.9146 | **0.5578** | 15 |
| LGBM_Reduced | 0.8496 | 0.8630 | 0.8630 | 0.8630 | 0.8815 | 0.5182 | 12 |
| SVM_Full | 0.8453 | 0.8630 | 0.8630 | 0.8630 | 0.8686 | 0.5182 | 15 |
| SVM_Reduced | 0.8538 | 0.8611 | 0.8493 | 0.8552 | 0.9119 | 0.4994 | 12 |
| RF_Reduced | 0.8536 | 0.8592 | 0.8356 | 0.8472 | **0.9161** | 0.4813 | 12 |
| RF_Full | 0.8547 | 0.8243 | 0.8356 | 0.8299 | 0.9158 | 0.3915 | 15 |

---

## 3. Champion-Modell: LGBM_Full

### Begruendung
| Kriterium | Wert | Rang |
|-----------|------|------|
| Precision (Primaermetrik) | 0.8667 | #1 |
| Recall | 0.8904 | #1 |
| F1 | 0.8784 | #1 |
| MCC | 0.5578 | #1 |
| Score gesamt (0.4×P + 0.4×PR-AUC + 0.2×F1) | 0.8882 | #1 |

### Optimale Hyperparameter
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
| Feature Set | Full (15 Features) |

---

## 4. Feature Set Analyse – Ansatz B

| Modell | Full Precision | Reduced Precision | Gewinner |
|--------|---------------|------------------|---------|
| LightGBM | **0.8667** | 0.8630 | Full |
| SVM | **0.8630** | 0.8611 | Full |
| Random Forest | 0.8243 | **0.8592** | Reduced |

**Befund:** Feature Set Entscheidung ist modellabhaengig.
LightGBM und SVM profitieren vom Full Set – internes Regularisierung handhabt irrelevante Features.
Random Forest profitiert vom Reduced Set – sensibler gegenueber Rauschen.

---

## 5. Verbesserung gegenueber Baseline

| Metrik | Baseline LGBM | Champion LGBM_Full | Delta |
|--------|--------------|-------------------|-------|
| Precision | 0.8571 | **0.8667** | +0.010 |
| Recall | 0.9041 | 0.8904 | -0.014 |
| F1 | 0.8800 | 0.8784 | -0.002 |
| PR-AUC | 0.9068 | **0.9146** | +0.008 |

**Tuning hat Precision und PR-AUC verbessert** – minimale Reduktion Recall akzeptabel
aus Bankperspektive (False Positives reduziert).

---

## 6. Theoretische Grenzen

| Grenze | Detail |
|--------|--------|
| Test-Set Groesse | 102 Samples – Metrik-Varianz hoch |
| Optuna n_trials=100 | Lokales Optimum moeglich – mehr Trials wuerden stabilisieren |
| Kein Threshold-Tuning | Default 0.5 – Optimierung in 05_evaluation.ipynb |
| Keine SHAP-Analyse | Feature Interactions nicht erklaert |
| Kleine Stichprobe | Overfitting-Risiko trotz CV |

---

## 7. Naechste Schritte – 05_evaluation.ipynb

- Threshold-Optimierung (Bankperspektive)
- SHAP-Analyse (Modellerklaerbarkeit)
- Finale Metriken + Visualisierungen
- Kritische Gesamtbewertung

---

*Erstellt: 04_hyperparameter_tuning.ipynb – MSIT Mock Interview | 2026*
