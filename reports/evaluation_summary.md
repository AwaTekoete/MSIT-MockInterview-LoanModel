# Evaluation Summary – Loan Status Klassifikation
**Projekt:** MSIT Mock Interview
**Notebook:** 05_evaluation.ipynb
**Finaler Champion:** LGBM_Full v1

---

## 1. Champion Metriken – Finale Übersicht

### Test-Set (n=102)
| Metrik | Wert | Einordnung |
|--------|------|-----------|
| Precision | **0.8667** | 86.7% der genehmigten Kredite korrekt |
| Recall | **0.8904** | 89.0% aller genehmigbaren Kredite erkannt |
| F1 | **0.8784** | Harmonisches Mittel |
| PR-AUC | **0.9146** | Robust bei Klassenungleichgewicht |
| ROC-AUC | **0.8295** | Allgemeine Trennschärfe |
| MCC | **0.5578** | Ehrlichste Einzelmetrik |

### Bootstrap Konfidenzintervalle (95%, n=1000)
| Metrik | Punkt-Schätzung | 95% KI | KI Breite |
|--------|----------------|--------|-----------|
| Precision | 0.8667 | [0.781, 0.939] | 0.158 |
| Recall | 0.8904 | [0.817, 0.958] | 0.141 |
| F1 | 0.8784 | [0.816, 0.929] | 0.113 |
| PR-AUC | 0.9146 | [0.845, 0.967] | 0.122 |
| ROC-AUC | 0.8295 | [0.731, 0.915] | 0.184 |
| MCC | 0.5578 | [0.380, 0.735] | 0.355 |

**Interpretation:** Breite KI bestätigen statistische Instabilität
durch kleine Test-Set Größe (102 Samples).

---

## 2. Nested Cross-Validation – Realistische Schätzung

| Metrik | Test-Set | Nested CV | Delta | Bewertung |
|--------|---------|-----------|-------|-----------|
| Precision | 0.8667 | 0.8295 | +0.037 | Test überschätzt |
| Recall | 0.8904 | 0.8541 | +0.036 | Test überschätzt |
| F1 | 0.8784 | 0.8412 | +0.037 | Test überschätzt |
| ROC-AUC | 0.8295 | 0.7707 | +0.059 | Test überschätzt |
| PR-AUC | 0.9146 | 0.8693 | +0.045 | Test überschätzt |

**Realistischer Erwartungswert auf neuen Daten:**
PR-AUC ~0.869, Precision ~0.830

---

## 3. Overfitting Analyse & Gegenmassnahme

### Befund
| Metrik | Train | Nested CV | Gap |
|--------|-------|-----------|-----|
| Precision | 1.000 | 0.830 | **0.170** |
| PR-AUC | 1.000 | 0.869 | **0.131** |
| ROC-AUC | 1.000 | 0.771 | **0.229** |

**Ursache:** LGBM 680 Bäume + max_depth=14 auf 406 Trainingssamples –
Modell memoriert Trainingsdaten.

### Gegenmassnahme – v2 Re-Tuning mit Gap-Penalty
Objective_v2 = CV PR-AUC - 0.2 × (Train PR-AUC - CV PR-AUC)

### v1 vs. v2 Vergleich
| Metrik | v1 Gap | v2 Gap | Reduktion |
|--------|--------|--------|-----------|
| Precision | 0.171 | 0.137 | -0.034 ✅ |
| Recall | 0.141 | 0.116 | -0.025 ✅ |
| F1 | 0.156 | 0.127 | -0.030 ✅ |
| PR-AUC | 0.131 | 0.132 | +0.001 ❌ |

### Entscheidung: v1 als Champion behalten
- Nested CV PR-AUC v1 (0.869) > v2 (0.855)
- Gap-Reduktion bei PR-AUC marginal
- Fundamentale Grenze: Datenmenge – kein Tuning kann das beheben

---

## 4. Threshold-Optimierung

| Schwellenwert | Precision | Recall | F1 | FP | Empfehlung |
|--------------|-----------|--------|-----|-----|-----------|
| Default (0.50) | 0.867 | 0.890 | 0.878 | 10 | Gut kalibriert |
| Max MCC (0.41) | 0.872 | 0.932 | 0.901 | 10 | Beste Balance |
| Bank (0.90) | 0.900 | 0.493 | 0.637 | 4 | FP minimal |
| Max Precision (0.98) | 1.000 | 0.110 | 0.198 | 0 | Nicht praxistauglich |

**Operative Empfehlung:**
- Bankperspektive (FP minimieren): Threshold **0.90**
- Gesamtbalance (MCC maximieren): Threshold **0.41**

---

## 5. SHAP Feature Importance

| Rang | Feature | Mean |SHAP| | Richtung |
|------|---------|------------|---------|
| 1 | credit_history | 0.969 | Dominant |
| 2 | loan_amount | 0.519 | Positiv |
| 3 | total_income | 0.435 | Gemischt |
| 4 | coapplicant_income | 0.395 | Positiv |
| 5 | applicant_income | 0.375 | Positiv |
| 6 | property_area_Semiurban | 0.299 | Positiv |
| 7 | married | 0.247 | Positiv |
| 8 | dependents | 0.233 | Gemischt |
| 14 | self_employed | 0.049 | Minimal |
| 15 | has_coapplicant | 0.011 | Irrelevant |

**Key Insight SHAP Waterfall (False Positive Sample 2):**
credit_history=1 (+0.43) und total_income hoch (+0.59) haben
den Ausschlag gegeben – trotz negativer Signale bei education,
coapplicant_income und property_area.

---

## 6. Theoretische Grenzen

| Grenze | Detail | Konsequenz |
|--------|--------|-----------|
| Test-Set Groesse (102) | KI Breite 0.11–0.35 | Metriken statistisch instabil |
| Trainingsgroesse (406) | Overfitting Gap 0.13–0.23 | Reale Performance ~3–6% niedriger |
| Feature Dominance | credit_history SHAP=0.969 | Andere Features unterschaetzt |
| Klassenungleichgewicht | 71.4% vs. 28.6% | TN-Rate nur 65.5% |
| Bayes Error Rate | Geschaetzt ~0.93–0.95 PR-AUC | Wir bei ~90% des Moeglichen |

---

## 7. Finale Empfehlungen

### Operativer Einsatz
| Szenario | Threshold | Begründung |
|----------|-----------|-----------|
| Konservativ (Bank) | 0.90 | FP=4, Precision=0.90 |
| Balanced | 0.41 | Bester MCC, FP=10 |
| Standard | 0.50 | Gut kalibriert |

### Verbesserungspotenzial
| Massnahme | Erwarteter Gewinn |
|-----------|-----------------|
| 10× mehr Daten | +3–5% PR-AUC, Gap < 0.05 |
| 5 stärkere Features | +5–10% PR-AUC |
| Ensemble v1+v2 | +1–2% Stabilität |

---

*Erstellt: 05_evaluation.ipynb – MSIT Mock Interview | 2026*
