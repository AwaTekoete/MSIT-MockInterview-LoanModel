# MSIT Mock Interview – Loan Status Klassifikation

**Projekt:** MSIT Mock Interview 2026
**Autor:** Erik Gerst
**Zeitraum:** April – Mai 2026

---

## Ziel

Entwicklung eines Machine Learning Modells zur Klassifikation von Kreditanträgen (loan_status: 0 = Abgelehnt, 1 = Genehmigt) auf Basis eines realen Loan-Datensatzes.

---

## Champion Modell

| Parameter | Wert |
|-----------|------|
| Algorithmus | LightGBM (LGBM_Full v1) |
| Tuning | Optuna (100 Trials, TPE Sampler) |
| Features | 15 (Full Feature Set) |
| Precision | 0.867 |
| Recall | 0.890 |
| F1 | 0.878 |
| PR-AUC | 0.915 |
| MCC | 0.558 |
| Nested CV PR-AUC | 0.869 |

---

## ProjektstrukturMSIT-MockInterview-LoanModel/
├── data/
│   ├── raw/                    # Originaldatensatz (nicht versioniert)
│   └── processed/              # Verarbeitete Daten (nicht versioniert)
├── notebooks/
│   ├── 01_eda.ipynb            # Explorative Datenanalyse
│   ├── 02_preprocessing.ipynb  # Datenvorbereitung
│   ├── 03_modeling.ipynb       # Baseline Modeling (6 Modelle)
│   ├── 04_hyperparameter_tuning.ipynb  # Optuna Tuning
│   ├── 05_evaluation.ipynb     # Finale Evaluation + Overfitting Analyse
│   └── 06_experiment_tracking.ipynb    # MLflow Tracking
├── src/
│   ├── data_loader.py          # Daten & Modell laden
│   ├── preprocessing.py        # Einzelantrag Preprocessing
│   └── evaluate.py             # Predictions, Metriken, SHAP
├── app/
│   └── main.py                 # Streamlit Web-App
├── models/                     # Gespeicherte Modelle (nicht versioniert)
├── mlruns/                     # MLflow Tracking (nicht versioniert)
├── reports/
│   ├── figures/                # Visualisierungen
│   ├── eda_csv/                # EDA Auswertungen
│   ├── modeling_csv/           # Modeling Auswertungen
│   ├── eda_summary.md
│   ├── preprocessing_summary.md
│   ├── modeling_summary.md
│   ├── tuning_summary.md
│   ├── evaluation_summary.md
│   ├── mlflow_summary.md
│   └── model_limits_and_quality.md
├── presentation/
│   ├── projektanweisungen.md
│   └── Praesentationsstil_Store44.md
├── requirements.txt
└── README.md---

## Projektphasen

| Phase | Notebook | Inhalt |
|-------|----------|--------|
| 1 | 01_eda.ipynb | EDA, Univariat, Bivariat, Kruskal-Wallis, SHAP |
| 2 | 02_preprocessing.ipynb | Imputation, Encoding, Yeo-Johnson, Feature Engineering |
| 3 | 03_modeling.ipynb | 6 Modelle, CV, Feature Importance, ROC/PR Kurven |
| 4 | 04_hyperparameter_tuning.ipynb | Optuna, RandomizedSearch, GridSearch, Feature Sets |
| 5 | 05_evaluation.ipynb | Bootstrap CI, Threshold, SHAP, Nested CV, Overfitting |
| 6 | 06_experiment_tracking.ipynb | MLflow Tracking & Model Registry |

---

## Modellvergleich (Baseline)

| Modell | Precision | PR-AUC | FP |
|--------|-----------|--------|-----|
| Logistic Regression | 0.865 | 0.821 | 10 |
| SVM | 0.863 | 0.886 | 10 |
| LightGBM | 0.857 | 0.907 | 11 |
| Random Forest | 0.854 | 0.950 | 12 |
| CatBoost | 0.848 | 0.871 | 12 |
| XGBoost | 0.840 | 0.875 | 12 |

---

## Streamlit App starten

```bash
streamlit run app/main.py
```

App Features:
- Dashboard: KPI Karten, Confusion Matrix, Threshold Analyse, SHAP Importance
- Einzelantrag: Formular + SHAP Waterfall Erklarung
- Sidebar: Interaktiver Threshold Slider

---

## Installation

```bash
# Virtual Environment aktivieren
.venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt

# App starten
streamlit run app/main.py
```

---

## Overfitting Analyse

Das Champion Modell (LGBM v1) zeigt moderates Overfitting:

| Metrik | Test-Set | Nested CV | Gap |
|--------|---------|-----------|-----|
| Precision | 0.867 | 0.830 | 0.037 |
| PR-AUC | 0.915 | 0.869 | 0.046 |

Ursache: 406 Trainingssamples zu wenig fuer LGBM mit 680 Baeumen.
Gegenmassnahme: Re-Tuning mit Gap-Penalty (v2) dokumentiert in 05_evaluation.ipynb.

---

## Theoretische Obergrenze

Geschaetzte maximale PR-AUC auf diesem Datensatz: ~0.93-0.95
Champion erreicht PR-AUC 0.915 = ~85-90% des theoretisch Erreichbaren.

Hauptlimitierung: Datenmenge (508 Samples) und Feature-Qualitaet.
Details: reports/model_limits_and_quality.md

---

## MLflow UI starten

```bash
mlflow ui --backend-store-uri mlruns
# Browser: http://127.0.0.1:5000
```

---

*MSIT Mock Interview 2026 | Erik Gerst*