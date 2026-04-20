# MIST Final Mock Interview – Projektanweisungen
**Projekt:** Loan Status Klassifikationsmodell  
**Zeitraum:** 20. April – 15. Mai 2026  
**Speicherort:** `C:\_AA-LapTop-MSIT\_MyMSIT_MockInterview\presentation\projektanweisungen.md`

---

## Sprache & Stil
- Deutsch, technische Begriffe bleiben Englisch
- Sachlich, Markdown-Format, keine Personalpronomina (kein „du", „ich", „wir" etc.)
- Professioneller Ton, kein Smalltalk

---

## Level & Arbeitsweise
- Fortgeschrittener – kurze Grundlagenerklärungen, tiefe Erklärung / Begründung
- Schrittweise Führung – kein komplettes Notebook auf einmal
- Code pro Schritt mit Kommentaren (PEP8, professionelle Struktur)
- Nach jedem Schritt: Ergebnisse teilen → kritische Analyse → Entscheidung über nächsten Schritt
- IDE: PyCharm
- Notebooks zuerst → Refactoring in `src/` ab Phase 3 (Modeling)

---

## ML-Tiefe
- Konzeptionelle Erklärungen inklusive theoretischer Grenzen
- Ziel: Maximum herausholen was mit heutigen Mitteln auf diesem Datensatz möglich ist
- Begründung jeder Entscheidung – nicht nur „was", sondern „warum"
- Grenzen der Methoden aufzeigen (Bias-Variance Tradeoff, Datenmenge, Feature-Qualität etc.)

---

## Projektphasen

| # | Notebook | Inhalt |
|---|----------|--------|
| 1 | `01_eda.ipynb` | Explorative Datenanalyse & Visualisierungen |
| 2 | `02_preprocessing.ipynb` | Datenvorbereitung & Feature Engineering |
| 3 | `03_modeling.ipynb` | Modellauswahl & Training |
| 4 | `04_hyperparameter_tuning.ipynb` | HyperOpt / GridSearch / RandomizedSearch |
| 5 | `05_evaluation.ipynb` | Metriken, kritische Analyse, Visualisierungen |
| 6 | `06_experiment_tracking.ipynb` | MLflow – Parameter, Metriken, Artefakte |

---

## Modelle & Optimierung
- Modellkandidaten werden gemeinsam definiert und begründet
- Metriken werden gemeinsam erarbeitet – technisch modern + aufgabenspezifisch begründet
- Professionelles Hyperparameter-Tuning: HyperOpt, GridSearch, RandomizedSearch
- MLflow Experiment-Tracking (Parameter, Metriken, Artefakte loggen)
- Modelloptimierung als besonderer Schwerpunkt

---

## Evaluationsmetriken
- Gemeinsam entwickelt – technisch modern, nicht nur Accuracy
- Aufgabenspezifisch begründet (Klassifikation, Klassenungleichgewicht beachten)
- Kritische Analyse aller Zwischen- und Endergebnisse
- Ergebnisse visualisieren (Confusion Matrix, ROC-AUC, Precision-Recall etc.)

---

## Projektstruktur

```
C:\_AA-LapTop-MSIT\_MyMSIT_MockInterview\
│
├── data\
│   ├── raw\                        ← Originaldatensatz (nie überschreiben)
│   └── processed\                  ← Bereinigte/transformierte Daten
│
├── notebooks\
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_experiment_tracking.ipynb
│
├── src\                            ← Produktiver Python-Code (ab Phase 3)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models\                         ← Gespeicherte Modelle (.pkl, .joblib)
├── mlruns\                         ← MLflow Tracking (auto-generiert)
├── app\                            ← Streamlit Web-App
│   └── main.py                     ← importiert ausschließlich aus src\
├── reports\
│   └── figures\                    ← Visualisierungen & Plots
├── presentation\
│   ├── projektanweisungen.md       ← diese Datei
│   └── Praesentationsstil_Store44.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## GitHub

- **Repository:** neu, `Private`
- **Empfohlener Name:** `MSIT-MockInterview-LoanModel`
- **Commit-Konvention:** Conventional Commits

### Commit-Types

| Type | Verwendung |
|------|------------|
| `feat` | Neues Feature / neuer Schritt |
| `fix` | Bugfix |
| `docs` | README, Kommentare, Dokumentation |
| `refactor` | Code umstrukturiert (kein neues Feature) |
| `test` | Tests hinzugefügt |
| `chore` | Setup, Konfiguration, `.gitignore` |
| `viz` | Neue Visualisierung |
| `model` | Modell hinzugefügt oder verändert |

### Beispiele
```
chore: Projektstruktur initialisiert
feat: 01_eda.ipynb – erste Datenübersicht
fix: fehlende Werte in loan_amount behandelt
refactor: preprocessing in src/ überführt
model: Random Forest Basismodell trainiert
viz: Korrelationsmatrix hinzugefügt
docs: README aktualisiert
```

### `.gitignore`
```
# Daten
data/raw/
data/processed/

# Modelle
models/

# MLflow
mlruns/

# Python
__pycache__/
*.pkl
*.joblib
*.pyc
.env

# Jupyter Checkpoints
.ipynb_checkpoints/

# PyCharm
.idea/
```

---

## Deployment
- Streamlit Web-App (`app/main.py`)
- `app/main.py` importiert ausschließlich aus `src/`
- Lokal oder Cloud

---

## Präsentation
- Stil-Referenz: `presentation/Praesentationsstil_Store44.md`
- Format: Live-Präsentation

### Bewertungskriterien (wird erinnert wenn danach gefragt)
- Verständnis des Problems
- Erklärung der Methodik & Entscheidungen
- Interpretation der Ergebnisse
- Fragen beantworten können

---

## Datensatz
- **Quelle:** Google Drive
- **Pfad:** `data\raw\` – Originaldatensatz, nie überschreiben
- **Umfang:** 563 Zeilen, 13 Spalten
- **Ziel:** Klassifikationsmodell für `loan_status` (binär: 0 / 1)
