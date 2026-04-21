# Preprocessing Summary – Loan Status Klassifikation
**Projekt:** MSIT Mock Interview  
**Notebook:** 02_preprocessing.ipynb  
**Basis:** EDA Summary – reports/eda_summary.md  

---

## 1. Ausgangslage

| Merkmal | Wert |
|---------|------|
| Zeilen original | 563 |
| Zeilen nach Bereinigung | 508 |
| Entfernte Zeilen | 55 (~9.8%) |
| Features original | 13 |
| Features nach Preprocessing | 15 |
| Zielvariable | loan_status (0/1) |

---

## 2. Entfernte Zeilen

| Grund | Anzahl |
|-------|--------|
| `loan_status` NaN – Zielvariable nie imputieren | 28 |
| `loan_id` NaN – ID ohne Wert nicht verwendbar | 27 |
| **Gesamt** | **55** |

---

## 3. Feature Engineering

| Feature | Formel | Begründung |
|---------|--------|-----------|
| `has_coapplicant` | 1 wenn coapplicant_income > 0 | Strukturproblem coapplicant_income – viele Nullwerte |
| `is_standard_term` | 1 wenn loan_amount_term == 360 | IQR=0 – kaum Varianz in loan_amount_term |
| `total_income` | applicant_income + coapplicant_income | Gesamteinkommen als Interaktionsterm |

### Verteilung neue Features
| Feature | Klasse 0 | Klasse 1 | Gesamt |
|---------|----------|----------|--------|
| `has_coapplicant` = 1 | 204 (40%) | 304 (60%) | 508 |
| `is_standard_term` = 1 | 440 (87%) | 68 (13%) | 508 |

⚠️ **Reihenfolge kritisch:**
- `has_coapplicant` und `is_standard_term` → vor Imputation (echte Nullwerte)
- `total_income` → nach Imputation (sonst NaN-Propagation)

---

## 4. Imputation

| Feature | Typ | Strategie | Begründung |
|---------|-----|-----------|-----------|
| `applicant_income` | Numerisch | Median | Robust gegenüber Ausreißern |
| `coapplicant_income` | Numerisch | Median | Robust gegenüber Ausreißern |
| `loan_amount` | Numerisch | Median | Robust gegenüber Ausreißern |
| `loan_amount_term` | Numerisch | Median | Robust gegenüber Ausreißern |
| `credit_history` | Binär | Modus | Binäres Feature |
| `gender` | Kategorisch | Modus | MCAR bestätigt |
| `married` | Kategorisch | Modus | MCAR bestätigt |
| `dependents` | Kategorisch | Modus | MCAR bestätigt |
| `education` | Kategorisch | Modus | MCAR bestätigt |
| `self_employed` | Kategorisch | Modus | MCAR bestätigt |
| `property_area` | Kategorisch | Modus | MCAR bestätigt |

**Basis:** MCAR in EDA bestätigt – Imputation statistisch vertretbar.

---

## 5. Encoding

| Feature | Methode | Mapping |
|---------|---------|---------|
| `gender` | Label Encoding | Female=0, Male=1 |
| `married` | Label Encoding | No=0, Yes=1 |
| `education` | Label Encoding | Graduate=0, Not Graduate=1 |
| `self_employed` | Label Encoding | No=0, Yes=1 |
| `dependents` | Ordinales Encoding | 0→0, 1→1, 2→2, 3+→3 |
| `property_area` | One-Hot Encoding (drop first) | Rural=Referenz, Semiurban, Urban |

**Dummy-Variable-Falle:** `drop="first"` → Rural als Referenzkategorie.

---

## 6. Yeo-Johnson Transformation

| Feature | Skewness Vorher | Skewness Nachher | Verbesserung |
|---------|----------------|-----------------|-------------|
| `coapplicant_income` | 7.48 | -0.26 | ✅ 7.22 |
| `total_income` | 5.72 | -0.04 | ✅ 5.68 |
| `applicant_income` | 6.78 | -0.13 | ✅ 6.65 |
| `loan_amount` | 2.60 | 0.03 | ✅ 2.57 |

**Warum Yeo-Johnson statt Log:**
- Log-Transformation funktioniert nicht bei Nullwerten (`coapplicant_income` = 0)
- Yeo-Johnson unterstützt negative Werte und Nullwerte
- Optimal durch Lambda-Schätzung – kein pauschaler Ansatz

⚠️ **`coapplicant_income` nach Transformation:**
Bimodale Struktur verbleibt – strukturelles Problem (mit/ohne Co-Applicant).
`has_coapplicant` Binary Feature fängt diese Information auf.

---

## 7. Entfernte Features

| Feature | Grund |
|---------|-------|
| `loan_id` | ID – kein Prädiktor |
| `loan_amount_term` | Ersetzt durch `is_standard_term` – MI=0.00 |

---

## 8. Train/Test Split

| Merkmal | Wert |
|---------|------|
| Split | 80/20 |
| Stratified | Ja |
| random_state | 42 |
| X_train | (406, 15) |
| X_test | (102, 15) |
| Klasse 1 Train | 71.43% |
| Klasse 1 Test | 71.57% |

---

## 9. Finale Feature-Liste

| # | Feature | Typ | Ursprung |
|---|---------|-----|---------|
| 1 | `gender` | int | Label Encoded |
| 2 | `married` | int | Label Encoded |
| 3 | `dependents` | int | Ordinal Encoded |
| 4 | `education` | int | Label Encoded |
| 5 | `self_employed` | int | Label Encoded |
| 6 | `applicant_income` | float | Yeo-Johnson |
| 7 | `coapplicant_income` | float | Yeo-Johnson |
| 8 | `loan_amount` | float | Yeo-Johnson |
| 9 | `credit_history` | int | Original (binär) |
| 10 | `has_coapplicant` | int | Feature Engineering |
| 11 | `is_standard_term` | int | Feature Engineering |
| 12 | `total_income` | float | Feature Engineering + Yeo-Johnson |
| 13 | `property_area_Semiurban` | float | One-Hot Encoded |
| 14 | `property_area_Urban` | float | One-Hot Encoded |

⚠️ **`loan_amount_term` entfernt** – ersetzt durch `is_standard_term`

---

## 10. Skalierung – Hinweis für Modeling

Skalierung wird **nicht** im Preprocessing durchgeführt –
sondern modellspezifisch in `03_modeling.ipynb`:

| Modell | Skalierung |
|--------|-----------|
| Logistic Regression | StandardScaler |
| SVM | StandardScaler |
| Random Forest | Keine |
| XGBoost | Keine |

---

## 11. Gespeicherte Dateien

| Datei | Inhalt |
|-------|--------|
| `data/processed/loans_processed.csv` | Vollständiger processed Datensatz |
| `data/processed/X_train.csv` | Training Features |
| `data/processed/X_test.csv` | Test Features |
| `data/processed/y_train.csv` | Training Labels |
| `data/processed/y_test.csv` | Test Labels |

---

## 12. Theoretische Grenzen

| Grenze | Detail |
|--------|--------|
| Modus-Imputation | Erhöht Dominanz der häufigsten Kategorie |
| Label Encoding | Impliziert Ordnung bei nominalen Features |
| Kleine Stichprobe | 508 Zeilen – Modell wird schnell overfitten |
| Yeo-Johnson coapplicant | Bimodale Struktur nicht vollständig auflösbar |
| One-Hot drop first | Rural als Referenz – Interpretation beachten |

---

*Erstellt: 02_preprocessing.ipynb – MSIT Mock Interview | 2026*
