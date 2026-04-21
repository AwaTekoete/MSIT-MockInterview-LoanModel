# EDA Summary – Loan Status Klassifikation
**Projekt:** MSIT Mock Interview
**Notebook:** 01_eda.ipynb
**Datensatz:** loans_modified.csv – 563 Zeilen, 13 Spalten
**Zielvariable:** loan_status (binär: 0 = Abgelehnt, 1 = Genehmigt)

---

## 1. Datensatz-Übersicht

| Merkmal | Wert |
|---------|------|
| Zeilen gesamt | 563 |
| Spalten gesamt | 13 |
| Numerische Features | 5 |
| Kategorische Features | 6 |
| ID-Spalte | loan_id |
| Zielvariable | loan_status |

---

## 2. Fehlende Werte

### Befund
- **Alle 13 Spalten** haben fehlende Werte (3.4% – 6.0%)
- Kein Feature überschreitet 10% → kein Feature wird entfernt
- **MCAR bestätigt** – Missing Indikator Korrelationen alle nahe 0
- Fehlende Werte entstehen zufällig – Imputation statistisch vertretbar

### Strategie
| Feature | Typ | Strategie |
|---------|-----|-----------|
| `loan_status` | Zielvariable | Zeilen entfernen |
| `loan_id` | ID | Zeilen entfernen |
| `applicant_income`, `coapplicant_income`, `loan_amount`, `loan_amount_term` | Numerisch | Median |
| `gender`, `married`, `dependents`, `education`, `self_employed`, `property_area` | Kategorisch | Modus |
| `credit_history` | Binär | Modus |

---

## 3. Zielvariable – Klassenungleichgewicht

| Klasse | N | % |
|--------|---|---|
| 1 – Genehmigt | 384 | 71.78% |
| 0 – Abgelehnt | 151 | 28.22% |

**Klassenungleichgewicht: 2.54:1**

### Konsequenzen
- `Accuracy` als Metrik ungeeignet – Baseline 71.78% ohne Lernen erreichbar
- Metriken: F1-Score, ROC-AUC, Precision-Recall AUC, Matthews Correlation Coefficient
- Modellseitig: `class_weight='balanced'` oder SMOTE evaluieren

---

## 4. Numerische Features – Univariate Analyse

### Verteilung & Transformation
| Feature | Skewness | Kurtosis | Normalverteilt? | Transformation |
|---------|----------|----------|-----------------|---------------|
| `applicant_income` | 6.74 | 65.34 | Nein | Yeo-Johnson ✅ |
| `coapplicant_income` | 7.30 | 78.20 | Nein | Yeo-Johnson + Binary Feature ✅ |
| `loan_amount` | 2.43 | 8.36 | Nein | Yeo-Johnson ✅ |
| `loan_amount_term` | -2.43 | 6.82 | Nein | Binary Feature (is_standard_term) |
| `credit_history` | -2.32 | 3.38 | Nein | Keine – bleibt binär |

### Ausreißer (IQR-Methode)
| Feature | N Outliers | % |
|---------|-----------|---|
| `applicant_income` | 43 | 8.0% |
| `loan_amount` | 29 | 5.4% |
| `coapplicant_income` | 17 | 3.2% |
| `loan_amount_term` | übersprungen (IQR=0) | – |
| `credit_history` | übersprungen (IQR=0) | – |

**Strategie:** Transformation statt Entfernung – reale Werte, kein Datenverlust

---

## 5. Kategorische Features – Univariate Analyse

### Zusammenhang mit loan_status (Cramér's V + Chi²)
| Feature | Cramér's V | Chi² p | Assoziation | Unabhängig? |
|---------|-----------|--------|-------------|-------------|
| `property_area` | 0.157 | 0.002 | Moderat | Nein ✅ |
| `married` | 0.090 | 0.041 | Schwach | Nein ✅ |
| `dependents` | 0.071 | 0.473 | Schwach | Ja |
| `self_employed` | 0.035 | 0.435 | Schwach | Ja |
| `education` | 0.034 | 0.443 | Schwach | Ja |
| `gender` | 0.008 | 0.853 | Keine | Ja |

### Genehmigungsrate pro Gruppe
| Feature | Stärkste Gruppe | Rate | Schwächste Gruppe | Rate | Δ |
|---------|----------------|------|-------------------|------|---|
| `property_area` | Semiurban | 81.4% | Rural | 65.3% | 16.1% |
| `married` | Yes | 75.3% | No | 66.5% | 8.8% |
| `dependents` | 2 | 78.1% | 1 | 67.8% | 10.3% |
| `education` | Graduate | 72.1% | Not Graduate | 67.9% | 4.3% |
| `self_employed` | No | 72.8% | Yes | 67.2% | 5.6% |
| `gender` | Male | 72.3% | Female | 70.7% | 1.6% |

---

## 6. Bivariate Analyse – Numerisch vs. loan_status

### Pearson / Point-Biserial / Mutual Information
| Feature | Pearson r | MI | Signifikant? |
|---------|-----------|-----|-------------|
| `credit_history` | 0.45 | 0.09 | Ja ✅ |
| `coapplicant_income` | -0.10 | 0.00 | Ja (knapp) |
| `loan_amount` | -0.08 | 0.00 | Nein |
| `applicant_income` | -0.05 | 0.03 | Nein |
| `loan_amount_term` | 0.04 | 0.00 | Nein |

**Befund:** `credit_history` einziger starker numerischer Prädiktor.
Alle anderen numerischen Features zeigen schwache lineare und nicht-lineare Zusammenhänge.

---

## 7. Kategorische Multikollinearität (Cramér's V Matrix)

| Paar | V | Bewertung |
|------|---|-----------|
| `married` ↔ `dependents` | 0.382 | ⚠️ Stark |
| `gender` ↔ `married` | 0.364 | ⚠️ Stark |
| `gender` ↔ `dependents` | 0.175 | Moderat |

**Konsequenz:** Features vorerst behalten – Feature Importance entscheidet final.

---

## 8. Kruskal-Wallis – Numerisch vs. Kategorisch

**Einziger mittlerer Effekt:**
- `married` ↔ `coapplicant_income` (η²=0.06) – Verheiratete haben höheres Co-Einkommen

**Alle anderen Effekte:** Klein bis minimal – kategorische Features erklären numerische kaum.

**`credit_history` vollständig unabhängig** von allen kategorischen Features (p > 0.59).

---

## 9. Gesamtbefund & Feature-Ranking

| Rang | Feature | Typ | Stärke | Basis |
|------|---------|-----|--------|-------|
| 1 | `credit_history` | Numerisch/Binär | ⭐⭐⭐⭐⭐ | Pearson 0.45, MI 0.09, Genehmigungsrate Δ67% |
| 2 | `property_area` | Kategorisch | ⭐⭐⭐ | Cramér's V 0.157, Δ16.1% |
| 3 | `married` | Kategorisch | ⭐⭐ | Cramér's V 0.090, Δ8.8% |
| 4 | `dependents` | Kategorisch | ⭐⭐ | Genehmigungsrate Δ10.3% |
| 5 | `applicant_income` | Numerisch | ⭐ | MI 0.03, schwach nicht-linear |
| 6 | `coapplicant_income` | Numerisch | ⭐ | Pearson -0.10 |
| 7 | `education` | Kategorisch | ⭐ | Cramér's V 0.034 |
| 8 | `loan_amount` | Numerisch | ⭐ | Pearson -0.08 |
| 9 | `self_employed` | Kategorisch | ⭐ | Cramér's V 0.035 |
| 10 | `loan_amount_term` | Numerisch | ✗ | MI 0.00, Pearson 0.04 |
| 11 | `gender` | Kategorisch | ✗ | Cramér's V 0.008, Δ1.6% |

---

## 10. Entscheidungen für 02_preprocessing.ipynb

### Fehlende Werte
- `loan_status` NaN → Zeilen entfernen
- `loan_id` NaN → Zeilen entfernen
- Numerische Features → Median Imputation
- Kategorische Features → Modus Imputation

### Transformationen
- `applicant_income` → Yeo-Johnson
- `coapplicant_income` → Yeo-Johnson + neues Feature `has_coapplicant` (0/1)
- `loan_amount` → Yeo-Johnson
- `loan_amount_term` → neues Feature `is_standard_term` (1 wenn 360, sonst 0)
- `credit_history` → keine Transformation

### Feature Engineering
- `has_coapplicant`: 1 wenn coapplicant_income > 0, sonst 0
- `is_standard_term`: 1 wenn loan_amount_term == 360, sonst 0
- `total_income`: applicant_income + coapplicant_income (vor Transformation)

### Encoding
- Binäre Features (`gender`, `married`, `education`, `self_employed`) → Label Encoding
- `dependents` → Ordinales Encoding (0, 1, 2, 3+ → 0, 1, 2, 3)
- `property_area` → One-Hot Encoding (3 Kategorien)

### Feature Removal Kandidaten
- `loan_id` → ID, kein Prädiktor
- `gender` → Cramér's V 0.008, statistisch irrelevant (finale Entscheidung nach Feature Importance)
- `loan_amount_term` → MI 0.00, wird durch `is_standard_term` ersetzt

### Skalierung
- StandardScaler für lineare Modelle (Logistic Regression, SVM)
- Baumbasierte Modelle (Random Forest, XGBoost) → keine Skalierung nötig

---

## 11. Theoretische Grenzen dieser EDA

| Grenze | Detail |
|--------|--------|
| Pearson nur linear | Nicht-lineare Zusammenhänge unsichtbar |
| Cramér's V nur bivariat | Interaktionen zwischen 3+ Features nicht erfasst |
| Kleine Stichprobe (n=563) | Statistische Tests instabil bei seltenen Kategorien |
| Klassenungleichgewicht | Alle Metriken leicht verzerrt Richtung Mehrheitsklasse |
| Kruskal-Wallis kein Post-hoc | Paarweise Gruppenunterschiede nicht identifiziert |

---

*Erstellt: 01_eda.ipynb – MSIT Mock Interview | 2026*
