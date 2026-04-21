# =============================================================================
# app/main.py – Streamlit Web-App
# Projekt: MSIT Mock Interview – Loan Status Klassifikation
# Stil: Store44 (Dunkel, Gold, Dunkelgrau)
# Tabs: Dashboard | Einzelantrag
# Imports: ausschliesslich aus src/
# =============================================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_model, load_test_data
from src.preprocessing import preprocess_single
from src.evaluate import (
    predict_single,
    compute_test_metrics,
    compute_shap_single,
    get_threshold_analysis
)

# =============================================================================
# SEITEN-KONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Loan Status Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STORE44 CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Hintergrund */
    .stApp {
        background-color: #1C1C1C;
        color: #FFFFFF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2A2A2A;
        border-right: 1px solid #F5A623;
    }

    /* Haupttitel */
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #FFFFFF;
        border-left: 6px solid #F5A623;
        padding-left: 20px;
        margin-bottom: 0.2rem;
    }

    /* Goldene Trennlinie */
    .gold-line {
        height: 3px;
        background: #F5A623;
        margin: 0.5rem 0 1.5rem 0;
        border: none;
    }

    /* KPI Karten */
    .kpi-card {
        background-color: #2A2A2A;
        border-top: 3px solid #F5A623;
        border-radius: 4px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 900;
        color: #F5A623;
        font-family: 'Georgia', serif;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #AAAAAA;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Section Header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F5A623;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
    }

    /* Entscheidungs-Box Genehmigt */
    .decision-approved {
        background-color: #4A5C3A;
        border-top: 4px solid #F5A623;
        border-radius: 4px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Entscheidungs-Box Abgelehnt */
    .decision-rejected {
        background-color: #5C3A3A;
        border-top: 4px solid #E44C6C;
        border-radius: 4px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Decision Text */
    .decision-text {
        font-size: 2rem;
        font-weight: 900;
        color: #FFFFFF;
        font-family: 'Georgia', serif;
    }

    .confidence-text {
        font-size: 1rem;
        color: #AAAAAA;
        margin-top: 0.4rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2A2A2A;
        border-bottom: 2px solid #F5A623;
    }
    .stTabs [data-baseweb="tab"] {
        color: #AAAAAA;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: #F5A623 !important;
        border-bottom: 3px solid #F5A623;
    }

    /* Input Labels */
    label {
        color: #AAAAAA !important;
        font-size: 0.85rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #F5A623;
        color: #1C1C1C;
        font-weight: 800;
        border: none;
        border-radius: 2px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        letter-spacing: 0.05em;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #D4891F;
        color: #1C1C1C;
    }

    /* Metriken */
    [data-testid="stMetricValue"] {
        color: #F5A623;
        font-weight: 900;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555555;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2A2A2A;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATEN & MODELL LADEN (gecacht)
# =============================================================================
@st.cache_resource
def load_resources():
    model = load_model()
    X_train, X_test, y_train, y_test = load_test_data()
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = load_resources()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">LOAN STATUS CLASSIFIER</div>',
            unsafe_allow_html=True)
st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#AAAAAA; font-size:0.9rem; margin-bottom:2rem;">'
    'MSIT Mock Interview &nbsp;|&nbsp; LightGBM Champion v1 &nbsp;|&nbsp; '
    'Precision: 0.867 &nbsp;|&nbsp; PR-AUC: 0.915</p>',
    unsafe_allow_html=True
)

# =============================================================================
# SIDEBAR – Threshold Slider
# =============================================================================
st.sidebar.markdown(
    '<p style="color:#F5A623; font-weight:700; '
    'text-transform:uppercase; letter-spacing:0.1em;">'
    'Entscheidungsschwellenwert</p>',
    unsafe_allow_html=True
)
threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="0.50 = Standard | 0.41 = Bester MCC | 0.90 = Bank (min. FP)"
)

st.sidebar.markdown(f"""
<div style="background:#2A2A2A; border-top:2px solid #F5A623;
     padding:1rem; border-radius:4px; margin-top:1rem;">
    <p style="color:#AAAAAA; font-size:0.8rem; margin:0;">
    <b style="color:#F5A623;">0.41</b> → Bester MCC<br>
    <b style="color:#F5A623;">0.50</b> → Standard<br>
    <b style="color:#F5A623;">0.90</b> → Bank (min. FP)
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color:#555; font-size:0.75rem;">'
    'MSIT Mock Interview 2026<br>'
    'Champion: LGBM_Full v1</p>',
    unsafe_allow_html=True
)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["📊  DASHBOARD", "🏦  EINZELANTRAG"])

# ===========================================================================
# TAB 1 – DASHBOARD
# ===========================================================================
with tab1:

    # --- Metriken berechnen ---
    metrics = compute_test_metrics(model, X_test, y_test,
                                   threshold=threshold)

    # --- KPI Karten ---
    st.markdown('<p class="section-header">Modellperformance – Test-Set</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Precision",  metrics["precision"]),
        (c2, "Recall",     metrics["recall"]),
        (c3, "F1-Score",   metrics["f1"]),
        (c4, "PR-AUC",     metrics["pr_auc"]),
        (c5, "ROC-AUC",    metrics["roc_auc"]),
        (c6, "MCC",        metrics["mcc"]),
    ]
    for col, label, value in kpis:
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value:.3f}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)

    # --- Confusion Matrix + Threshold Analyse ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">Confusion Matrix</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#2A2A2A")
        ax.set_facecolor("#2A2A2A")

        cm_data = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ])

        im = ax.imshow(cm_data, cmap="YlOrBr", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Abgelehnt (0)", "Genehmigt (1)"],
                           color="white", fontsize=9)
        ax.set_yticklabels(["Abgelehnt (0)", "Genehmigt (1)"],
                           color="white", fontsize=9)
        ax.set_xlabel("Vorhergesagt", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Tatsächlich", color="#AAAAAA", fontsize=9)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_data[i, j]),
                        ha="center", va="center",
                        color="white", fontsize=16, fontweight="bold")

        ax.set_title(f"Threshold: {threshold:.2f} | "
                     f"FP: {metrics['fp']}",
                     color="#F5A623", fontsize=10, pad=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#F5A623")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown('<p class="section-header">Threshold Analyse</p>',
                    unsafe_allow_html=True)

        df_thr = get_threshold_analysis(model, X_test, y_test)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#2A2A2A")
        ax2.set_facecolor("#2A2A2A")

        ax2.plot(df_thr["Threshold"], df_thr["Precision"],
                 color="#F5A623", linewidth=2, label="Precision")
        ax2.plot(df_thr["Threshold"], df_thr["Recall"],
                 color="#6CB4E4", linewidth=2, label="Recall")
        ax2.plot(df_thr["Threshold"], df_thr["F1"],
                 color="#2ECC71", linewidth=2, label="F1")
        ax2.axvline(x=threshold, color="white", linestyle="--",
                    linewidth=1.5, label=f"Aktuell ({threshold:.2f})")

        ax2.set_xlabel("Threshold", color="#AAAAAA", fontsize=9)
        ax2.set_ylabel("Metrik-Wert", color="#AAAAAA", fontsize=9)
        ax2.tick_params(colors="white")
        ax2.legend(fontsize=8, facecolor="#1C1C1C",
                   labelcolor="white", edgecolor="#F5A623")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#F5A623")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)

    # --- Feature Importance (SHAP) ---
    st.markdown('<p class="section-header">SHAP Feature Importance</p>',
                unsafe_allow_html=True)

    @st.cache_data
    def get_shap_importance(_model, _X_test):
        explainer   = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(_X_test)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        return pd.DataFrame({
            "Feature":    _X_test.columns,
            "Importance": mean_abs
        }).sort_values("Importance", ascending=True)

    df_shap = get_shap_importance(model, X_test)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor("#2A2A2A")
    ax3.set_facecolor("#2A2A2A")

    colors = ["#F5A623" if i >= len(df_shap) - 3
              else "#6CB4E4"
              for i in range(len(df_shap))]

    ax3.barh(df_shap["Feature"], df_shap["Importance"],
             color=colors, edgecolor="#1C1C1C")
    ax3.set_xlabel("Mean |SHAP Value|", color="#AAAAAA", fontsize=9)
    ax3.tick_params(colors="white", labelsize=9)
    ax3.set_title("Gold = Top 3 Features",
                  color="#AAAAAA", fontsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#2A2A2A")

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # --- Modell Info ---
    st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Modell Information</p>',
                unsafe_allow_html=True)

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown("""
        <div style="background:#2A2A2A; border-top:2px solid #F5A623;
             padding:1rem; border-radius:4px;">
            <p style="color:#F5A623; font-weight:700; margin:0;">MODELL</p>
            <p style="color:#FFF; margin:0.3rem 0 0 0;">LightGBM v1</p>
            <p style="color:#AAA; font-size:0.8rem;">680 Bäume | max_depth=14</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col2:
        st.markdown("""
        <div style="background:#2A2A2A; border-top:2px solid #F5A623;
             padding:1rem; border-radius:4px;">
            <p style="color:#F5A623; font-weight:700; margin:0;">TUNING</p>
            <p style="color:#FFF; margin:0.3rem 0 0 0;">Optuna</p>
            <p style="color:#AAA; font-size:0.8rem;">100 Trials | TPE Sampler</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col3:
        st.markdown("""
        <div style="background:#2A2A2A; border-top:2px solid #F5A623;
             padding:1rem; border-radius:4px;">
            <p style="color:#F5A623; font-weight:700; margin:0;">DATENSATZ</p>
            <p style="color:#FFF; margin:0.3rem 0 0 0;">508 Samples</p>
            <p style="color:#AAA; font-size:0.8rem;">15 Features | 80/20 Split</p>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# TAB 2 – EINZELANTRAG
# ===========================================================================
with tab2:

    st.markdown('<p class="section-header">Kreditantrag Eingabe</p>',
                unsafe_allow_html=True)

    # --- Formular ---
    with st.form("loan_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Persönliche Daten**")
            gender = st.selectbox(
                "Geschlecht", ["Male", "Female"])
            married = st.selectbox(
                "Familienstand", ["Yes", "No"])
            dependents = st.selectbox(
                "Unterhaltsberechtigte", ["0", "1", "2", "3+"])
            education = st.selectbox(
                "Bildungsstand", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox(
                "Selbstständig", ["No", "Yes"])

        with col2:
            st.markdown("**Finanzdaten**")
            applicant_income = st.number_input(
                "Einkommen Antragsteller",
                min_value=0, max_value=200000,
                value=5000, step=100
            )
            coapplicant_income = st.number_input(
                "Einkommen Co-Antragsteller",
                min_value=0, max_value=100000,
                value=0, step=100
            )
            loan_amount = st.number_input(
                "Kreditbetrag (in Tausend)",
                min_value=1, max_value=1000,
                value=150, step=10
            )
            loan_amount_term = st.selectbox(
                "Laufzeit (Monate)",
                [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                index=8
            )

        with col3:
            st.markdown("**Weitere Angaben**")
            credit_history = st.selectbox(
                "Kredithistorie erfüllt",
                [1, 0],
                format_func=lambda x: "Ja (1)" if x == 1 else "Nein (0)"
            )
            property_area = st.selectbox(
                "Lage der Immobilie",
                ["Semiurban", "Urban", "Rural"]
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("KREDITENTSCHEIDUNG TREFFEN")

    # --- Prediction ---
    if submitted:
        input_dict = {
            "gender":            gender,
            "married":           married,
            "dependents":        dependents,
            "education":         education,
            "self_employed":     self_employed,
            "applicant_income":  applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount":       loan_amount,
            "loan_amount_term":  loan_amount_term,
            "credit_history":    credit_history,
            "property_area":     property_area,
        }

        # Preprocessing
        X_single = preprocess_single(input_dict)

        # Prediction
        decision, confidence = predict_single(model, X_single,
                                              threshold=threshold)

        st.markdown('<div class="gold-line"></div>',
                    unsafe_allow_html=True)
        st.markdown('<p class="section-header">Entscheidung</p>',
                    unsafe_allow_html=True)

        # --- Entscheidungs-Box ---
        if decision == 1:
            st.markdown(f"""
            <div class="decision-approved">
                <div class="decision-text">✅ GENEHMIGT</div>
                <div class="confidence-text">
                    Konfidenz: {confidence:.1%} &nbsp;|&nbsp;
                    Threshold: {threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-rejected">
                <div class="decision-text">❌ ABGELEHNT</div>
                <div class="confidence-text">
                    Konfidenz: {1-confidence:.1%} &nbsp;|&nbsp;
                    Threshold: {threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- SHAP Waterfall ---
        st.markdown('<p class="section-header">SHAP Erklärung</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#AAAAAA; font-size:0.85rem;">'
            'Welche Features haben diese Entscheidung beeinflusst?</p>',
            unsafe_allow_html=True
        )

        explanation = compute_shap_single(model, X_single)

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        fig_shap.patch.set_facecolor("#2A2A2A")

        shap.waterfall_plot(explanation, show=False, max_display=12)

        fig_shap = plt.gcf()
        fig_shap.patch.set_facecolor("#2A2A2A")
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

        # --- Eingabe Zusammenfassung ---
        st.markdown('<div class="gold-line"></div>',
                    unsafe_allow_html=True)
        st.markdown('<p class="section-header">Eingabe Zusammenfassung</p>',
                    unsafe_allow_html=True)

        summary_df = pd.DataFrame([{
            "Einkommen": f"{applicant_income:,}",
            "Co-Einkommen": f"{coapplicant_income:,}",
            "Kreditbetrag": f"{loan_amount}k",
            "Laufzeit": f"{loan_amount_term} Monate",
            "Kredithistorie": "Ja" if credit_history == 1 else "Nein",
            "Immobilienlage": property_area,
            "Familienstand": married,
            "Bildung": education,
        }]).T.reset_index()
        summary_df.columns = ["Merkmal", "Wert"]
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True
        )

# --- Footer ---
st.markdown(
    '<div class="footer">'
    'MSIT Mock Interview 2026 &nbsp;|&nbsp; '
    'LightGBM Champion v1 &nbsp;|&nbsp; '
    'Erik Gerst'
    '</div>',
    unsafe_allow_html=True
)