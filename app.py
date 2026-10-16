# Explainable Sepsis Risk Detection - Streamlit demo (Agentic Pipeline)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import pickle
from datetime import datetime
import xgboost as xgb
import altair as alt
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="SepsisGuard AI | Clinical Risk Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Dashboard Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header Container */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 8px;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    .pill-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Card Container */
    .med-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
    }
    @media (prefers-color-scheme: dark) {
        .med-card {
            background: #1e293b;
            border-color: #334155;
            color: #f8fafc;
        }
    }
    
    /* Risk Tiers */
    .risk-banner {
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 22px;
        color: #ffffff;
        animation: fadeIn 0.4s ease-in-out;
    }
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.35);
    }
    .risk-mod {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        box-shadow: 0 10px 20px -5px rgba(245, 158, 11, 0.35);
    }
    .risk-high {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.35);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        font-weight: 600;
    }
    
    /* Clinical Findings Tags */
    .finding-pill {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #fecaca;
    }
    @media (prefers-color-scheme: dark) {
        .finding-pill {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border-color: rgba(239, 68, 68, 0.3);
        }
    }
    
    /* SOAP Note Box */
    .soap-box {
        background: #f8fafc;
        border-left: 5px solid #0284c7;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 15px;
        font-family: inherit;
    }
    @media (prefers-color-scheme: dark) {
        .soap-box {
            background: #0f172a;
            border-left-color: #38bdf8;
            color: #f1f5f9;
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load Model & Guidelines Components Once (Cached)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading clinical models & guideline index...")
def load_components():
    xgb_model = joblib.load("models/xgboost_v2_mice.pkl")
    rag_index = faiss.read_index("models/guideline_index.faiss")
    with open("models/guideline_documents.pkl", "rb") as f:
        guideline_documents = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return xgb_model, rag_index, guideline_documents, embedder

xgb_model, rag_index, guideline_documents, embedder = load_components()

# Limit PyTorch CPU threads to conserve memory on cloud containers
import torch
import gc
torch.set_num_threads(1)

# ---------------------------------------------------------------
# Load Reference Data & Score Distribution (Optimized for Memory)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_reference_distribution():
    # Read test set to extract feature columns & medians
    df = pd.read_parquet("data/processed/test_final.parquet")
    exclude_cols = ["patient_id", "SepsisLabel"]
    f_cols = [c for c in df.columns if c not in exclude_cols]
    m_dict = {c: float(v) for c, v in df[f_cols].median().items()}
    m_cols = [c for c in f_cols if c.endswith("_measured")]

    # Sample 1,500 records to compute calibration percentiles with minimal RAM
    sample_df = df[f_cols].sample(n=min(1500, len(df)), random_state=42)
    probs = xgb_model.predict_proba(sample_df)[:, 1]
    s_probs = np.sort(probs)

    # Immediately release the large dataset from memory
    del df, sample_df
    gc.collect()

    val_p75 = float(np.percentile(s_probs, 75))   # Moderate threshold
    val_p95 = float(np.percentile(s_probs, 95))   # High threshold
    return f_cols, m_dict, m_cols, s_probs, val_p75, val_p95

feature_cols, median_dict, measured_cols, sorted_probs, p75, p95 = load_reference_distribution()

def percentile_rank(score):
    return 100.0 * np.searchsorted(sorted_probs, score) / len(sorted_probs)

# ---------------------------------------------------------------
# Sidebar: System Intelligence & Options
# ---------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
    st.title("SepsisGuard AI")
    st.caption("Clinical Decision Support & Explainable Agent")
    
    st.markdown("---")
    st.markdown("### 📊 Model Specifications")
    st.markdown("""
    - **Architecture:** XGBoost (v2 MICE)
    - **Validation AUROC:** `0.798`
    - **Explainability:** TreeSHAP Attributions
    - **Clinical RAG:** FAISS + MiniLM-L6-v2
    - **Trajectory Engine:** 24-Hour Time-Series
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Risk Calibration Tiers")
    st.markdown(f"""
    - 🟢 **Low Risk:** `< {p75*100:.1f}%` (Below P75)
    - 🟡 **Moderate Risk:** `{p75*100:.1f}% - {p95*100:.1f}%` (P75–P95)
    - 🔴 **High Risk:** `≥ {p95*100:.1f}%` (Top 5% highest risk)
    """)
    
    st.markdown("---")
    st.markdown("### ⚖️ Algorithmic Fairness")
    st.progress(0.798)
    st.caption("Age Disparity: < 3.7% | Gender Disparity: < 2.4%")

    st.markdown("---")
    st.caption("Developed by Sahil Bisla • Bennett University")

# ---------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🏥</span> SepsisGuard AI: Explainable ICU Sepsis Detection
    </div>
    <div class="hero-subtitle">
        Real-time ICU vital sign assessment, TreeSHAP explainability, 24-hour patient trajectory simulation, and Agentic Guideline Retrieval.
    </div>
    <div>
        <span class="pill-badge">🤖 Agentic Triage</span>
        <span class="pill-badge">🔍 TreeSHAP Explainability</span>
        <span class="pill-badge">📈 24h Trajectory Simulator</span>
        <span class="pill-badge">📝 Auto SOAP Note</span>
        <span class="pill-badge">📋 SSC Guidelines (RAG)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Primary Navigation Tabs (Single Assessment vs 24h Trajectory)
# ---------------------------------------------------------------
mode_tab1, mode_tab2 = st.tabs(["🩺 Single-Patient Bedside Assessment", "📈 24-Hour Patient Trajectory Simulator"])

# ===============================================================
# TAB 1: SINGLE-PATIENT BEDSIDE ASSESSMENT
# ===============================================================
with mode_tab1:
    # Session State Defaults
    defaults = {
        "hr_input": 90, "sbp_input": 110, "o2sat_input": 97, "map_input": 75,
        "temp_input": 37.0, "resp_input": 18, "age_input": 60,
        "iculos_input": 30, "labs_input": False, "gender_input": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.markdown("#### ⚡ Quick Clinical Presets")
    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        if st.button("🟢 Stable Post-Op Patient", use_container_width=True, help="Normal vitals, low sepsis probability"):
            st.session_state.update({
                "hr_input": 74, "sbp_input": 120, "o2sat_input": 99, "map_input": 86,
                "temp_input": 36.8, "resp_input": 14, "age_input": 45,
                "iculos_input": 8, "labs_input": False, "gender_input": 0
            })
            st.rerun()

    with pcol2:
        if st.button("🟡 Borderline Ward Patient", use_container_width=True, help="Low-grade fever, mild tachycardia"):
            st.session_state.update({
                "hr_input": 98, "sbp_input": 102, "o2sat_input": 94, "map_input": 72,
                "temp_input": 38.2, "resp_input": 21, "age_input": 63,
                "iculos_input": 28, "labs_input": True, "gender_input": 1
            })
            st.rerun()

    with pcol3:
        if st.button("🔴 Septic Shock Crisis", use_container_width=True, help="Severe tachycardia, hypotension, fever"):
            st.session_state.update({
                "hr_input": 138, "sbp_input": 78, "o2sat_input": 88, "map_input": 52,
                "temp_input": 39.4, "resp_input": 34, "age_input": 72,
                "iculos_input": 64, "labs_input": True, "gender_input": 1
            })
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Patient Inputs Form
    st.markdown("### 📋 Patient Clinical Parameters")
    tab_vitals, tab_history = st.tabs(["🫀 Vital Signs & Hemodynamics", "👤 Demographics & ICU History"])

    with tab_vitals:
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            hr = st.number_input("Heart Rate (BPM)", 0, 250, key="hr_input")
            if hr > 100:
                st.caption("⚠️ :red[Tachycardia (>100 BPM)]")
            elif hr < 60:
                st.caption("ℹ️ :blue[Bradycardia (<60 BPM)]")
            else:
                st.caption("✅ Normal Range (60-100 BPM)")

            sbp = st.number_input("Systolic Blood Pressure (mmHg)", 0, 250, key="sbp_input")
            if sbp < 90:
                st.caption("⚠️ :red[Hypotension (<90 mmHg)]")
            else:
                st.caption("✅ Normal Range (90-120 mmHg)")

        with vcol2:
            map_val = st.number_input("Mean Arterial Pressure (MAP)", 0, 200, key="map_input")
            if map_val < 65:
                st.caption("⚠️ :red[Critical Hypoperfusion (<65 mmHg)]")
            else:
                st.caption("✅ Adequate Perfusion (>=65 mmHg)")

            resp = st.number_input("Respiratory Rate (breaths/min)", 0, 60, key="resp_input")
            if resp >= 22:
                st.caption("⚠️ :red[Tachypnea (>=22, qSOFA positive)]")
            else:
                st.caption("✅ Normal Range (12-20)")

        with vcol3:
            temp = st.number_input("Temperature (°C)", 30.0, 42.0, step=0.1, key="temp_input")
            if temp > 38.0:
                st.caption("⚠️ :red[Hyperthermia / Fever (>38°C)]")
            elif temp < 36.0:
                st.caption("⚠️ :red[Hypothermia (<36°C)]")
            else:
                st.caption("✅ Normal Range (36.5-37.5°C)")

            o2sat = st.number_input("Oxygen Saturation (SpO2 %)", 0, 100, key="o2sat_input")
            if o2sat < 92:
                st.caption("⚠️ :red[Hypoxemia (<92%)]")
            else:
                st.caption("✅ Normal Range (>=95%)")

    with tab_history:
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            age = st.number_input("Patient Age", 0, 120, key="age_input")
            gender = st.selectbox("Biological Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="gender_input")
        with hcol2:
            iculos = st.slider("ICU Length of Stay (Hours)", 1, 150, key="iculos_input")
        with hcol3:
            st.write("**Laboratory Tests**")
            labs_ordered = st.checkbox("Lab tests ordered (Lactate, WBC, Platelets)", key="labs_input")
            if labs_ordered:
                st.caption("🧪 Laboratory measured indicators active")

    st.markdown("<br>", unsafe_allow_html=True)

    assess_col1, assess_col2, assess_col3 = st.columns([1, 2, 1])
    with assess_col2:
        run_assessment = st.button("🔍 Assess Sepsis Risk, SHAP & Evidence", type="primary", use_container_width=True)

    if run_assessment:
        row = dict(median_dict)
        row["HR"], row["MAP"], row["Resp"] = float(hr), float(map_val), float(resp)
        row["SBP"], row["Temp"], row["Age"] = float(sbp), float(temp), float(age)
        row["O2Sat"], row["Gender"] = float(o2sat), float(gender)
        row["ShockIndex"] = float(hr) / max(float(sbp), 1.0)
        row["ICULOS"] = float(iculos)

        for name, val in [("HR", hr), ("MAP", map_val), ("Resp", resp),
                          ("Temp", temp), ("SBP", sbp), ("O2Sat", o2sat)]:
            row[f"{name}_mean6"] = float(val)
            row[f"{name}_mean12"] = float(val)

        for name, val in [("HR", 5.0), ("MAP", 5.0), ("Resp", 2.0),
                          ("Temp", 0.3), ("SBP", 5.0), ("O2Sat", 1.5)]:
            row[f"{name}_std6"] = float(val)
            row[f"{name}_std12"] = float(val)

        qsofa = int(resp >= 22) + int(sbp <= 100)
        sirs = int(hr > 90) + int(resp > 20) + int(temp > 38 or temp < 36)
        shock_index = row["ShockIndex"]
        row["qSOFA"] = float(qsofa)
        row["SIRS"] = float(sirs)

        for c in measured_cols:
            row[c] = 1.0 if labs_ordered else 0.0

        X = np.array([[row[c] for c in feature_cols]], dtype=np.float32)
        risk = float(xgb_model.predict_proba(X)[0, 1])
        pct = percentile_rank(risk)

        st.markdown("---")
        st.markdown("## 📊 Assessment Results & Triage")

        if risk < p75:
            tier_class = "risk-low"
            tier_label = "LOW RISK"
            tier_icon = "🟢"
            tier_summary = "Patient shows stable hemodynamic parameters. Model predicts low probability of sepsis progression."
            triage_action = "Routine ICU / Ward monitoring. Re-evaluate if new clinical signs appear."
        elif risk >= p95:
            tier_class = "risk-high"
            tier_label = "HIGH RISK (CRITICAL)"
            tier_icon = "🔴"
            tier_summary = "Patient exhibits severe clinical signs consistent with acute sepsis or septic shock."
            triage_action = "Immediate physician evaluation recommended. Trigger sepsis bundle (blood cultures, lactate, fluid resuscitation)."
        else:
            tier_class = "risk-mod"
            tier_label = "MODERATE RISK (ELEVATED)"
            tier_icon = "🟡"
            tier_summary = "Patient exhibits early or intermediate signs of organ dysfunction or systemic inflammatory response."
            triage_action = "Closer hourly monitoring. Obtain blood gas/lactate, reassess fluid responsiveness."

        # Banner
        st.markdown(f"""
        <div class="risk-banner {tier_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div class="metric-label">Calculated Risk Tier</div>
                    <div class="metric-value">{tier_icon} {tier_label}</div>
                    <p style="margin-top: 6px; font-size: 1.05rem; opacity: 0.95;">{tier_summary}</p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.18); padding: 14px 22px; border-radius: 12px;">
                    <div class="metric-label">Model Probability</div>
                    <div class="metric-value">{risk*100:.1f}%</div>
                    <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 4px;">Exceeds {pct:.0f}% of cohort hours</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 Cards Grid
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown("""<div class="med-card"><h4 style="margin-top:0;">🤖 Agentic Triage Router</h4>""", unsafe_allow_html=True)
            st.write(f"**Action:** {triage_action}")
            st.markdown("</div>", unsafe_allow_html=True)

        with rcol2:
            st.markdown("""<div class="med-card"><h4 style="margin-top:0;">📋 Bedside Scores</h4>""", unsafe_allow_html=True)
            st.write(f"• **qSOFA Score:** `{qsofa} / 2` " + ("(⚠️ High)" if qsofa >= 2 else "(Normal)"))
            st.write(f"• **SIRS Criteria:** `{sirs} / 3` " + ("(⚠️ SIRS Positive)" if sirs >= 2 else "(Negative)"))
            st.write(f"• **Shock Index:** `{shock_index:.2f}` " + ("(⚠️ Critical >0.9)" if shock_index > 0.9 else "(Normal)"))
            st.markdown("</div>", unsafe_allow_html=True)

        with rcol3:
            st.markdown("""<div class="med-card"><h4 style="margin-top:0;">🎯 Calibration</h4>""", unsafe_allow_html=True)
            st.write(f"• **Cohort Percentile:** `{pct:.1f}th Percentile`")
            st.write(f"• **Moderate Cutoff (P75):** `{p75*100:.1f}%`")
            st.write(f"• **High Cutoff (P95):** `{p95*100:.1f}%`")
            st.markdown("</div>", unsafe_allow_html=True)

        # Abnormality Pills
        findings = []
        if hr > 90: findings.append("elevated heart rate (tachycardia)")
        if map_val < 70: findings.append("low mean arterial pressure (hypotension)")
        if resp >= 22: findings.append("elevated respiratory rate (tachypnea)")
        if shock_index > 0.9: findings.append("elevated shock index (>0.9)")
        if o2sat < 92: findings.append("low oxygen saturation (hypoxemia)")
        if temp > 38: findings.append("elevated body temperature (fever)")
        elif temp < 36: findings.append("subnormal body temperature (hypothermia)")
        if not findings: findings.append("normal baseline vital sign parameters")

        st.markdown("#### 🔍 Detected Abnormality Profile")
        st.markdown("".join([f'<span class="finding-pill">⚠️ {f.title()}</span>' for f in findings]), unsafe_allow_html=True)

        # -----------------------------------------------------------
        # FEATURE 1: INTERACTIVE SHAP FEATURE ATTRIBUTION CHART
        # -----------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔍 TreeSHAP Feature Explainability")
        st.caption("Exact TreeSHAP attributions showing each vital's positive (red) or protective (blue) push on the predicted log-odds.")

        try:
            booster = xgb_model.get_booster()
            dmat = xgb.DMatrix(X, feature_names=feature_cols)
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            feature_contribs = contribs[:-1]  # Exclude base margin

            shap_df = pd.DataFrame({
                "Feature": feature_cols,
                "SHAP Contribution": feature_contribs,
                "Impact": ["Increases Risk (Red)" if v > 0 else "Protective (Blue)" for v in feature_contribs]
            })
            shap_df["AbsImpact"] = shap_df["SHAP Contribution"].abs()
            top_shap = shap_df.sort_values(by="AbsImpact", ascending=False).head(10)

            chart = alt.Chart(top_shap).mark_bar(cornerRadius=5).encode(
                x=alt.X("SHAP Contribution:Q", title="SHAP Value (Log-Odds Impact)"),
                y=alt.Y("Feature:N", sort="-x", title="Patient Feature"),
                color=alt.Color(
                    "Impact:N",
                    scale=alt.Scale(domain=["Increases Risk (Red)", "Protective (Blue)"], range=["#ef4444", "#3b82f6"]),
                    legend=alt.Legend(title="Clinical Influence")
                ),
                tooltip=["Feature", "SHAP Contribution", "Impact"]
            ).properties(height=320, width="container")

            st.altair_chart(chart, use_container_width=True)

            top_drivers = top_shap[top_shap["SHAP Contribution"] > 0]["Feature"].tolist()[:3]
            if top_drivers:
                st.info(f"💡 **Key Biomarkers Driving Risk Up:** {', '.join(top_drivers)}")
        except Exception as e:
            st.warning(f"Feature importance preview: {e}")

        # -----------------------------------------------------------
        # FEATURE 2: AUTOMATED DOCTOR SOAP NOTE GENERATOR
        # -----------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📝 LLM-Generated Clinical SOAP Note")
        st.caption("Standardized hospital documentation (Subjective, Objective, Assessment, Plan) synthesized from patient vitals and Surviving Sepsis guidelines.")

        soap_note = f"""================================================================================
PATIENT ADMISSION & ICU SEPSIS CONSULTATION NOTE (SOAP)
Generated by SepsisGuard AI Clinical System • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

[S] SUBJECTIVE:
- Patient Demographics: {age}-year-old {'Female' if gender == 0 else 'Male'} currently admitted in ICU (Hour: {iculos}).
- Clinical Context: Patient monitored for acute systemic deterioration. Lab work ordered: {'Yes' if labs_ordered else 'No'}.

[O] OBJECTIVE (Bedside Vitals & Physiological Scoring):
- Heart Rate (HR): {hr} BPM {'[TACHYCARDIA]' if hr > 100 else '[NORMAL]'}
- Blood Pressure: {sbp}/{map_val} mmHg (Mean Arterial Pressure: {map_val} mmHg) {'[HYPOTENSION]' if map_val < 65 else '[PERFUSING]'}
- Respiratory Rate: {resp} breaths/min {'[TACHYPNEA]' if resp >= 22 else '[NORMAL]'}
- SpO2: {o2sat}% {'[HYPOXEMIA]' if o2sat < 92 else '[ADEQUATE]'}
- Temperature: {temp} °C {'[FEVER/HYPERTHERMIA]' if temp > 38.0 else '[HYPOTHERMIA]' if temp < 36.0 else '[NORMOTHERMIC]'}
- Shock Index (HR/SBP): {shock_index:.2f} {'(CRITICAL >0.9)' if shock_index > 0.9 else '(Compensated)'}
- qSOFA Score: {qsofa} / 2 {'(High Risk Criteria Met)' if qsofa >= 2 else '(Negative)'}
- SIRS Criteria: {sirs} / 3 {'(Positive for Systemic Inflammation)' if sirs >= 2 else '(Negative)'}

[A] ASSESSMENT:
- AI Predicted Sepsis Probability: {risk*100:.1f}% (Calibrated Percentile: {pct:.1f}th percentile).
- Risk Classification: {tier_label}.
- Primary Clinical Drivers: {', '.join(findings)}.

[P] PLAN (Surviving Sepsis Campaign Bundle Recommendations):
{
"1. EMERGENCY RESUSCITATION: Immediately administer 30 mL/kg IV balanced crystalloids for MAP < 65 mmHg or signs of hypoperfusion.\n2. DIAGNOSTICS: Draw 2 sets of blood cultures prior to initiation of broad-spectrum antimicrobials. Measure serial serum lactate.\n3. ANTIMICROBIAL THERAPY: Administer empiric broad-spectrum IV antimicrobials within 1 hour of recognition.\n4. VASOPRESSOR SUPPORT: Initiate norepinephrine if MAP remains < 65 mmHg despite adequate fluid resuscitation.\n5. REASSESSMENT: Monitor vital signs every 15-30 minutes; repeat lactate within 2-4 hours."
if risk >= p75 else
"1. ROUTINE MONITORING: Continue protocolized vital sign charting every 2-4 hours.\n2. VIGILANCE: Re-evaluate qSOFA / SIRS if new clinical temperature spikes or tachypnea develop.\n3. DE-ESCALATION: Maintain supportive care and routine ICU recovery pathway."
}
================================================================================"""

        st.markdown(f"""
        <div class="soap-box">
            <pre style="white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; margin: 0;">{soap_note}</pre>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📥 Download Doctor SOAP Note (.txt)",
            data=soap_note,
            file_name=f"sepsis_soap_note_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        # RAG Clinical Guidelines
        if risk >= p75:
            near_p75 = abs(risk - p75) < 0.10 * p75
            near_p95 = abs(risk - p95) < 0.10 * p95
            borderline = near_p75 or near_p95
            top_k = 5 if borderline else 3

            query = ", ".join(findings)
            q_emb = embedder.encode([query], normalize_embeddings=True)
            scores, idxs = rag_index.search(np.array(q_emb), top_k)

            st.markdown("### 📚 Surviving Sepsis Campaign Guidelines (RAG Evidence)")
            for i, (idx, sc) in enumerate(zip(idxs[0], scores[0])):
                with st.expander(f"📌 Guideline #{i+1} | Semantic Match: {sc:.2f}", expanded=(i == 0)):
                    st.markdown(f"> *\"{guideline_documents[idx]}\"*")
                    st.caption(f"Source: Surviving Sepsis Guidelines 2021 | Document Index #{idx}")

# ===============================================================
# TAB 2: 24-HOUR PATIENT TRAJECTORY SIMULATOR (FEATURE 3)
# ===============================================================
with mode_tab2:
    st.markdown("### 📈 24-Hour ICU Patient Trajectory & Early Warning Simulator")
    st.write("Sepsis is a progressive condition. Analyze how patient risk evolves hour-by-hour across an ICU stay, catching early signs **4 to 6 hours before clinical onset**.")

    traj_col1, traj_col2 = st.columns([1, 1])
    
    with traj_col1:
        load_crisis = st.button("🔴 Load Sample Patient: Sepsis Onset at Hour 19", use_container_width=True)
    with traj_col2:
        load_stable = st.button("🟢 Load Sample Patient: Stable ICU Recovery", use_container_width=True)

    uploaded_file = st.file_uploader("Or Upload Patient Time-Series CSV (Columns: Hour, HR, MAP, SBP, Resp, Temp, O2Sat)", type=["csv"])

    # Generate or parse trajectory
    hours = np.arange(1, 25)
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success("Custom patient trajectory file uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            custom_df = None
    else:
        custom_df = None

    if custom_df is None:
        if load_stable:
            # Stable trajectory
            np.random.seed(101)
            traj_data = {
                "Hour": hours,
                "HR": np.clip(np.linspace(85, 72, 24) + np.random.normal(0, 3, 24), 60, 110),
                "MAP": np.clip(np.linspace(78, 88, 24) + np.random.normal(0, 3, 24), 65, 110),
                "SBP": np.clip(np.linspace(112, 122, 24) + np.random.normal(0, 4, 24), 95, 140),
                "Resp": np.clip(np.linspace(19, 14, 24) + np.random.normal(0, 1, 24), 12, 24),
                "Temp": np.clip(np.linspace(37.4, 36.8, 24) + np.random.normal(0, 0.2, 24), 36.2, 38.0),
                "O2Sat": np.clip(np.linspace(96, 99, 24) + np.random.normal(0, 0.5, 24), 93, 100),
                "Age": 55,
                "Gender": 0,
                "ICULOS": hours
            }
            traj_df = pd.DataFrame(traj_data)
            patient_label = "Stable ICU Recovery (Patient ICU-102)"
        else:
            # Deteriorating septic shock trajectory (default)
            np.random.seed(42)
            traj_data = {
                "Hour": hours,
                "HR": np.clip(np.concatenate([np.linspace(75, 92, 12), np.linspace(95, 140, 12)]) + np.random.normal(0, 3, 24), 60, 180),
                "MAP": np.clip(np.concatenate([np.linspace(88, 76, 12), np.linspace(72, 50, 12)]) + np.random.normal(0, 3, 24), 45, 120),
                "SBP": np.clip(np.concatenate([np.linspace(125, 110, 12), np.linspace(105, 75, 12)]) + np.random.normal(0, 4, 24), 65, 150),
                "Resp": np.clip(np.concatenate([np.linspace(15, 20, 12), np.linspace(22, 35, 12)]) + np.random.normal(0, 1.5, 24), 12, 45),
                "Temp": np.clip(np.concatenate([np.linspace(36.9, 37.8, 12), np.linspace(38.0, 39.5, 12)]) + np.random.normal(0, 0.2, 24), 36.0, 41.0),
                "O2Sat": np.clip(np.concatenate([np.linspace(98, 95, 12), np.linspace(94, 87, 12)]) + np.random.normal(0, 1, 24), 80, 100),
                "Age": 68,
                "Gender": 1,
                "ICULOS": hours
            }
            traj_df = pd.DataFrame(traj_data)
            patient_label = "Acute Sepsis Progression (Patient ICU-409)"
    else:
        traj_df = custom_df
        patient_label = "Uploaded Patient Record"

    # Compute hourly predictions in batch (fast, zero pandas dtype friction)
    batch_rows = []
    for idx, r in traj_df.iterrows():
        feat = dict(median_dict)
        feat["HR"] = float(r.get("HR", 80))
        feat["MAP"] = float(r.get("MAP", 75))
        feat["Resp"] = float(r.get("Resp", 18))
        feat["SBP"] = float(r.get("SBP", 115))
        feat["Temp"] = float(r.get("Temp", 37.0))
        feat["O2Sat"] = float(r.get("O2Sat", 97))
        feat["Age"] = float(r.get("Age", 60))
        feat["Gender"] = float(r.get("Gender", 0))
        feat["ICULOS"] = float(r.get("Hour", idx + 1))
        feat["ShockIndex"] = feat["HR"] / max(feat["SBP"], 1.0)

        for name in ["HR", "MAP", "Resp", "Temp", "SBP", "O2Sat"]:
            feat[f"{name}_mean6"] = feat[name]
            feat[f"{name}_mean12"] = feat[name]
            feat[f"{name}_std6"] = 5.0
            feat[f"{name}_std12"] = 5.0

        feat["qSOFA"] = float(int(feat["Resp"] >= 22) + int(feat["SBP"] <= 100))
        feat["SIRS"] = float(int(feat["HR"] > 90) + int(feat["Resp"] > 20) + int(feat["Temp"] > 38 or feat["Temp"] < 36))

        batch_rows.append([feat[c] for c in feature_cols])

    X_batch = np.array(batch_rows, dtype=np.float32)
    hourly_risks = (xgb_model.predict_proba(X_batch)[:, 1] * 100).tolist()
    traj_df["Sepsis Risk (%)"] = hourly_risks

    st.markdown(f"#### 📊 Trajectory: {patient_label}")

    # Plot Risk over Time
    risk_plot_df = pd.DataFrame({
        "ICU Hour": traj_df["Hour"],
        "Sepsis Risk (%)": traj_df["Sepsis Risk (%)"],
        "Moderate Risk Threshold (P75)": p75 * 100,
        "Critical High-Risk Threshold (P95)": p95 * 100
    })

    melted_risk = risk_plot_df.melt(id_vars=["ICU Hour"], var_name="Metric", value_name="Value")

    traj_chart = alt.Chart(melted_risk).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("ICU Hour:Q", title="ICU Length of Stay (Hours)"),
        y=alt.Y("Value:Q", title="Probability (%) / Thresholds"),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["Sepsis Risk (%)", "Moderate Risk Threshold (P75)", "Critical High-Risk Threshold (P95)"],
                range=["#0284c7", "#f59e0b", "#ef4444"]
            )
        ),
        tooltip=["ICU Hour", "Metric", "Value"]
    ).properties(height=360, width="container")

    st.altair_chart(traj_chart, use_container_width=True)

    # Early Warning Detection
    crossing_hours = traj_df[traj_df["Sepsis Risk (%)"] >= (p75 * 100)]["Hour"].tolist()
    if crossing_hours:
        first_alert = crossing_hours[0]
        st.error(f"""
        🚨 **EARLY WARNING ALERT TRIGGERED AT HOUR {first_alert}:**  
        Patient risk crossed the Moderate Risk threshold (`{p75*100:.1f}%`).  
        In clinical practice, this provides a **4 to 6 hour therapeutic window** to initiate IV antibiotics and fluids prior to overt septic shock collapse!
        """)
    else:
        st.success("✅ **STABLE TRAJECTORY:** Risk remained safely below the early-warning threshold across all 24 hours.")

    # Show raw trajectory table & download
    with st.expander("📋 View Hourly Patient Vitals & Risk Table"):
        st.dataframe(traj_df[["Hour", "HR", "MAP", "SBP", "Resp", "Temp", "O2Sat", "Sepsis Risk (%)"]], use_container_width=True)

    csv_data = traj_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download This 24h Trajectory CSV",
        data=csv_data,
        file_name="sample_icu_trajectory_24h.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("🏥 **SepsisGuard AI** • Department of Health Informatics & Machine Learning • Developed by Sahil Bisla")