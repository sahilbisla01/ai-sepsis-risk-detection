# Explainable Sepsis Risk Detection - Streamlit demo (Agentic Pipeline)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import pickle
from datetime import datetime
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
        transition: transform 0.2s ease, box-shadow 0.2s ease;
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
    
    /* Preset buttons */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
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

# Load Reference Data
test_df = pd.read_parquet("data/processed/test_final.parquet")
exclude_cols = ["patient_id", "SepsisLabel"]
feature_cols = [c for c in test_df.columns if c not in exclude_cols]
median_row = test_df[feature_cols].median()
measured_cols = [c for c in feature_cols if c.endswith("_measured")]

# ---------------------------------------------------------------
# Score Distribution Reference
# ---------------------------------------------------------------
@st.cache_resource
def compute_score_reference():
    probs = xgb_model.predict_proba(test_df[feature_cols])[:, 1]
    return np.sort(probs)

sorted_probs = compute_score_reference()
p75 = float(np.percentile(sorted_probs, 75))   # Moderate threshold
p95 = float(np.percentile(sorted_probs, 95))   # High threshold

def percentile_rank(score):
    return 100.0 * np.searchsorted(sorted_probs, score) / len(sorted_probs)

# ---------------------------------------------------------------
# Sidebar: System Intelligence & Clinical Benchmarks
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
    - **Imputation:** Multiple Imputation (MICE)
    - **Guideline RAG:** FAISS + MiniLM-L6-v2
    - **Knowledge Base:** Surviving Sepsis Guidelines
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Risk Calibration Tiers")
    st.markdown(f"""
    - 🟢 **Low Risk:** `< {p75*100:.1f}%` (Below 75th percentile)
    - 🟡 **Moderate Risk:** `{p75*100:.1f}% - {p95*100:.1f}%` (75th–95th percentile)
    - 🔴 **High Risk:** `≥ {p95*100:.1f}%` (Top 5% highest risk)
    """)
    
    st.markdown("---")
    st.markdown("### ⚖️ Algorithmic Fairness")
    st.caption("Evaluated across patient demographics:")
    st.progress(0.798)
    st.caption("Age Bracket Disparity: < 3.7% | Gender Disparity: < 2.4%")
    
    st.markdown("---")
    st.caption("Developed for ICU clinical decision support. Not a standalone diagnostic tool.")

# ---------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🏥</span> Explainable Sepsis Risk Detection System
    </div>
    <div class="hero-subtitle">
        Real-time ICU vital sign assessment powered by Machine Learning and Agentic Clinical Guideline Retrieval.
    </div>
    <div>
        <span class="pill-badge">🤖 Agentic Triage</span>
        <span class="pill-badge">🫀 ICU Vital Signs</span>
        <span class="pill-badge">📋 SSC Guidelines (RAG)</span>
        <span class="pill-badge">⚡ Real-time Calibration</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Session State & Preset Profiles
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Patient Information Form
# ---------------------------------------------------------------
st.markdown("### 📋 Patient Clinical Parameters")

tab_vitals, tab_history = st.tabs(["🫀 Vital Signs & Hemodynamics", "👤 Patient Demographics & ICU History"])

with tab_vitals:
    vcol1, vcol2, vcol3 = st.columns(3)
    
    with vcol1:
        hr = st.number_input(
            "Heart Rate (BPM)", 
            min_value=0, max_value=250, 
            key="hr_input", 
            help="Normal adult range: 60-100 BPM"
        )
        if hr > 100:
            st.caption("⚠️ :red[Tachycardia (>100 BPM)]")
        elif hr < 60:
            st.caption("ℹ️ :blue[Bradycardia (<60 BPM)]")
        else:
            st.caption("✅ Normal Range (60-100 BPM)")

        sbp = st.number_input(
            "Systolic Blood Pressure (mmHg)", 
            min_value=0, max_value=250, 
            key="sbp_input",
            help="Normal adult range: 90-120 mmHg"
        )
        if sbp < 90:
            st.caption("⚠️ :red[Hypotension (<90 mmHg)]")
        else:
            st.caption("✅ Normal Range (90-120 mmHg)")

    with vcol2:
        map_val = st.number_input(
            "Mean Arterial Pressure (MAP)", 
            min_value=0, max_value=200, 
            key="map_input",
            help="Target MAP for tissue perfusion: >= 65 mmHg"
        )
        if map_val < 65:
            st.caption("⚠️ :red[Critical Hypoperfusion (<65 mmHg)]")
        else:
            st.caption("✅ Adequate Perfusion (>=65 mmHg)")

        resp = st.number_input(
            "Respiratory Rate (breaths/min)", 
            min_value=0, max_value=60, 
            key="resp_input",
            help="qSOFA threshold: >= 22 breaths/min"
        )
        if resp >= 22:
            st.caption("⚠️ :red[Tachypnea (>=22, qSOFA positive)]")
        else:
            st.caption("✅ Normal Range (12-20)")

    with vcol3:
        temp = st.number_input(
            "Temperature (°C)", 
            min_value=30.0, max_value=42.0, 
            step=0.1, 
            key="temp_input",
            help="SIRS criteria: >38°C or <36°C"
        )
        if temp > 38.0:
            st.caption("⚠️ :red[Hyperthermia / Fever (>38°C)]")
        elif temp < 36.0:
            st.caption("⚠️ :red[Hypothermia (<36°C)]")
        else:
            st.caption("✅ Normal Range (36.5-37.5°C)")

        o2sat = st.number_input(
            "Oxygen Saturation (SpO2 %)", 
            min_value=0, max_value=100, 
            key="o2sat_input",
            help="Normal adult range: >= 95%"
        )
        if o2sat < 92:
            st.caption("⚠️ :red[Hypoxemia (<92%)]")
        else:
            st.caption("✅ Normal Range (>=95%)")

with tab_history:
    hcol1, hcol2, hcol3 = st.columns(3)
    
    with hcol1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, key="age_input")
        gender = st.selectbox(
            "Biological Gender", 
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            key="gender_input"
        )
        
    with hcol2:
        iculos = st.slider(
            "ICU Length of Stay (Hours)", 
            min_value=1, max_value=150, 
            key="iculos_input",
            help="Cumulative hours spent in the intensive care unit"
        )
        
    with hcol3:
        st.write("**Laboratory Tests**")
        labs_ordered = st.checkbox(
            "Lab tests ordered (Lactate, WBC, Platelets, Creatinine)", 
            key="labs_input",
            help="Signals active clinical investigation for sepsis"
        )
        if labs_ordered:
            st.caption("🧪 Laboratory measured indicators active")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Assessment Action
# ---------------------------------------------------------------
assess_col1, assess_col2, assess_col3 = st.columns([1, 2, 1])
with assess_col2:
    run_assessment = st.button("🔍 Assess Sepsis Risk & Evidence", type="primary", use_container_width=True)

# ---------------------------------------------------------------
# Inference, Decision Routing & Evidence Retrieval
# ---------------------------------------------------------------
if run_assessment:
    # 1. Prepare Feature Vector
    row = median_row.copy()
    row["HR"], row["MAP"], row["Resp"] = hr, map_val, resp
    row["SBP"], row["Temp"], row["Age"] = sbp, temp, age
    row["O2Sat"], row["Gender"] = o2sat, gender
    row["ShockIndex"] = hr / max(sbp, 1)
    row["ICULOS"] = iculos

    # Rolling statistics approximations
    for name, val in [("HR", hr), ("MAP", map_val), ("Resp", resp),
                      ("Temp", temp), ("SBP", sbp), ("O2Sat", o2sat)]:
        row[f"{name}_mean6"] = val
        row[f"{name}_mean12"] = val

    for name, val in [("HR", 5.0), ("MAP", 5.0), ("Resp", 2.0),
                      ("Temp", 0.3), ("SBP", 5.0), ("O2Sat", 1.5)]:
        row[f"{name}_std6"] = val
        row[f"{name}_std12"] = val

    # Bedside Scores
    qsofa = int(resp >= 22) + int(sbp <= 100)
    sirs = int(hr > 90) + int(resp > 20) + int(temp > 38 or temp < 36)
    shock_index = row["ShockIndex"]
    row["qSOFA"] = qsofa
    row["SIRS"] = sirs

    for c in measured_cols:
        row[c] = 1 if labs_ordered else 0

    # Model Prediction
    X = row[feature_cols].values.reshape(1, -1)
    risk = float(xgb_model.predict_proba(X)[0, 1])
    pct = percentile_rank(risk)

    st.markdown("---")
    st.markdown("## 📊 Assessment Results & Triage")

    # Risk Tier Classification & Banner Styling
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

    # Top Hero Result Card
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

    # 3-Card Summary Grid
    rcol1, rcol2, rcol3 = st.columns(3)
    
    with rcol1:
        st.markdown("""
        <div class="med-card">
            <h4 style="margin-top:0;">🤖 Agentic Triage Router</h4>
        """, unsafe_allow_html=True)
        st.write(f"**Recommended Action:** {triage_action}")
        if risk < p75:
            st.info("Agent Decision: No immediate guideline retrieval needed for low-risk status.")
        else:
            st.warning("Agent Decision: Evidence retrieval triggered for guideline-directed support.")
        st.markdown("</div>", unsafe_allow_html=True)

    with rcol2:
        st.markdown("""
        <div class="med-card">
            <h4 style="margin-top:0;">📋 Bedside Clinical Scores</h4>
        """, unsafe_allow_html=True)
        st.write(f"• **qSOFA Score:** `{qsofa} / 2` " + ("(⚠️ High)" if qsofa >= 2 else "(Normal)"))
        st.write(f"• **SIRS Criteria:** `{sirs} / 3` " + ("(⚠️ SIRS Positive)" if sirs >= 2 else "(Negative)"))
        st.write(f"• **Shock Index (HR/SBP):** `{shock_index:.2f}` " + ("(⚠️ Elevated >0.9)" if shock_index > 0.9 else "(Normal)"))
        st.markdown("</div>", unsafe_allow_html=True)

    with rcol3:
        st.markdown("""
        <div class="med-card">
            <h4 style="margin-top:0;">🎯 Model Calibration</h4>
        """, unsafe_allow_html=True)
        st.write(f"• **Cohort Percentile:** `{pct:.1f}th Percentile`")
        st.write(f"• **Moderate Cutoff (P75):** `{p75*100:.1f}%`")
        st.write(f"• **High Cutoff (P95):** `{p95*100:.1f}%`")
        st.markdown("</div>", unsafe_allow_html=True)

    # Key Findings Detection
    findings = []
    if hr > 90:
        findings.append("elevated heart rate (tachycardia)")
    if map_val < 70:
        findings.append("low mean arterial pressure (hypotension)")
    if resp >= 22:
        findings.append("elevated respiratory rate (tachypnea)")
    if shock_index > 0.9:
        findings.append("elevated shock index (>0.9)")
    if o2sat < 92:
        findings.append("low oxygen saturation (hypoxemia)")
    if temp > 38:
        findings.append("elevated body temperature (fever)")
    elif temp < 36:
        findings.append("subnormal body temperature (hypothermia)")

    if not findings:
        findings.append("normal baseline vital sign parameters")

    st.markdown("#### 🔍 Detected Abnormality Profile")
    pills_html = "".join([f'<span class="finding-pill">⚠️ {f.title()}</span>' for f in findings])
    st.markdown(pills_html, unsafe_allow_html=True)

    # RAG Retrieval for Moderate or High Risk
    if risk >= p75:
        near_p75 = abs(risk - p75) < 0.10 * p75
        near_p95 = abs(risk - p95) < 0.10 * p95
        borderline = near_p75 or near_p95
        top_k = 5 if borderline else 3

        if borderline:
            st.info("⚠️ **Adaptive Retrieval Notice:** Score is near a decision boundary; retrieving expanded top-5 guideline evidence.")

        query = ", ".join(findings)
        q_emb = embedder.encode([query], normalize_embeddings=True)
        scores, idxs = rag_index.search(np.array(q_emb), top_k)

        st.markdown("### 📚 Supporting Clinical Guidelines (Surviving Sepsis Campaign)")
        st.caption(f"Retrieved via semantic embedding matching on patient findings: '{query}'")

        for i, (idx, sc) in enumerate(zip(idxs[0], scores[0])):
            with st.expander(f"📌 Evidence #{i+1} | Semantic Relevance Match: {sc:.2f}", expanded=(i == 0)):
                st.markdown(f"> *\"{guideline_documents[idx]}\"*")
                st.caption(f"Source: Surviving Sepsis Campaign Guidelines 2021 | Document Index #{idx}")

    # Clinical Consultation Note Export
    st.markdown("---")
    st.markdown("### 📝 Export Clinical Summary Note")
    
    note_text = f"""================================================================================
CLINICAL SEPSIS RISK ASSESSMENT CONSULTATION NOTE
Generated by SepsisGuard AI System
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

PATIENT DATA SUMMARY:
- Age: {age} | Gender: {'Female' if gender == 0 else 'Male'}
- ICU Length of Stay: {iculos} hours
- Labs Ordered: {'Yes' if labs_ordered else 'No'}

VITAL SIGNS:
- Heart Rate (HR): {hr} BPM
- Blood Pressure: {sbp}/{map_val} mmHg (MAP: {map_val} mmHg)
- Respiratory Rate: {resp} breaths/min
- SpO2: {o2sat}%
- Temperature: {temp} °C

BEDSIDE METRICS:
- Shock Index: {shock_index:.2f}
- qSOFA Score: {qsofa} / 2
- SIRS Criteria: {sirs} / 3

AI RISK PREDICTION:
- Predicted Sepsis Probability: {risk*100:.1f}%
- Cohort Percentile: {pct:.1f}th Percentile
- Risk Category: {tier_label}
- Detected Findings: {', '.join(findings)}

RECOMMENDED ACTION:
{triage_action}

DISCLAIMER:
This report is generated by an algorithmic decision-support tool (XGBoost + RAG)
for investigational use only and must be verified by the attending physician.
================================================================================"""

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        st.download_button(
            label="📥 Download Clinical Consultation Note (.txt)",
            data=note_text,
            file_name=f"sepsis_assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Debug Model Inputs
    with st.expander("🛠️ Inspect Engineered Features (For Clinical / Technical Audits)"):
        show_cols = ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat", "ShockIndex",
                     "ICULOS", "qSOFA", "SIRS", "HospAdmTime", "Unit1", "Unit2"]
        st.dataframe(row[show_cols].to_frame().T, use_container_width=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("🏥 **SepsisGuard AI** • Department of Health Informatics & Machine Learning • Developed by Sahil Bisla")