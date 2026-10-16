# Explainable Sepsis Risk Detection - Streamlit demo (agentic pipeline)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Sepsis Risk Detection", layout="wide")
st.title("🏥 Explainable Sepsis Risk Detection System")
st.write("Enter patient information to get a risk assessment with clinical evidence.")


# ---------------------------------------------------------------
# Load saved components once (cached)
# ---------------------------------------------------------------
@st.cache_resource
def load_components():
    xgb_model = joblib.load("models/xgboost_v2_mice.pkl")
    rag_index = faiss.read_index("models/guideline_index.faiss")
    with open("models/guideline_documents.pkl", "rb") as f:
        guideline_documents = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return xgb_model, rag_index, guideline_documents, embedder


xgb_model, rag_index, guideline_documents, embedder = load_components()

test_df = pd.read_parquet("data/processed/test_final.parquet")
exclude_cols = ["patient_id", "SepsisLabel"]
feature_cols = [c for c in test_df.columns if c not in exclude_cols]
median_row = test_df[feature_cols].median()
measured_cols = [c for c in feature_cols if c.endswith("_measured")]


# ---------------------------------------------------------------
# Score reference: the model's real score distribution on the test set.
# Because the model was trained with heavy class weighting, its raw
# probabilities are compressed, so we use percentile-based risk tiers
# instead of fixed cut-offs like 30% / 60%.
# ---------------------------------------------------------------
@st.cache_resource
def compute_score_reference():
    probs = xgb_model.predict_proba(test_df[feature_cols])[:, 1]
    return np.sort(probs)


sorted_probs = compute_score_reference()
p75 = float(np.percentile(sorted_probs, 75))   # above this = MODERATE
p95 = float(np.percentile(sorted_probs, 95))   # above this = HIGH


def percentile_rank(score):
    return 100.0 * np.searchsorted(sorted_probs, score) / len(sorted_probs)


# ---------------------------------------------------------------
# Session-state defaults (must be set BEFORE widgets are created)
# ---------------------------------------------------------------
defaults = {
    "hr_input": 90, "sbp_input": 110, "o2sat_input": 97, "map_input": 75,
    "temp_input": 37.0, "resp_input": 18, "age_input": 60,
    "iculos_input": 30, "labs_input": False, "gender_input": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------
# Quick presets (placed before widgets so they can set values safely)
# ---------------------------------------------------------------
st.write("**Quick presets (for demo purposes):**")
pc1, pc2 = st.columns(2)
with pc1:
    use_low = st.button("Fill Low-Risk Example")
with pc2:
    use_high = st.button("Fill High-Risk Example")

if use_low:
    st.session_state.update({
        "hr_input": 78, "sbp_input": 118, "o2sat_input": 98, "map_input": 88,
        "temp_input": 36.9, "resp_input": 15, "age_input": 45,
        "iculos_input": 8, "labs_input": False,
    })
    st.rerun()

if use_high:
    st.session_state.update({
        "hr_input": 135, "sbp_input": 80, "o2sat_input": 88, "map_input": 55,
        "temp_input": 39.2, "resp_input": 32, "age_input": 70,
        "iculos_input": 60, "labs_input": True,
    })
    st.rerun()

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.header("Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    hr = st.number_input("Heart Rate (HR)", 0, 250, key="hr_input")
    map_val = st.number_input("Mean Arterial Pressure (MAP)", 0, 200, key="map_input")
    resp = st.number_input("Respiratory Rate (Resp)", 0, 60, key="resp_input")

with col2:
    sbp = st.number_input("Systolic BP (SBP)", 0, 250, key="sbp_input")
    temp = st.number_input("Temperature (Celsius)", 30.0, 42.0, key="temp_input")
    age = st.number_input("Age", 0, 120, key="age_input")

with col3:
    o2sat = st.number_input("Oxygen Saturation (O2Sat)", 0, 100, key="o2sat_input")
    gender = st.selectbox("Gender", [0, 1],
                          format_func=lambda x: "Female" if x == 0 else "Male",
                          key="gender_input")
    iculos = st.slider("Hours in ICU so far (ICULOS)", 1, 150, key="iculos_input")
    labs_ordered = st.checkbox("Lab tests ordered (lactate, WBC, etc.)", key="labs_input")

# ---------------------------------------------------------------
# Agentic risk assessment
# ---------------------------------------------------------------
if st.button("🔍 Assess Sepsis Risk"):

    # Build the full feature row: median for unknown features, override with inputs
    row = median_row.copy()
    row["HR"], row["MAP"], row["Resp"] = hr, map_val, resp
    row["SBP"], row["Temp"], row["Age"] = sbp, temp, age
    row["O2Sat"], row["Gender"] = o2sat, gender
    row["ShockIndex"] = hr / max(sbp, 1)
    row["ICULOS"] = iculos

    # Rolling means: assume the current reading reflects a sustained trend
    for name, val in [("HR", hr), ("MAP", map_val), ("Resp", resp),
                      ("Temp", temp), ("SBP", sbp), ("O2Sat", o2sat)]:
        row[f"{name}_mean6"] = val
        row[f"{name}_mean12"] = val

    # Rolling std: assume moderate real-world variability
    for name, val in [("HR", 5.0), ("MAP", 5.0), ("Resp", 2.0),
                      ("Temp", 0.3), ("SBP", 5.0), ("O2Sat", 1.5)]:
        row[f"{name}_std6"] = val
        row[f"{name}_std12"] = val

    # Clinical scores from current values
    row["qSOFA"] = int(resp >= 22) + int(sbp <= 100)
    row["SIRS"] = int(hr > 90) + int(resp > 20) + int(temp > 38 or temp < 36)

    # Lab-measured flags
    for c in measured_cols:
        row[c] = 1 if labs_ordered else 0

    # STEP 1: predict
    X = row[feature_cols].values.reshape(1, -1)
    risk = float(xgb_model.predict_proba(X)[0, 1])
    pct = percentile_rank(risk)

    st.header("Results")

    # STEP 2: agent decides the path based on the risk tier
    if risk < p75:
        st.success(f"### Risk Score: {risk*100:.1f}% — LOW RISK")
        st.write(f"This score is higher than {pct:.0f}% of hourly records in the test set.")
        st.write("**Agent decision:** risk is low, so no evidence retrieval is needed. "
                 "Continue routine monitoring.")
    else:
        if risk >= p95:
            st.error(f"### Risk Score: {risk*100:.1f}% — HIGH RISK")
        else:
            st.warning(f"### Risk Score: {risk*100:.1f}% — MODERATE RISK")
        st.write(f"This score is higher than {pct:.0f}% of hourly records in the test set.")

        # Key findings from the inputs
        findings = []
        if hr > 90:
            findings.append("elevated heart rate")
        if map_val < 70:
            findings.append("low mean arterial pressure")
        if resp >= 22:
            findings.append("elevated respiratory rate")
        if row["ShockIndex"] > 0.9:
            findings.append("elevated shock index")
        if o2sat < 92:
            findings.append("low oxygen saturation")
        if temp > 38 or temp < 36:
            findings.append("abnormal temperature")
        if not findings:
            findings.append("general vital sign patterns")
        query = ", ".join(findings)
        st.write(f"**Key Findings:** {query}")

        # STEP 3: agent checks if the score is borderline (close to a tier cut-off)
        near_p75 = abs(risk - p75) < 0.10 * p75
        near_p95 = abs(risk - p95) < 0.10 * p95
        borderline = near_p75 or near_p95
        top_k = 5 if borderline else 3
        if borderline:
            st.info("⚠️ Score is close to a risk-tier boundary, so the agent "
                    "retrieves extra clinical evidence (top-5).")

        # STEP 4: retrieve evidence
        q_emb = embedder.encode([query], normalize_embeddings=True)
        scores, idxs = rag_index.search(np.array(q_emb), top_k)

        st.subheader("📋 Supporting Clinical Evidence")
        for i, (idx, sc) in enumerate(zip(idxs[0], scores[0])):
            with st.expander(f"[{i+1}] Relevance: {sc:.2f}"):
                st.write(guideline_documents[idx])

        st.subheader("📝 Summary")
        st.write(f"This patient has a **{risk*100:.1f}%** predicted risk of sepsis "
                 f"(higher than {pct:.0f}% of test-set hours), driven by **{query}** "
                 f"[1-{top_k}]. Clinical correlation and physician judgment are recommended.")

    st.caption(f"Risk tiers are percentile-based on the test set: LOW < {p75*100:.1f}%, "
               f"MODERATE {p75*100:.1f}%-{p95*100:.1f}%, HIGH ≥ {p95*100:.1f}%. "
               "Features not entered in this form are filled with test-set medians.")

    with st.expander("Show model input values (for debugging / explanation)"):
        show = ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat", "ShockIndex",
                "ICULOS", "qSOFA", "SIRS", "HospAdmTime", "Unit1", "Unit2"]
        st.write(row[show])