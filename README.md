# 🏥 SepsisGuard AI: Explainable ICU Sepsis Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sepsis-detection-ai.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sahilbisla01/ai-sepsis-risk-detection)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![AUROC](https://img.shields.io/badge/Validation%20AUROC-0.798-emerald)](https://github.com/sahilbisla01/ai-sepsis-risk-detection)

An agentic AI and machine learning clinical decision support system for early sepsis risk prediction in intensive care units (ICU), coupled with TreeSHAP explainability, 24-hour time-series trajectory simulation, automated SOAP note generation, and clinical guideline retrieval (RAG).

---

## 🌟 Key Features

- **Predictive ML Modeling:** Optimized XGBoost model trained on PhysioNet/CinC 2019 (40,336 ICU patients) with MICE imputation and percentile-based calibration (AUROC 0.80, AUPRC 0.11).
- **🔍 TreeSHAP Feature Attributions:** Exact log-odds feature contributions displayed as interactive color-coded bar plots (Red = Risk-increasing, Blue = Protective).
- **📈 24-Hour Patient Trajectory Simulator:** Time-series simulation tracking patient risk hour-by-hour across an ICU stay, triggering early warnings 4 to 6 hours prior to clinical septic shock onset.
- **📝 Automated Doctor SOAP Note Generator:** Synthesizes patient vitals, bedside clinical scores (qSOFA, SIRS, Shock Index), and Surviving Sepsis Campaign guidelines into standardized hospital consultation notes with one-click `.txt` download.
- **📚 Semantic Guideline Retrieval (RAG):** FAISS index combined with `sentence-transformers` (`all-MiniLM-L6-v2`) to retrieve targeted recommendations from the Surviving Sepsis Campaign 2021 guidelines based on detected vital abnormalities.
- **⚡ Quick Clinical Presets:** One-click presets for Stable Post-Op, Borderline Ward Monitoring, and Acute Septic Shock cases.

---

## 🚀 Quickstart (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/sahilbisla01/ai-sepsis-risk-detection.git
cd ai-sepsis-risk-detection
```

### 2. Set up the environment
```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

---

## ☁️ Live Cloud Deployment

Access the deployed system live in your browser:
👉 **[https://sepsis-detection-ai.streamlit.app](https://sepsis-detection-ai.streamlit.app)**

---

## 📁 Repository Structure

```text
├── app.py                     # Main Streamlit web application & dashboards
├── requirements.txt           # Python dependencies (Streamlit, XGBoost, FAISS, PyTorch)
├── Dockerfile                 # Container deployment specification
├── .streamlit/config.toml     # Streamlit performance & theme settings
├── models/                    # Serialized models & FAISS guideline indices
│   ├── xgboost_v2_mice.pkl    # Trained XGBoost sepsis detection model
│   ├── guideline_index.faiss  # Vector index for clinical guidelines
│   └── guideline_documents.pkl# Sepsis guideline reference texts
├── notebooks/                 # Jupyter research & experiment notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_split_and_features.ipynb
│   ├── 04_xgboost_training.ipynb
│   ├── 07_explainability_shap.ipynb
│   ├── 08_uncertainty_estimation.ipynb
│   ├── 09_fairness_analysis.ipynb
│   └── 10_rag_pipeline.ipynb
├── reports/                   # Performance, fairness, and uncertainty metrics
└── data/processed/            # Evaluation test set for calibration
```

---

## ⚖️ Disclaimer
*This system is intended for research and educational purposes to demonstrate agentic AI and clinical decision support. It should not be used as a standalone diagnostic tool without physician oversight.*
