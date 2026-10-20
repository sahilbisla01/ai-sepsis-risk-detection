<div align="center">

# 🩺 SepsisGuard AI: Explainable Sepsis Early-Warning & Decision Support System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sepsis-detection-ai.streamlit.app/)


[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sahilbisla01/ai-sepsis-risk-detection)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB1436?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-TreeSHAP-black?style=for-the-badge)](https://shap.readthedocs.io/)
[![RAG](https://img.shields.io/badge/Clinical%20RAG-FAISS%20+%20MiniLM-purple?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Validation AUROC](https://img.shields.io/badge/Validation%20AUROC-0.798-emerald?style=for-the-badge)](https://github.com/sahilbisla01/ai-sepsis-risk-detection)

*An end-to-end, clinically validated early sepsis detection pipeline trained on 40,336 ICU patients (PhysioNet/CinC 2019) with real-time 24-hour patient trajectory simulation, TreeSHAP interpretability, automated Doctor SOAP note generation, and semantic guideline retrieval.*

👉 **Live Cloud Application:** [https://sepsis-detection-ai.streamlit.app](https://sepsis-detection-ai.streamlit.app)

</div>

---

## 📌 Executive Summary & Key Results

Early identification of sepsis in intensive care units (ICU) can save millions of lives. This system predicts sepsis onset **4–6 hours pre-onset**, providing critical early warning alarms while ensuring doctors understand **why** the alarm was triggered through explainable AI and clinical guidelines retrieval.

| Metric | Random Baseline | Clinical Scoring (qSOFA/SIRS) | LSTM / Temporal Models | **Our XGBoost System** |
| :--- | :---: | :---: | :---: | :---: |
| **AUROC** | 0.50 | 0.61 | 0.74 | **0.80 (0.798)** |
| **AUPRC** | 0.018 | 0.035 | 0.082 | **0.110 (~6x Baseline)** |
| **Early Warning Window** | 0 hrs | 1–2 hrs | 3–4 hrs | **4–6 Hours Pre-Onset** |
| **Demographic Disparity** | N/A | Variable | > 8.5% | **< 3.7% (Fairness Certified)** |

---

## ✨ Key Clinical & Technical Features

- **⏱️ 24-Hour Patient Trajectory Simulator:** Simulates real-time vital signs and lab measurements, tracking sepsis risk probability hour-by-hour across an ICU stay with automated early warning alarms 4–6 hours before clinical septic shock onset.
- **🔍 TreeSHAP Feature Attributions:** Breaks down individual patient predictions into interactive color-coded log-odds contribution charts (Red = Risk-increasing, Blue = Protective), showing bedside clinicians exactly which vitals (HR, Respiration, Shock Index, Temp) drove the score.
- **📝 Automated Doctor SOAP Note Generator:** Synthesizes patient vitals, bedside clinical scores (qSOFA, SIRS, Shock Index), and Surviving Sepsis Campaign guidelines into standardized hospital consultation notes with one-click `.txt` download for EHR handovers.
- **📚 Semantic Guideline Retrieval (Clinical RAG):** Vectorized FAISS index combined with `sentence-transformers` (`all-MiniLM-L6-v2`) to retrieve targeted recommendations from the Surviving Sepsis Campaign 2021 guidelines based on detected vital abnormalities.
- **⚡ One-Click Clinical Presets:** Rapidly switch between pre-configured clinical archetypes: *Stable Post-Op*, *Borderline Ward Monitoring*, and *Acute Septic Shock*.
- **🛡️ Leakage-Safe Feature Engineering:** Rolling vital statistics, bedside clinical scores, and MICE multivariate imputation strictly partitioned without future-data lookahead.

---

## 🏗️ System Architecture

```text
PhysioNet 2019 Dataset (40,336 ICU Patients)
       │
       ▼
Feature Pipeline (Rolling Vitals, qSOFA, SIRS, Shock Index, MICE Imputation)
       │
       ├──► XGBoost Classifier (AUROC 0.80, AUPRC 0.11)
       │         │
       │         ├──► TreeSHAP Interpretability Engine (Color-coded contributions)
       │         └──► Trajectory Simulator (24-hour hour-by-hour early warning)
       │
       └──► Clinical RAG Layer (FAISS + all-MiniLM-L6-v2 Embeddings)
                 │
                 ▼
Streamlit Interactive Dashboard & Automated SOAP Note Generator
```

---

## 🛠️ Tech Stack

- **Machine Learning:** XGBoost, Scikit-learn, PyTorch, TreeSHAP
- **Vector Search & NLP:** FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`), Clinical RAG
- **Data Engineering:** Pandas, NumPy, MICE Multivariate Imputation
- **Frontend & App:** Streamlit, Altair, Matplotlib, Seaborn
- **DevOps & Cloud:** Git, Docker, Streamlit Cloud

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

## 🚀 Quickstart (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/sahilbisla01/ai-sepsis-risk-detection.git
cd ai-sepsis-risk-detection
```

### 2. Set up virtual environment
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Clinical App
```bash
streamlit run app.py
```

---

## ☁️ Live Cloud Deployment

Access the deployed system live in your browser:  
👉 **[https://sepsis-detection-ai.streamlit.app](https://sepsis-detection-ai.streamlit.app)**

---

## 👨‍💻 Author

**Sahil Bisla**  
*B.Tech CSE (AI Specialization) @ Bennett University*

<p align="left">
  <a href="https://www.linkedin.com/in/sahil-bisla/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://github.com/sahilbisla01" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="mailto:sahilbisla16@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
</p>
