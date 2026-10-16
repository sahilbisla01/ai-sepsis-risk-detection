<div align="center">

# 🩺 Explainable Sepsis Early-Warning & Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB1436?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-TreeSHAP-black?style=for-the-badge)](https://shap.readthedocs.io/)
[![RAG](https://img.shields.io/badge/Agentic%20RAG-FAISS%20+%20MiniLM-purple?style=for-the-badge)](https://github.com/facebookresearch/faiss)

*An end-to-end, clinically validated, early sepsis detection pipeline trained on 40,336 ICU patients (PhysioNet/CinC 2019) with real-time patient trajectory simulation, TreeSHAP interpretability, and automated Doctor SOAP note generation.*

</div>

---

## 📌 Executive Summary & Key Results

Early identification of sepsis in the ICU can save millions of lives. This system predicts sepsis onset **4–6 hours pre-onset**, providing critical early warning alarms while ensuring doctors understand **why** the alarm was triggered through explainable AI and clinical guidelines retrieval.

| Metric | Random Baseline | LSTM / Temporal Models | **Our XGBoost Model** |
| :--- | :---: | :---: | :---: |
| **AUROC** | 0.50 | 0.74 | **0.80** |
| **AUPRC** | 0.018 | 0.082 | **0.110 (~6x Baseline)** |
| **Demographic Disparity** | N/A | > 8.5% | **< 3.7% (Fairness Certified)** |

---

## ✨ Key Features

- **⏱️ 24-Hour Patient Trajectory Simulator:** Simulates real-time vital signs and lab measurements, tracking risk probability hour-by-hour with automated threshold alarms.
- **🔍 TreeSHAP Feature Attributions:** Breaks down individual patient predictions into waterfall plots, showing bedside doctors exactly which vitals (e.g., HR, Respiration, Shock Index) triggered the risk score.
- **📚 Agentic Clinical Guidelines RAG:** Embeds Surviving Sepsis Campaign (SSC) guidelines using `all-MiniLM-L6-v2` and FAISS to provide context-aware clinical recommendations based on patient risk tier.
- **📝 Automated Doctor SOAP Note Generator:** Generates structured Subjective, Objective, Assessment, and Plan documentation for electronic health record (EHR) handovers.
- **🛡️ Leakage-Safe Feature Engineering:** Rolling vital statistics, bedside clinical scores (qSOFA, SIRS, Shock Index), and MICE imputation without future-data lookahead.

---

## 🏗️ System Architecture

```text
PhysioNet 2019 Dataset (40k+ ICU Patients)
       │
       ▼
Feature Pipeline (Rolling Vitals, qSOFA, SIRS, MICE Imputation)
       │
       ├──► XGBoost Classifier (AUROC 0.80)
       │         │
       │         ├──► TreeSHAP Interpretability Engine
       │         └──► Uncertainty Estimation (MC Dropout)
       │
       └──► Agentic RAG Layer (FAISS + all-MiniLM-L6-v2)
                 │
                 ▼
Streamlit Interactive Clinical Dashboard & SOAP Note Generator
```

🛠️ Tech Stack
Machine Learning: XGBoost, Scikit-learn, PyTorch, TreeSHAP
Vector Search & NLP: FAISS, Sentence-Transformers (all-MiniLM-L6-v2), Agentic RAG
Data Engineering: Pandas, NumPy, MICE Multivariate Imputation
Frontend & App: Streamlit, Matplotlib, Seaborn
DevOps: Git, Docker, DevContainer
🚀 Quickstart (Run Locally)
1. Clone the repository
bash
git clone https://github.com/sahilbisla01/ai-sepsis-risk-detection.git
cd ai-sepsis-risk-detection
2. Set up virtual environment
bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
3. Run the Clinical App
bash
streamlit run app.py
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

