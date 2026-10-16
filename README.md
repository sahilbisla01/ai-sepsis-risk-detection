# 🏥 Explainable Sepsis Risk Detection System

An agentic AI and machine learning system for early sepsis risk prediction in intensive care units (ICU), coupled with clinical guideline retrieval (RAG) and explainability.

---

## 🌟 Key Features

- **Predictive ML Modeling:** Optimized XGBoost model trained on clinical ICU time-series data with MICE imputation and percentile-based calibration.
- **Agentic Decision Pipeline:** Dynamically determines clinical risk tiers (Low, Moderate, High) and triggers guideline retrieval for actionable physician support.
- **RAG Clinical Evidence:** FAISS index combined with `sentence-transformers` (`all-MiniLM-L6-v2`) to retrieve relevant Surviving Sepsis Campaign guidelines based on patient vital abnormalities.
- **Interactive Web Interface:** Streamlit dashboard with instant preset patient cases, interactive vital sign adjustments, and real-time clinical evidence display.

---

## 🚀 Quickstart (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up the environment
```bash
# Create and activate a virtual environment
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

## ☁️ Deployment

### Deploy to Streamlit Community Cloud (Free)
1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** and select:
   - **Repository:** your repository
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

---

## 📁 Repository Structure

```text
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container deployment specification
├── models/                    # Serialized models & FAISS guideline indices
│   ├── xgboost_v2_mice.pkl    # Trained XGBoost sepsis detection model
│   ├── guideline_index.faiss  # Vector index for clinical guidelines
│   └── guideline_documents.pkl# Sepsis guideline reference texts
├── notebooks/                 # Jupyter research & experiment notebooks
├── reports/                   # Performance, fairness, and uncertainty metrics
└── data/processed/            # Evaluation test set for calibration
```

---

## ⚖️ Disclaimer
*This system is intended for research and educational purposes to demonstrate agentic AI and clinical decision support. It should not be used as a standalone diagnostic tool without physician oversight.*
