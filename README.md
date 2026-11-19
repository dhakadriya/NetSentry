# NetSentry: Intelligent Network Intrusion Detection & Analysis System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NetSentry is an enterprise-ready, machine learning-powered Network Intrusion Detection System (NIDS) designed to ingest, inspect, and classify network flow records into **Normal** or **Malicious** network traffic with high precision and low false-positive rates.

---

## 1. Problem Statement & Motivation

Modern computer networks face continuous cybersecurity threats including Denial of Service (DoS/DDoS), reconnaissance probes, brute force attacks, exploits, and zero-day vulnerabilities. Traditional signature-based intrusion detection systems (Snort, Suricata) rely on fixed rules that struggle against polymorphic attacks and novel malicious patterns.

**NetSentry** provides an intelligent, data-driven solution that:
1. Validates and preprocesses complex network flow datasets (e.g., UNSW-NB15 and CIC-IDS2017).
2. Investigates the impact of multiple feature selection strategies on detection accuracy and latency.
3. Implements, optimizes, and benchmarks four supervised machine learning algorithms:
   - **Logistic Regression** (L1/L2 regularized baseline)
   - **Decision Tree** (Interpretable rule-based classifier)
   - **Random Forest** (High-accuracy ensemble with class-weight balancing)
   - **K-Nearest Neighbors (KNN)** (Distance-metric instance-based classification)
4. Provides a full **FastAPI REST API** for single-packet and batch network flow inference.
5. Delivers a **Streamlit Web Dashboard** for exploratory data analysis (EDA), model training, algorithm comparison, threat analytics, and executive detection reports.

---

## 2. System Architecture

```text
+-------------------------------------------------------------------------------+
|                                NetSentry NIDS                                 |
+-------------------------------------------------------------------------------+
|                                                                               |
|   +-------------------+    +----------------------+    +------------------+   |
|   |   Network Flow    |--->|   Schema Validator   |--->|  Data Cleaner    |   |
|   |  CSV / Live Flow  |    |  (UNSW-NB15/CIC-IDS) |    |  & Normalizer    |   |
|   +-------------------+    +----------------------+    +------------------+   |
|                                                                 |             |
|                                                                 v             |
|   +-------------------+    +----------------------+    +------------------+   |
|   | ML Evaluation     |<---|  4 ML Classifiers    |<---| Feature Selector |   |
|   | - Accuracy / F1   |    |  (LR, DT, RF, KNN)   |    | (Variance/Corr/  |   |
|   | - FPR / Confusion |    +----------------------+    |  RandomForest/MI)|   |
|   +-------------------+               |                +------------------+   |
|             |                         v                                       |
|             v              +----------------------+                           |
|   +-------------------+    |  FastAPI REST Engine |                           |
|   | SQLite Database   |<-->|  - /api/v1/predict   |                           |
|   | & Audit Store     |    |  - /api/v1/models    |                           |
|   +-------------------+    +----------------------+                           |
|                                       |                                       |
|                                       v                                       |
|                            +----------------------+                           |
|                            |  Streamlit Dashboard |                           |
|                            |  - EDA & Training    |                           |
|                            |  - Threat Radar      |                           |
|                            |  - Detection Reports |                           |
|                            +----------------------+                           |
+-------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Core & Backend:** Python 3.10+, FastAPI, Pydantic v2, Uvicorn, SQLAlchemy 2.0, SQLite
- **Machine Learning:** Scikit-learn, Pandas, NumPy, SciPy, Joblib
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Frontend Dashboard:** Streamlit
- **Testing & Quality:** Pytest, Ruff, MyPy, HTTPX
- **DevOps:** Docker, Docker Compose, GitHub Actions CI

---

## 4. Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/dhakadriya/netsentry.git
cd netsentry

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # or pip install -e .
```

### Running the Services

1. **Launch the FastAPI REST Server:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: `http://localhost:8000/docs`

2. **Launch the Streamlit Dashboard:**
```bash
streamlit run src/frontend/app.py --server.port 8501
```
Access the web dashboard: `http://localhost:8501`

---

## 5. REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System health check and diagnostics |
| `POST` | `/api/v1/datasets/upload` | Upload and validate network flow CSV |
| `GET` | `/api/v1/datasets/{id}/eda` | Exploratory Data Analysis & statistics |
| `POST` | `/api/v1/models/train` | Train a machine learning classifier |
| `POST` | `/api/v1/models/compare` | Benchmark all 4 algorithms on selected dataset |
| `POST` | `/api/v1/predict/single` | Real-time classification of single flow record |
| `POST` | `/api/v1/predict/batch` | Batch classification of uploaded traffic CSV |
| `GET` | `/api/v1/alerts` | Query detected intrusion alerts and threat levels |
| `GET` | `/api/v1/export/report/{id}/csv` | Download intrusion detection report |

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
