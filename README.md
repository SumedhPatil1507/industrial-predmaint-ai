# industrial-predmaint-ai

**Live Demo:** https://industrial-predmaint-ai-d3sdstpce4nxkhghcq8zpk.streamlit.app

Full-stack Industrial Machine Predictive Maintenance System. Self-contained, runs entirely on Streamlit Cloud with no backend required.

---

## Business Impact

Manufacturing plants lose **Rs.40L to Rs.4Cr per unplanned breakdown** in production loss, emergency repairs, and missed deadlines. This system predicts failures before they happen.

| Metric | Impact |
|--------|--------|
| Breakdown prediction accuracy | ~95% AUC-ROC |
| Unplanned downtime reduction | 40-60% |
| Annual savings (mid-size plant) | Rs.20L - 2Cr (-) |
| Payback period | 3-6 months |
| Assets monitored | 10 machines, 5 types |
| Data coverage | 219,000 records, 3+ years |

---

## Features

### Core ML
- Random Forest classifier with class-balanced training
- Isolation Forest anomaly detection on normal-state data
- SHAP explainability - feature importance + beeswarm
- Engineered features: temp diff, vibration total, power per load
- Model Registry - version, compare, and track every training run

### Live IoT Simulation
- In-browser real-time sensor stream (no WebSocket server needed on Cloud)
- Realistic degradation physics - gradual wear, spike events, simulated repairs
- Live breakdown probability computed on every tick
- Fleet-wide dashboard showing all 5 assets simultaneously

### Analytics
- 12-page interactive dashboard - zero static plots, 100% Plotly
- EDA Explorer with 10 tabs: distributions, trends, boxplots, violins, 3D scatter, rolling anomaly band, scatter matrix, correlation heatmap
- Interactive filters: machine type, date range, breakdown status

### Business Tools
- Machine Health Score (0-100) with component radar chart
- Time-to-Failure prediction using degradation trend + hazard rate
- Downtime Cost Calculator - INR/USD, ROI waterfall, all-machine comparison
- AI Maintenance Advisor - prescriptive recommendations per asset

### Production Ready
- Universal file upload: CSV, Excel, JSON, Parquet, TSV, Feather
- Batch prediction with downloadable results
- Audit logs with Supabase integration
- Docker + docker-compose for local deployment
- GitHub Actions CI/CD - syntax check + model smoke test on every push
- 14 unit tests covering all core modules

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| ML | scikit-learn, SHAP, joblib |
| Backend (local) | FastAPI + WebSocket |
| Database | Supabase (PostgreSQL + RLS) |
| Alerts | Slack SDK + aiosmtplib |
| LLM | OpenAI GPT-4o / Groq Llama3 |
| CI/CD | GitHub Actions |
| Deploy | Docker, Streamlit Cloud |

---

## Quick Start

\\ash
pip install -r requirements.txt
cp .env.example .env

python scripts/generate_sample_data.py
python scripts/train_from_csv.py data/synthetic_industrial_machine_data.csv

# Terminal 1 - backend (optional for local use)
uvicorn backend.main:app --reload

# Terminal 2 - frontend
streamlit run frontend/app.py
\
## Streamlit Cloud Deploy

1. Push repo to GitHub
2. Go to share.streamlit.io
3. Entry point: frontend/app.py
4. Requirements: requirements_cloud.txt

## Docker

\\ash
docker-compose up --build
\
## Run Tests

\\ash
pytest tests/ -v
\
---

## Dataset

- 219,000 rows | 10 assets | 5 machine types | 3+ years daily data
- Breakdown rate: ~9.9%
- Sensors: bearing temp, motor temp, horizontal/vertical vibration, oil pressure, load %, shaft RPM, power consumption
- Labels: breakdown_flag (0=Normal, 1=Breakdown), wo_type (BD/PM)

## Machines Covered

| Machine | Criticality | Typical Failure Mode |
|---------|-------------|---------------------|
| CNC Lathe | A | Bearing wear, misalignment |
| Hydraulic Press | A | Seal failure, pressure loss |
| Belt Conveyor | B | Belt wear, roller failure |
| Screw Compressor | B | Valve wear, overheating |
| EOT Crane | C | Brake wear, electrical fault |

---

## Project Structure

\backend/
  ml_engine.py           Train, predict, SHAP, anomaly detection
  health_score.py        0-100 health index per asset
  ttf_predictor.py       Time-to-failure estimation
  model_registry.py      Version and compare training runs
  downtime_calculator.py INR/USD cost + ROI analysis
  iot_simulator.py       Realistic sensor degradation simulation
  llm_advisor.py         OpenAI/Groq prescriptive advice
  alerts.py              Slack + Email breakdown alerts

frontend/
  app.py                 Main Streamlit app (self-contained, 12 pages)
  charts.py              20+ interactive Plotly chart functions
  api_client.py          HTTP client for local backend

scripts/
  generate_sample_data.py  Generate 219k row synthetic dataset
  train_from_csv.py        CLI model training

tests/
  test_core.py    14 unit tests (downtime, health, TTF, file parser, registry)
\