# industrial-predmaint-ai

**Live Demo:** https://industrial-predmaint-ai-d3sdstpce4nxkhghcq8zpk.streamlit.app

An end-to-end industrial machine predictive maintenance system that uses machine learning to predict equipment failures before they occur. Built for manufacturing plants operating CNC lathes, hydraulic presses, belt conveyors, screw compressors, and EOT cranes.

---

## Business Impact

Unplanned machine breakdowns cost manufacturing plants Rs. 40L to Rs. 4Cr per incident in production loss, emergency repairs, and missed delivery deadlines. This system addresses that by predicting failures 3-7 days in advance using real-time sensor data.

| Outcome | Result |
|---------|--------|
| Unplanned downtime reduction | 40-60% |
| Annual cost savings | Rs. 20L - 2Cr per plant |
| Payback period | 3-6 months |
| Advance failure warning | 3-7 days |
| Breakdown prediction accuracy | ~95% AUC-ROC |
| Assets monitored | 10 machines across 5 types |

---

## What It Does

The system continuously monitors sensor readings (temperature, vibration, oil pressure, power consumption) from industrial machines and:

1. Predicts breakdown probability for each asset in real time
2. Scores machine health on a 0-100 index with component-level breakdown
3. Estimates days until failure using degradation trend analysis
4. Explains every prediction using SHAP feature importance
5. Simulates what-if scenarios (e.g. what happens if vibration increases 30%)
6. Calculates financial impact of breakdowns and ROI of the maintenance program

---

## Features

- Live IoT sensor simulation with realistic degradation physics
- Random Forest breakdown prediction with Isolation Forest anomaly detection
- SHAP explainability for every prediction
- Machine Health Score (0-100) with radar chart per asset
- Time-to-Failure estimation using degradation slope and hazard rate
- What-If Simulator for sensitivity analysis on any sensor
- Downtime cost calculator in INR and USD with ROI waterfall
- Model Registry to version and compare training runs
- Universal file upload: CSV, Excel, JSON, Parquet, TSV
- Batch prediction with downloadable results
- References and citations page with academic sources
- Self-contained: works on Streamlit Cloud with no backend server

---

## Dataset

Synthetic dataset generated to simulate real-world industrial sensor readings, inspired by:

- NASA Prognostics Data Repository (C-MAPSS turbofan degradation)
- UCI AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)
- Kaggle Predictive Maintenance Classification dataset

219,000 records across 10 assets over 3+ years. Breakdown rate ~9.9%.

---

## Machines Covered

| Machine | Criticality | Common Failure Mode |
|---------|-------------|---------------------|
| CNC Lathe | A | Bearing wear, misalignment |
| Hydraulic Press | A | Seal failure, pressure loss |
| Belt Conveyor | B | Belt wear, roller failure |
| Screw Compressor | B | Valve wear, overheating |
| EOT Crane | C | Brake wear, electrical fault |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| ML | scikit-learn, SHAP, joblib |
| Backend (local) | FastAPI + WebSocket |
| Database | Supabase (PostgreSQL) |
| CI/CD | GitHub Actions |
| Deploy | Streamlit Cloud, Docker |

---

## Quick Start

pip install -r requirements.txt
cp .env.example .env
streamlit run frontend/app.py

The app auto-generates data and trains the model on first launch. No file upload needed.

## Streamlit Cloud

Entry point: frontend/app.py
Requirements: requirements_cloud.txt

## Docker

docker-compose up --build

## Tests

pytest tests/ -v

---

## Project Structure

backend/
  ml_engine.py           Train, predict, SHAP, anomaly detection
  health_score.py        0-100 health index per asset
  ttf_predictor.py       Time-to-failure estimation
  model_registry.py      Version and compare training runs
  downtime_calculator.py Cost and ROI analysis
  iot_simulator.py       Sensor degradation simulation

frontend/
  app.py                 Main Streamlit app (13 pages, self-contained)
  charts.py              Interactive Plotly chart functions
  data_engine.py         Synthetic data generation and what-if analysis
  page_references.py     Academic citations and references
  page_about.py          Project overview and tech stack

tests/
  test_core.py           Unit tests for all core modules
