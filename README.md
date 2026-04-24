# industrial-predmaint-ai

Full-stack Industrial Machine Predictive Maintenance System powered by FastAPI, Streamlit, and ML.

## Features

- Universal file upload (CSV, Excel, JSON, Parquet, TSV)
- Random Forest + Isolation Forest breakdown prediction
- Machine Health Score (0-100 per asset)
- Time-to-Failure estimation with degradation trend analysis
- Model Registry with version comparison
- SHAP explainability
- Live IoT sensor simulation via WebSocket
- LLM prescriptive advice (OpenAI GPT-4o / Groq Llama3)
- Downtime cost calculator (INR/USD) with ROI waterfall
- Slack + Email alerts on breakdown prediction
- Supabase audit logs
- Docker + GitHub Actions CI/CD

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your keys

python scripts/generate_sample_data.py   # generate dataset
python scripts/train_from_csv.py data/synthetic_industrial_machine_data.csv

# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

## Docker

```bash
docker-compose up --build
```

## Run Tests

```bash
pytest tests/ -v
```

## Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + WebSocket |
| Frontend | Streamlit + Plotly |
| ML | scikit-learn, SHAP |
| Database | Supabase (PostgreSQL) |
| LLM | OpenAI / Groq |
| Alerts | Slack SDK + aiosmtplib |
| CI/CD | GitHub Actions |
| Deploy | Docker + docker-compose |
