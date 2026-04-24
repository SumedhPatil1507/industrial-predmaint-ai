# 🔧 Industrial Machine Predictive Maintenance System

A full-stack, production-grade predictive maintenance platform built for real-world manufacturing environments.

## Stack
- **Backend**: FastAPI + WebSocket (live IoT simulation)
- **Database**: Supabase (PostgreSQL + Realtime + Auth)
- **Frontend**: Streamlit (interactive Plotly charts)
- **ML**: Random Forest, Isolation Forest, SHAP
- **LLM**: Prescriptive maintenance via OpenAI/Groq
- **Alerts**: Slack + Email (SMTP)

## Features
- 📁 Upload any file type (CSV, Excel, JSON, Parquet)
- 📊 Fully interactive Plotly charts (no static plots)
- 🔴 Live IoT sensor simulation via WebSocket
- 🤖 LLM-powered prescriptive maintenance recommendations
- 💰 Downtime cost calculator (INR/USD)
- 🚨 Automated Slack/Email alerts on breakdown prediction
- 📋 Legal audit logs in Supabase
- 🔐 Supabase Auth (JWT)

## Quick Start
```bash
pip install -r requirements.txt
# Set .env variables (see .env.example)
uvicorn backend.main:app --reload        # Terminal 1
streamlit run frontend/app.py            # Terminal 2
```
