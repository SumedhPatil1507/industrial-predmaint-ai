# industrial-predmaint-ai

**Live Demo:** https://industrial-predmaint-ai-d3sdstpce4nxkhghcq8zpk.streamlit.app

End-to-end industrial machine predictive maintenance system. Predicts equipment failures before they occur using real-time sensor data, physics-based validation, and machine learning.

---

## Business Impact

| Outcome | Result |
|---------|--------|
| Unplanned downtime reduction | 40-60% |
| Annual cost savings | Rs. 20L - 2Cr per plant |
| Payback period | 3-6 months |
| Advance failure warning | 3-7 days |
| Breakdown prediction accuracy | ~95% AUC-ROC |

---

## Architecture

```
+---------------------------+     +-----------------------------+
|   Industrial Protocols    |     |   Hardware Stress-Testing   |
|                           |     |   Harness (Simulator)       |
|  MQTT  /ingest/mqtt       +---->+   StressTestHarness         |
|  OPC-UA /ingest/opcua     |     |   Modes: NORMAL / DEGRADING |
|  REST   /predict          |     |   SPIKE / MALFUNCTION /     |
|  Batch  /upload-predict   |     |   FAILURE                   |
+---------------------------+     +-----------------------------+
              |                                |
              v                                v
+------------------------------------------+
|   Sensor Validation Layer                |
|   (backend/sensor_validator.py)          |
|                                          |
|   Pydantic type checks                   |
|   Physical hard limits per machine       |
|   Cross-sensor physics rules             |
|   Rate-of-change impossibility check     |
|                                          |
|   SENSOR_MALFUNCTION -> suppresses ML    |
|   Valid reading -> ML pipeline           |
+------------------------------------------+
              |
              v
+------------------------------------------+
|   ML Backend (FastAPI)                   |
|   backend/main.py                        |
|                                          |
|   Random Forest breakdown prediction     |
|   Isolation Forest anomaly detection     |
|   Health Score (0-100 per asset)         |
|   Time-to-Failure estimation             |
|   SHAP explainability                    |
|                                          |
|   WS /ws/scored-stream                   |
|   Pushes pre-calculated scores           |
+------------------------------------------+
              |
              v
+------------------------------------------+
|   Streamlit Frontend (display only)      |
|   frontend/app.py                        |
|                                          |
|   Zero ML logic in frontend              |
|   Receives scored data from backend      |
|   18 interactive pages                   |
|   100% Plotly charts                     |
+------------------------------------------+
```

---

## Features

### Sensor Validation (New)
- Pydantic models with physical boundary enforcement
- Cross-sensor physics rules (e.g. Screw Compressor: temp surge must correlate with pressure rise)
- Rate-of-change impossibility detection (50%+ change in one tick = sensor fault)
- Returns SENSOR_MALFUNCTION instead of false breakdown warning
- Suppresses ML prediction on invalid readings

### Industrial Protocol Ingestion (New)
- MQTT payload normaliser (V1 compact schema + V2 verbose schema + direct fields)
- OPC-UA NodeId/Value frame parser with StatusCode quality filtering
- Unified ingestion router: `POST /ingest/mqtt`, `POST /ingest/opcua`
- All ingestion paths run through validation before ML

### Hardware Stress-Testing Harness (New)
- Repositioned IoT simulator as explicit test harness
- Modes: NORMAL, DEGRADING, SPIKE, MALFUNCTION, FAILURE
- MALFUNCTION mode injects cross-sensor inconsistencies to test validator
- Used for pipeline validation and load testing

### High-Throughput WebSocket
- `WS /ws/scored-stream` pushes pre-calculated anomaly scores, health indices, TTF
- Frontend is a pure display layer — zero ML computation in Streamlit
- `WS /ws/live-sensors` retained for backward compatibility

### Machine Learning
- Random Forest (~95% AUC-ROC) with class-balanced training
- Isolation Forest anomaly detection on normal-state data
- SHAP explainability for every prediction
- Model Drift Detection (PSI + mean shift)
- Model Registry with version comparison

### Business Tools
- OEE Calculator (Availability x Performance x Quality)
- Executive Summary with shift handover report
- Maintenance Schedule Generator with work orders
- Downtime Cost Calculator (INR/USD + ROI waterfall)
- PDF Report Export per asset
- What-If Simulator for sensor sensitivity analysis

### Infrastructure
- Self-contained on Streamlit Cloud (no backend needed)
- FastAPI backend for local/production deployment
- Docker + docker-compose
- GitHub Actions CI/CD
- Supabase audit logs (optional)
- Slack + Email alerts

---

## Dataset

Synthetic dataset inspired by NASA C-MAPSS, UCI AI4I 2020, and Kaggle PdM datasets.
219,000 records, 10 assets, 5 machine types, 3+ years. Breakdown rate ~9.9%.

---

## Machines

| Machine | Criticality | Failure Mode |
|---------|-------------|--------------|
| CNC Lathe | A | Bearing wear, misalignment |
| Hydraulic Press | A | Seal failure, pressure loss |
| Belt Conveyor | B | Belt wear, roller failure |
| Screw Compressor | B | Valve wear, overheating |
| EOT Crane | C | Brake wear, electrical fault |

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run frontend/app.py
```

App auto-generates data and trains the model on first launch.

## Local Backend

```bash
uvicorn backend.main:app --reload
```

## Docker

```bash
docker-compose up --build
```

## Tests

```bash
pytest tests/ -v
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.35 + Plotly 5.22 |
| Validation | Pydantic 2.7 |
| ML | scikit-learn 1.4, SHAP 0.45 |
| Backend | FastAPI 0.111 + WebSocket |
| Protocols | MQTT, OPC-UA (ingestion layer) |
| Database | Supabase (PostgreSQL) |
| CI/CD | GitHub Actions |
| Deploy | Streamlit Cloud, Docker |
| Python | 3.11 |

## Edge‑Gateway Architecture

The system now follows a **Local Edge‑Gateway‑First** design. All sensor ingestion, OPC‑UA write‑back, and database operations run on‑premises behind the factory firewall. TimescaleDB provides a high‑performance time‑series store on the edge node.

### OT Write‑Back Service (`backend/ot_writeback.py`)

- Exposes `async trigger_emergency_stop(asset_id: str, reason: str)`.
- Maintains a pooled OPC‑UA client connection to the PLC endpoint.
- Sends a 1‑second heartbeat to verify connectivity.
- Performs exponential‑backoff retries on network failures.
- Logs critical failures with `logging.critical`.

### Ingestion Worker (`backend/ingestion_worker.py`)

- Subscribes to the local Mosquitto broker on `factory/sensors/#`.
- Parses incoming JSON payloads and validates them via `sensor_validator.py`.
- Writes validated readings in batches to TimescaleDB using the `timescaledb` helper functions.
- Gracefully pauses MQTT consumption if the DB connection is lost, preserving message offsets.

## Database Migrations (`backend/migrations.sql`)

- Creates `predictions`, `audit_logs`, and `alerts` tables with `asset_id` and indexed `timestamp`.
- Converts `predictions` and `audit_logs` to TimescaleDB hypertables.
- Adds a retention policy that compresses data older than **7 days** to conserve edge‑gateway storage.

## Updating the Repository

```bash
git add backend/ot_writeback.py backend/ingestion_worker.py backend/migrations.sql README.md
git commit -m "Add OT write‑back service, MQTT ingestion worker, and database migrations for edge‑gateway architecture"
git push origin main
```

---

## References

- Breiman (2001). Random Forests. Machine Learning 45(1).
- Lundberg & Lee (2017). SHAP. NeurIPS.
- Liu et al. (2008). Isolation Forest. IEEE ICDM.
- Matzka (2020). AI4I 2020 PdM Dataset. UCI ML Repository.

Full citations in the app References page.

---

## License

MIT License.
