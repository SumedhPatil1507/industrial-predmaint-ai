# Changelog

All notable changes to PredMaint AI are documented here.

---

## [3.0.0] - 2025-04-24

### Added
- Self-contained Streamlit app — no backend server required on Streamlit Cloud
- Live IoT sensor simulation always running (no toggle needed)
- Auto-generates 2-year synthetic dataset on first load
- Auto-trains model on startup
- What-If Simulator — sensitivity analysis on any sensor
- Maintenance Schedule Generator — optimal work orders with cost estimates
- Model Drift Detection — PSI + KS test against training baseline
- PDF Report Generator — professional asset health report download
- References & Citations page with 7 academic papers and BibTeX
- About page with tech stack cards and version history
- Jupyter notebook with full EDA → model → business impact story
- CONTRIBUTING.md and CHANGELOG.md
- `runtime.txt` to pin Python 3.11 on Streamlit Cloud
- `model_is_compatible()` to detect and clear stale cached models

### Fixed
- `month`/`day_of_week` not in index error — removed date features from model
- SHAP format string error — explicit float cast on all importance values
- Top-level `import shap` crash on Python 3.14 — moved to lazy import
- `applymap` deprecated — replaced with `map` for pandas 2.x
- `background_gradient` requires matplotlib — removed dependency

### Changed
- `get_all_features()` no longer includes date-derived columns
- Model retrains automatically when incompatible cached model detected

---

## [2.0.0] - 2025-03-15

### Added
- Machine Health Score (0-100) with component radar chart
- Time-to-Failure prediction using degradation trend + hazard rate
- Model Registry — version and compare every training run
- Docker + docker-compose for local deployment
- GitHub Actions CI/CD pipeline
- 14 unit tests covering all core modules
- Fleet health treemap visualization
- TTF Gantt chart

### Changed
- Upgraded to FastAPI 0.111 + Streamlit 1.35
- Improved IoT simulator with realistic spike events

---

## [1.0.0] - 2025-02-01

### Added
- FastAPI REST backend with WebSocket live sensor stream
- Random Forest breakdown prediction
- Isolation Forest anomaly detection
- SHAP explainability
- LLM advisor (OpenAI GPT-4o / Groq Llama3)
- Slack + Email alerts on breakdown prediction
- Supabase audit logs
- Downtime cost calculator (INR/USD)
- Universal file upload (CSV, Excel, JSON, Parquet, TSV)
- Batch prediction with downloadable results
