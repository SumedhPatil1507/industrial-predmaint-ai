"""About page content."""
import streamlit as st


def render():
    st.title("ℹ️ About PredMaint AI")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
### What is PredMaint AI?

PredMaint AI is a **full-stack industrial predictive maintenance system** that uses
machine learning to predict machine breakdowns before they happen. Designed for
manufacturing plants operating CNC lathes, hydraulic presses, belt conveyors,
screw compressors, and EOT cranes.

### Problem It Solves

Unplanned machine breakdowns cost manufacturing plants **Rs. 40L to Rs. 4Cr per incident**
in production loss, emergency repairs, and missed delivery deadlines. Traditional
threshold-based monitoring catches failures too late — this system predicts failures
**3–7 days in advance**.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| Live IoT Simulation | Real-time sensor stream with degradation physics |
| Breakdown Prediction | Random Forest, ~95% AUC-ROC |
| Health Score | Weighted 0–100 index per asset with radar chart |
| Time-to-Failure | Degradation trend + hazard rate estimation |
| SHAP Explainability | Feature-level explanation for every prediction |
| What-If Simulator | Sensitivity analysis on any sensor value |
| Downtime Calculator | INR/USD cost + ROI waterfall chart |
| Model Registry | Version and compare every training run |

### Architecture

```
Data Layer      → Synthetic generator + optional file upload
ML Layer        → Random Forest + Isolation Forest + SHAP
Business Layer  → Health Score + TTF + Downtime Calculator
Presentation    → Streamlit + Plotly (100% interactive)
Deployment      → Streamlit Cloud (self-contained, no backend)
```

### Business Impact

| Metric | Value |
|--------|-------|
| Downtime reduction | 40–60% |
| Annual savings | Rs. 20L – 2Cr |
| Payback period | 3–6 months |
| Advance warning | 3–7 days before breakdown |
| Model accuracy | ~95% AUC-ROC |
        """)

    with col2:
        st.markdown("""
### Developer

**Sumedh Patil**
Industrial ML Engineer

**GitHub:**
[SumedhPatil1507](https://github.com/SumedhPatil1507/industrial-predmaint-ai)

**Live Demo:**
[Open App](https://industrial-predmaint-ai-d3sdstpce4nxkhghcq8zpk.streamlit.app)

---

### Version History

| Version | Highlights |
|---------|-----------|
| v3.0 | Self-contained, live data always on, What-If Simulator |
| v2.0 | Health Score, TTF, Model Registry, Docker, CI/CD |
| v1.0 | FastAPI backend, SHAP, LLM advisor, Supabase |

---

### License
MIT — free to use, modify, distribute.
        """)

    st.divider()
    st.subheader("🛠️ Tech Stack")
    stack = [
        ("🐍", "Python 3.12", "Core language"),
        ("⚡", "Streamlit 1.35", "Web UI"),
        ("🤖", "scikit-learn 1.4", "ML models"),
        ("🔍", "SHAP 0.45", "Explainability"),
        ("📊", "Plotly 5.22", "Charts"),
        ("🐼", "pandas 2.2", "Data processing"),
        ("🔢", "NumPy 1.26", "Numerics"),
        ("💾", "joblib 1.4", "Model persistence"),
        ("🚀", "FastAPI 0.111", "REST API (local)"),
        ("🗄️", "Supabase", "Database (optional)"),
        ("🐳", "Docker", "Containerization"),
        ("⚙️", "GitHub Actions", "CI/CD"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(stack):
        with cols[i % 4]:
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #30363d;"
                f"border-radius:8px;padding:10px;text-align:center;margin-bottom:8px;'>"
                f"<div style='font-size:1.3rem;'>{icon}</div>"
                f"<div style='font-size:.82rem;font-weight:600;color:#58a6ff;'>{name}</div>"
                f"<div style='font-size:.7rem;color:#8b949e;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#8b949e;font-size:.8rem;padding:20px;'>"
        "PredMaint AI v3.0 &nbsp;·&nbsp; Built by Sumedh Patil &nbsp;·&nbsp; MIT License &nbsp;·&nbsp; 2025<br>"
        "<a href='https://github.com/SumedhPatil1507/industrial-predmaint-ai' style='color:#58a6ff;'>"
        "github.com/SumedhPatil1507/industrial-predmaint-ai</a>"
        "</div>",
        unsafe_allow_html=True,
    )
