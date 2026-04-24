"""References & Citations page content."""
import streamlit as st


def render():
    st.title("📚 References & Citations")
    st.markdown("Academic papers, datasets, and libraries powering this system.")
    st.divider()

    # ── Research Papers ───────────────────────────────────────────────────────
    st.subheader("📄 Research Papers")
    papers = [
        (
            "Breiman, L. (2001)",
            "Random Forests",
            "Machine Learning, 45(1), 5–32",
            "https://doi.org/10.1023/A:1010933404324",
            "Core algorithm used for breakdown prediction in this system.",
        ),
        (
            "Lundberg, S. & Lee, S.I. (2017)",
            "A Unified Approach to Interpreting Model Predictions",
            "Advances in Neural Information Processing Systems (NeurIPS 2017)",
            "https://arxiv.org/abs/1705.07874",
            "SHAP explainability — explains why the model predicts a breakdown.",
        ),
        (
            "Liu, F.T., Ting, K.M. & Zhou, Z.H. (2008)",
            "Isolation Forest",
            "IEEE 8th International Conference on Data Mining (ICDM 2008)",
            "https://doi.org/10.1109/ICDM.2008.17",
            "Anomaly detection on sensor readings to catch outliers before breakdown.",
        ),
        (
            "Mobley, R.K. (2002)",
            "An Introduction to Predictive Maintenance",
            "Butterworth-Heinemann, 2nd Edition",
            "https://www.sciencedirect.com/book/9780750675314",
            "Foundational reference for maintenance engineering concepts.",
        ),
        (
            "Lei, Y. et al. (2018)",
            "Machinery Health Prognostics: A Systematic Review",
            "Mechanical Systems and Signal Processing, 104, 799–834",
            "https://doi.org/10.1016/j.ymssp.2017.11.016",
            "Health score and time-to-failure methodology reference.",
        ),
        (
            "Susto, G.A. et al. (2015)",
            "Machine Learning for Predictive Maintenance",
            "IEEE Transactions on Industrial Informatics, 11(3), 812–820",
            "https://doi.org/10.1109/TII.2014.2349359",
            "ML-based PdM framework that inspired the feature engineering pipeline.",
        ),
        (
            "Matzka, S. (2020)",
            "Explainable AI for Predictive Maintenance Applications",
            "3rd International Conference on AI for Industries (AI4I 2020)",
            "https://doi.org/10.24432/C5HS5C",
            "AI4I 2020 dataset — basis for synthetic data generation in this project.",
        ),
    ]

    for authors, title, journal, url, note in papers:
        with st.expander(f"{authors} — {title}"):
            st.markdown(f"**Journal/Conference:** {journal}")
            st.markdown(f"**DOI/URL:** {url}")
            st.markdown(f"**Relevance to this project:** {note}")

    st.divider()

    # ── Open Source Libraries ─────────────────────────────────────────────────
    st.subheader("📦 Open Source Libraries")
    libs = [
        ("scikit-learn", "1.4", "ML models: Random Forest, Isolation Forest",
         "Pedregosa et al., JMLR 12, pp. 2825–2830, 2011",
         "https://scikit-learn.org"),
        ("SHAP", "0.45", "Model explainability and feature importance",
         "Lundberg & Lee, NeurIPS 2017",
         "https://shap.readthedocs.io"),
        ("Streamlit", "1.35", "Interactive web application framework",
         "Streamlit Inc., 2019",
         "https://streamlit.io"),
        ("Plotly", "5.22", "Interactive visualizations",
         "Plotly Technologies Inc., 2015",
         "https://plotly.com"),
        ("pandas", "2.2", "Data manipulation and analysis",
         "McKinney, Proc. of the 9th Python in Science Conf., 2010",
         "https://pandas.pydata.org"),
        ("NumPy", "1.26", "Numerical computing",
         "Harris et al., Nature 585, 357–362, 2020",
         "https://numpy.org"),
        ("FastAPI", "0.111", "REST API and WebSocket backend",
         "Ramirez, S., 2018",
         "https://fastapi.tiangolo.com"),
        ("joblib", "1.4", "Model serialization and parallel processing",
         "Varoquaux et al., 2008",
         "https://joblib.readthedocs.io"),
    ]

    cols = st.columns(2)
    for i, (lib, ver, desc, cite, url) in enumerate(libs):
        with cols[i % 2]:
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #30363d;"
                f"border-radius:8px;padding:12px 16px;margin-bottom:10px;'>"
                f"<div style='font-size:1rem;font-weight:600;color:#58a6ff;'>{lib} v{ver}</div>"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin:4px 0;'>{desc}</div>"
                f"<div style='font-size:.75rem;color:#8b949e;'>Cite: {cite}</div>"
                f"<a href='{url}' style='font-size:.75rem;color:#3fb950;' target='_blank'>{url}</a>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📊 Dataset & Data Sources")
    st.markdown("""
The dataset used in this system is **synthetically generated** to simulate real-world
industrial machine sensor readings, inspired by:

- **NASA Prognostics Data Repository** — C-MAPSS turbofan engine degradation dataset.
  Available at: https://www.nasa.gov/intelligent-systems-division/pcoe/pcoe-data-set-repository/

- **UCI ML Repository — AI4I 2020 Predictive Maintenance Dataset**
  Matzka, S. (2020). DOI: https://doi.org/10.24432/C5HS5C

- **Kaggle — Predictive Maintenance Classification**
  https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

> The synthetic generator (`frontend/data_engine.py`) produces statistically similar
> distributions with realistic degradation physics, seasonal patterns, and failure modes.
    """)

    st.divider()

    # ── Standards ─────────────────────────────────────────────────────────────
    st.subheader("📐 Industry Standards Referenced")
    for std, title, desc in [
        ("ISO 13374", "Condition monitoring and diagnostics of machines",
         "Defines data processing and communication for machine condition monitoring."),
        ("ISO 17359", "Condition monitoring — General guidelines",
         "Framework for selecting condition monitoring methods used in health scoring."),
        ("ISO 55000", "Asset management",
         "Standard for managing physical assets — informs the downtime cost calculator."),
        ("IEC 62264", "Enterprise-control system integration",
         "Reference for integrating PdM systems with manufacturing execution systems."),
    ]:
        st.markdown(f"- **{std}** — *{title}*: {desc}")

    st.divider()

    # ── BibTeX ────────────────────────────────────────────────────────────────
    st.subheader("📝 How to Cite This Project")
    st.code(
        '@software{predmaint_ai_2025,\n'
        '  author    = {Sumedh Patil},\n'
        '  title     = {PredMaint AI: Industrial Machine Predictive Maintenance System},\n'
        '  year      = {2025},\n'
        '  url       = {https://github.com/SumedhPatil1507/industrial-predmaint-ai},\n'
        '  version   = {3.0},\n'
        '  note      = {Self-contained Streamlit app with live IoT simulation,\n'
        '               Random Forest breakdown prediction, SHAP explainability,\n'
        '               Health Score, Time-to-Failure, and What-If Simulator.}\n'
        '}',
        language="bibtex",
    )
