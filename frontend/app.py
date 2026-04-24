"""
Industrial Machine Predictive Maintenance System
Premium Streamlit UI - fully interactive, live data, dark theme
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import time
import sys
import os
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend import api_client, charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredMaint AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.main { background: #0d1117; }
.block-container { padding-top: 1.5rem; }

.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.kpi-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 0.85rem; color: #3fb950; margin-top: 4px; }

.alert-critical {
    background: linear-gradient(135deg, #3d0000, #5a0a0a);
    border: 1px solid #e74c3c; border-radius: 8px;
    padding: 12px 16px; color: #ff6b6b; font-weight: 600;
    animation: pulse 2s infinite;
}
.alert-warning {
    background: linear-gradient(135deg, #2d1a00, #3d2500);
    border: 1px solid #e67e22; border-radius: 8px;
    padding: 12px 16px; color: #ffa94d;
}
.alert-ok {
    background: linear-gradient(135deg, #001a0d, #002b15);
    border: 1px solid #2ecc71; border-radius: 8px;
    padding: 12px 16px; color: #69db7c;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(231,76,60,0); }
}

.status-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
}
.badge-critical { background: #3d0000; color: #ff6b6b; border: 1px solid #e74c3c; }
.badge-high     { background: #2d1500; color: #ffa94d; border: 1px solid #e67e22; }
.badge-medium   { background: #2d2500; color: #ffd43b; border: 1px solid #f1c40f; }
.badge-low      { background: #001a0d; color: #69db7c; border: 1px solid #2ecc71; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #58a6ff;
    border-bottom: 1px solid #21262d; padding-bottom: 8px; margin-bottom: 16px;
}
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "df": None, "train_metrics": None,
    "live_history": defaultdict(lambda: deque(maxlen=80)),
    "live_all": {},
    "live_pred_history": defaultdict(lambda: deque(maxlen=80)),
    "live_running": False,
    "ws_thread": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 PredMaint AI")
    st.caption("Industrial Predictive Maintenance")
    st.divider()

    health = api_client.health_check()
    api_ok = health.get("status") == "ok"
    model_ok = health.get("model_ready", False)

    col_a, col_b = st.columns(2)
    col_a.markdown(f"{'🟢' if api_ok else '🔴'} **API**")
    col_b.markdown(f"{'✅' if model_ok else '⚠️'} **Model**")
    st.divider()

    page = st.radio("", [
        "🏠  Dashboard",
        "📁  Upload & Train",
        "📊  EDA Explorer",
        "🤖  Predict Breakdown",
        "🔴  Live IoT Monitor",
        "🔍  SHAP Explainability",
        "❤️  Health Score",
        "⏱️  Time-to-Failure",
        "📈  Model Registry",
        "💰  Downtime Calculator",
        "🧠  AI Advisor",
        "📋  Audit Logs",
    ], label_visibility="collapsed")

    st.divider()
    st.caption("v2.0 | Python 3.12 | FastAPI + Streamlit")


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "🏠  Dashboard":
    st.markdown("## 🔧 Industrial Machine Predictive Maintenance")
    st.markdown("AI-powered breakdown prediction · anomaly detection · prescriptive maintenance")
    st.divider()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("219,000", "Dataset Records", "3+ years"),
        ("5", "Machine Types", "10 assets"),
        ("~9.9%", "Breakdown Rate", "Class imbalance"),
        ("40-60%", "Downtime Reduction", "With PdM"),
        ("12", "Features", "Sensor + engineered"),
    ]
    for col, (val, label, delta) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown('<div class="section-header">System Capabilities</div>', unsafe_allow_html=True)
        features = [
            ("📁", "Universal file upload", "CSV, Excel, JSON, Parquet, TSV"),
            ("🤖", "ML Models", "Random Forest + Isolation Forest"),
            ("🔍", "SHAP Explainability", "Feature importance + beeswarm"),
            ("🔴", "Live IoT Simulation", "Real-time WebSocket stream"),
            ("🧠", "LLM Advisor", "GPT-4o / Llama3 recommendations"),
            ("💰", "Downtime Calculator", "INR/USD cost + ROI waterfall"),
            ("❤️", "Health Score", "0-100 per asset with radar chart"),
            ("⏱️", "Time-to-Failure", "Days until predicted breakdown"),
            ("📈", "Model Registry", "Version + compare training runs"),
            ("🚨", "Alerts", "Slack + Email on breakdown"),
        ]
        for icon, name, desc in features:
            st.markdown(f"**{icon} {name}** — {desc}")

    with col2:
        st.markdown('<div class="section-header">Supported Machines</div>', unsafe_allow_html=True)
        machines = [
            ("🔩", "CNC Lathe", "High-precision cutting"),
            ("⚙️", "Hydraulic Press", "High-force forming"),
            ("📦", "Belt Conveyor", "Material transport"),
            ("💨", "Screw Compressor", "Compressed air supply"),
            ("🏗️", "EOT Crane", "Heavy lifting"),
        ]
        for icon, name, desc in machines:
            st.markdown(f"{icon} **{name}** — {desc}")

    with col3:
        st.markdown('<div class="section-header">Quick Start</div>', unsafe_allow_html=True)
        st.markdown("""
1. **Upload & Train** → upload CSV
2. **EDA Explorer** → visualize data
3. **Predict** → single or batch
4. **Live IoT** → real-time monitor
        """)

    st.divider()
    if not model_ok:
        st.markdown('<div class="alert-warning">⚠️ Model not trained yet. Go to Upload & Train to get started.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-ok">✅ Model ready. All features unlocked.</div>',
                    unsafe_allow_html=True)


# =============================================================================
# UPLOAD & TRAIN
# =============================================================================
elif page == "📁  Upload & Train":
    st.title("📁 Upload Dataset & Train Model")
    st.markdown("Supports **CSV, Excel, JSON, Parquet, TSV, Feather**")

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "feather"],
    )

    if uploaded:
        file_bytes = uploaded.read()
        st.success(f"File received: `{uploaded.name}` ({len(file_bytes)/1024:.1f} KB)")

        col1, col2 = st.columns(2)
        with col1:
            action = st.radio("Action", ["Train Model", "Batch Predict"])
        with col2:
            st.info("**Train Model** – needs `breakdown_flag` column\n\n"
                    "**Batch Predict** – needs sensor columns only")

        if st.button("Execute", type="primary"):
            with st.spinner("Processing..."):
                try:
                    if action == "Train Model":
                        result = api_client.upload_and_train(file_bytes, uploaded.name)
                        st.session_state.train_metrics = result.get("metrics", {})
                        st.success("Model trained successfully!")
                        m = result["metrics"]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
                        c2.metric("AUC-ROC", f"{m.get('auc', 0):.3f}")
                        c3.metric("Precision", f"{m.get('precision_breakdown', 0):.3f}")
                        c4.metric("Recall", f"{m.get('recall_breakdown', 0):.3f}")
                        with st.expander("Full metrics"):
                            st.json(m)
                    else:
                        result = api_client.upload_predict(file_bytes, uploaded.name)
                        st.success(f"Predictions complete! {result['total']} rows processed.")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Rows", result["total"])
                        c2.metric("Breakdowns Predicted", result["breakdowns_predicted"])
                        c3.metric("Critical Risk", result["critical"])
                        df_res = pd.DataFrame(result["results"])
                        st.dataframe(df_res, use_container_width=True)
                        st.plotly_chart(charts.prediction_results_chart(result["results"]),
                                        use_container_width=True)
                        st.download_button("Download Results CSV",
                                           df_res.to_csv(index=False),
                                           "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

        try:
            from backend.file_parser import parse_upload
            df = parse_upload(uploaded.name, file_bytes)
            st.session_state.df = df
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            c3.metric("Missing", int(df.isnull().sum().sum()))
            c4.metric("Breakdown Rate",
                      f"{df['breakdown_flag'].mean():.1%}" if "breakdown_flag" in df.columns else "N/A")
        except Exception:
            pass


# =============================================================================
# EDA EXPLORER
# =============================================================================
elif page == "📊  EDA Explorer":
    st.title("📊 Interactive EDA Explorer")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if "breakdown_flag" in df.columns and "is_breakdown" not in df.columns:
        df["is_breakdown"] = df["breakdown_flag"].map({0: "Normal", 1: "Breakdown"})
    if "month" not in df.columns and "transaction_date" in df.columns:
        df["month"] = df["transaction_date"].dt.month

    # Filters
    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if "machine_type" in df.columns:
                machines = st.multiselect("Machine Type", df["machine_type"].unique(),
                                          default=list(df["machine_type"].unique()))
                df = df[df["machine_type"].isin(machines)]
        with fc2:
            if "transaction_date" in df.columns:
                min_d = df["transaction_date"].min().date()
                max_d = df["transaction_date"].max().date()
                date_range = st.date_input("Date Range", [min_d, max_d])
                if len(date_range) == 2:
                    df = df[(df["transaction_date"].dt.date >= date_range[0]) &
                            (df["transaction_date"].dt.date <= date_range[1])]
        with fc3:
            if "is_breakdown" in df.columns:
                bd_filter = st.multiselect("Status", ["Normal", "Breakdown"],
                                           default=["Normal", "Breakdown"])
                df = df[df["is_breakdown"].isin(bd_filter)]

    st.caption(f"Showing {len(df):,} rows after filters")

    tabs = st.tabs(["Distributions", "Trends", "Categorical", "Boxplots",
                    "Violins", "Correlation", "3D Scatter", "Scatter Matrix",
                    "Rolling Anomaly", "Monthly"])

    with tabs[0]:
        st.plotly_chart(charts.histogram_grid(df), use_container_width=True)

    with tabs[1]:
        if "transaction_date" in df.columns:
            st.plotly_chart(charts.lineplot_trends(df), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(charts.pie_charts(df), use_container_width=True)
        if "machine_type" in df.columns and "is_breakdown" in df.columns:
            st.plotly_chart(charts.breakdown_by_machine(df), use_container_width=True)

    with tabs[3]:
        if "is_breakdown" in df.columns:
            st.plotly_chart(charts.boxplot_sensors(df), use_container_width=True)

    with tabs[4]:
        if "is_breakdown" in df.columns:
            st.plotly_chart(charts.violin_sensors(df), use_container_width=True)

    with tabs[5]:
        st.plotly_chart(charts.correlation_heatmap(df), use_container_width=True)

    with tabs[6]:
        if all(c in df.columns for c in ["temp_bearing_degC", "vibration_h_mms",
                                          "power_consumption_kw", "is_breakdown"]):
            st.plotly_chart(charts.scatter_3d(df), use_container_width=True)

    with tabs[7]:
        if "is_breakdown" in df.columns:
            sensor_cols = ["temp_bearing_degC", "temp_motor_degC",
                           "vibration_h_mms", "oil_pressure_bar", "power_consumption_kw"]
            available = [c for c in sensor_cols if c in df.columns]
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", available, index=0)
            y_col = c2.selectbox("Y axis", available, index=2)
            st.plotly_chart(charts.sensor_vs_breakdown_scatter(df, x_col, y_col),
                            use_container_width=True)

    with tabs[8]:
        if "transaction_date" in df.columns:
            sensor_cols = [c for c in ["vibration_h_mms", "temp_bearing_degC",
                                        "oil_pressure_bar"] if c in df.columns]
            col = st.selectbox("Sensor", sensor_cols)
            st.plotly_chart(charts.rolling_anomaly_chart(df, col), use_container_width=True)

    with tabs[9]:
        if "breakdown_flag" in df.columns:
            st.plotly_chart(charts.monthly_breakdown_rate(df), use_container_width=True)


# =============================================================================
# PREDICT BREAKDOWN
# =============================================================================
elif page == "🤖  Predict Breakdown":
    st.title("🤖 Predict Machine Breakdown")

    if not model_ok:
        st.warning("Train a model first.")
        st.stop()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Predict"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            asset_tag = st.text_input("Asset Tag", "CNC-001")
            machine_type = st.selectbox("Machine Type",
                ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
            temp_bearing = st.slider("Bearing Temp (C)", 30.0, 120.0, 65.0, 0.5)
            temp_motor = st.slider("Motor Temp (C)", 30.0, 130.0, 75.0, 0.5)
            vibration_h = st.slider("H-Vibration (mm/s)", 0.0, 15.0, 2.5, 0.1)
            vibration_v = st.slider("V-Vibration (mm/s)", 0.0, 12.0, 2.0, 0.1)
        with col2:
            oil_pressure = st.slider("Oil Pressure (bar)", 0.0, 150.0, 5.5, 0.1)
            load_pct = st.slider("Load (%)", 0.0, 100.0, 65.0, 1.0)
            shaft_rpm = st.slider("Shaft RPM", 0.0, 4000.0, 1200.0, 10.0)
            power_kw = st.slider("Power (kW)", 0.0, 100.0, 25.0, 0.5)

        payload = {
            "asset_tag": asset_tag, "machine_type": machine_type,
            "temp_bearing_degC": temp_bearing, "temp_motor_degC": temp_motor,
            "vibration_h_mms": vibration_h, "vibration_v_mms": vibration_v,
            "oil_pressure_bar": oil_pressure, "load_pct": load_pct,
            "shaft_rpm": shaft_rpm, "power_consumption_kw": power_kw,
        }

        if st.button("Predict", type="primary"):
            with st.spinner("Running prediction..."):
                try:
                    result = api_client.predict_single(payload)
                    prob = result["probability"]
                    risk = result["risk_level"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prediction", "BREAKDOWN" if result["prediction"] else "NORMAL")
                    c2.metric("Probability", f"{prob:.1%}")
                    c3.metric("Risk Level", risk)
                    c4.metric("Anomaly Score", f"{result.get('anomaly_score', 0):.3f}")

                    st.plotly_chart(charts.risk_gauge(prob), use_container_width=True)

                    if risk == "CRITICAL":
                        st.markdown('<div class="alert-critical">CRITICAL RISK – Alerts sent to Slack/Email. Immediate inspection required.</div>',
                                    unsafe_allow_html=True)
                    elif risk == "HIGH":
                        st.markdown('<div class="alert-warning">HIGH RISK – Schedule inspection within 24 hours.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-ok">Normal / Low risk – Continue scheduled maintenance.</div>',
                                    unsafe_allow_html=True)

                    with st.expander("Full prediction details"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        uploaded = st.file_uploader("Upload file for batch prediction",
                                    type=["csv", "xlsx", "json", "parquet"], key="batch_pred")
        if uploaded:
            file_bytes = uploaded.read()
            if st.button("Run Batch Predict", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        result = api_client.upload_predict(file_bytes, uploaded.name)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total", result["total"])
                        c2.metric("Breakdowns", result["breakdowns_predicted"])
                        c3.metric("Critical", result["critical"])
                        df_res = pd.DataFrame(result["results"])
                        st.dataframe(df_res, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(charts.prediction_results_chart(result["results"]),
                                            use_container_width=True)
                        st.download_button("Download CSV", df_res.to_csv(index=False),
                                           "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")


# =============================================================================
# LIVE IoT MONITOR  (premium: all-asset fleet view + live prediction)
# =============================================================================
elif page == "🔴  Live IoT Monitor":
    st.title("🔴 Live IoT Sensor Monitor")
    st.markdown("Real-time WebSocket stream with live breakdown prediction per asset.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        selected_asset = st.selectbox("Asset", ["CNC-001", "HYD-001", "BLT-001", "CMP-001", "EOT-001"])
    with col_ctrl2:
        refresh_rate = st.selectbox("Refresh (s)", [1, 2, 5], index=0)
    with col_ctrl3:
        run = st.toggle("Start Live Feed", value=False)

    ph_fleet = st.empty()
    ph_kpis = st.empty()
    ph_chart = st.empty()
    ph_pred = st.empty()
    ph_table = st.empty()

    if run:
        try:
            import websocket as ws_lib
        except ImportError:
            st.error("Install websocket-client: pip install websocket-client")
            st.stop()

        history = defaultdict(lambda: deque(maxlen=80))
        all_latest = {}
        pred_history = deque(maxlen=80)
        lock = threading.Lock()

        def on_message(ws, message):
            readings = json.loads(message)
            with lock:
                for r in readings:
                    all_latest[r["asset_tag"]] = r
                    if r["asset_tag"] == selected_asset:
                        for k, v in r.items():
                            if isinstance(v, (int, float)):
                                history[k].append(v)

        def run_ws():
            wsa = ws_lib.WebSocketApp(
                "ws://localhost:8000/ws/live-sensors",
                on_message=on_message,
                on_error=lambda ws, e: None,
            )
            wsa.run_forever()

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()
        time.sleep(0.5)

        for _ in range(600):
            time.sleep(refresh_rate)
            with lock:
                snap = dict(all_latest.get(selected_asset, {}))
                hist_snap = {k: list(v) for k, v in history.items()}
                fleet_snap = list(all_latest.values())

            if not snap:
                continue

            # Live prediction
            if model_ok and snap:
                try:
                    pred = api_client.predict_single({
                        "asset_tag": snap.get("asset_tag", selected_asset),
                        "machine_type": snap.get("machine_type", "CNC Lathe"),
                        "temp_bearing_degC": snap.get("temp_bearing_degC", 65),
                        "temp_motor_degC": snap.get("temp_motor_degC", 75),
                        "vibration_h_mms": snap.get("vibration_h_mms", 2.5),
                        "vibration_v_mms": snap.get("vibration_v_mms", 2.0),
                        "oil_pressure_bar": snap.get("oil_pressure_bar", 5.5),
                        "load_pct": snap.get("load_pct", 65),
                        "shaft_rpm": snap.get("shaft_rpm", 1200),
                        "power_consumption_kw": snap.get("power_consumption_kw", 25),
                    })
                    pred_history.append({"probability": pred["probability"],
                                         "risk_level": pred["risk_level"]})
                except Exception:
                    pred = {}
            else:
                pred = {}

            # Fleet overview
            with ph_fleet.container():
                st.plotly_chart(charts.live_all_assets_chart(fleet_snap),
                                use_container_width=True)

            # KPIs
            with ph_kpis.container():
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Bearing Temp", f"{snap.get('temp_bearing_degC', 0):.1f} C")
                k2.metric("H-Vibration", f"{snap.get('vibration_h_mms', 0):.2f} mm/s")
                k3.metric("Oil Pressure", f"{snap.get('oil_pressure_bar', 0):.1f} bar")
                k4.metric("Power", f"{snap.get('power_consumption_kw', 0):.1f} kW")
                k5.metric("Degradation", f"{snap.get('degradation_index', 0):.1%}")

            # Sensor chart
            with ph_chart.container():
                col_chart, col_gauge = st.columns([3, 1])
                with col_chart:
                    st.plotly_chart(charts.live_sensor_chart(hist_snap, selected_asset),
                                    use_container_width=True)
                with col_gauge:
                    st.plotly_chart(charts.degradation_gauge(
                        snap.get("degradation_index", 0), selected_asset),
                        use_container_width=True)
                    if pred:
                        st.plotly_chart(charts.risk_gauge(pred.get("probability", 0)),
                                        use_container_width=True)

            # Prediction trend
            with ph_pred.container():
                if pred_history:
                    st.plotly_chart(charts.live_prediction_timeline(list(pred_history)),
                                    use_container_width=True)
                if pred:
                    risk = pred.get("risk_level", "LOW")
                    css = {"CRITICAL": "alert-critical", "HIGH": "alert-warning",
                           "MEDIUM": "alert-warning", "LOW": "alert-ok"}.get(risk, "alert-ok")
                    st.markdown(
                        f'<div class="{css}">Live Prediction: {risk} | '
                        f'Probability: {pred.get("probability", 0):.1%}</div>',
                        unsafe_allow_html=True)

            # Raw data table
            with ph_table.container():
                with st.expander("Raw sensor data"):
                    st.dataframe(pd.DataFrame([snap]), use_container_width=True)

    else:
        st.info("Toggle **Start Live Feed** to begin.\n\nMake sure backend is running: `uvicorn backend.main:app --reload`")
        st.plotly_chart(charts.live_all_assets_chart([]), use_container_width=True)


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================
elif page == "🔍  SHAP Explainability":
    st.title("🔍 SHAP Model Explainability")
    st.markdown("Understand **why** the model predicts a breakdown for each feature.")

    if not model_ok:
        st.warning("Train a model first.")
        st.stop()

    uploaded = st.file_uploader("Upload dataset for SHAP analysis",
                                type=["csv", "xlsx", "json", "parquet"])
    if uploaded:
        file_bytes = uploaded.read()
        max_rows = st.slider("Max rows for SHAP (more = slower)", 100, 1000, 300, 50)
        if st.button("Compute SHAP", type="primary"):
            with st.spinner("Computing SHAP values..."):
                try:
                    result = api_client.get_shap(file_bytes, uploaded.name)
                    st.success("SHAP analysis complete!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(charts.feature_importance_chart(result["feature_importance"]),
                                        use_container_width=True)
                    with col2:
                        # Waterfall for top feature
                        imp = result["feature_importance"]
                        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=[v for _, v in top], y=[k for k, _ in top],
                            orientation="h", marker_color="#e74c3c",
                        ))
                        fig.update_layout(title="Top 5 Risk Drivers",
                                          template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader("SHAP Values Heatmap (first 20 rows)")
                    shap_df = pd.DataFrame(
                        result["shap_values"][:20], columns=result["feature_names"])
                    st.dataframe(
                        shap_df.style.background_gradient(cmap="RdYlGn", axis=None),
                        use_container_width=True)

                    with st.expander("Feature importance dict"):
                        st.json(result["feature_importance"])
                except Exception as e:
                    st.error(f"Error: {e}")


# =============================================================================
# HEALTH SCORE
# =============================================================================
elif page == "❤️  Health Score":
    st.title("❤️ Machine Health Score")
    st.markdown("Weighted 0-100 health index per asset. Radar chart shows which component is degrading.")

    tab1, tab2 = st.tabs(["Single Asset", "Fleet Upload"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            asset_tag = st.text_input("Asset Tag", "CNC-001", key="hs_tag")
            machine_type = st.selectbox("Machine Type",
                ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"],
                key="hs_mtype")
            temp_bearing = st.slider("Bearing Temp (C)", 30.0, 120.0, 68.0, key="hs_tb")
            temp_motor = st.slider("Motor Temp (C)", 30.0, 130.0, 78.0, key="hs_tm")
            vibration_h = st.slider("H-Vibration (mm/s)", 0.0, 15.0, 3.2, key="hs_vh")
        with col2:
            vibration_v = st.slider("V-Vibration (mm/s)", 0.0, 12.0, 2.5, key="hs_vv")
            oil_pressure = st.slider("Oil Pressure (bar)", 0.0, 150.0, 5.2, key="hs_op")
            load_pct = st.slider("Load (%)", 0.0, 100.0, 65.0, key="hs_lp")
            shaft_rpm = st.slider("Shaft RPM", 0.0, 4000.0, 1200.0, key="hs_rpm")
            power_kw = st.slider("Power (kW)", 0.0, 100.0, 24.0, key="hs_pw")

        if st.button("Compute Health Score", type="primary"):
            payload = {
                "asset_tag": asset_tag, "machine_type": machine_type,
                "temp_bearing_degC": temp_bearing, "temp_motor_degC": temp_motor,
                "vibration_h_mms": vibration_h, "vibration_v_mms": vibration_v,
                "oil_pressure_bar": oil_pressure, "load_pct": load_pct,
                "shaft_rpm": shaft_rpm, "power_consumption_kw": power_kw,
            }
            try:
                from backend.health_score import compute_health_score
                hs = compute_health_score(payload)
                color = "#2ecc71" if hs.health_score >= 75 else "#f39c12" if hs.health_score >= 50 else "#e74c3c"

                c1, c2, c3 = st.columns(3)
                c1.metric("Health Score", f"{hs.health_score}/100")
                c2.metric("Status", hs.status)
                if model_ok:
                    try:
                        pred = api_client.predict_single(payload)
                        c3.metric("Breakdown Prob", f"{pred['probability']:.1%}")
                    except Exception:
                        pass

                col_g, col_r = st.columns(2)
                with col_g:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=hs.health_score,
                        title={"text": f"Health Score - {asset_tag}"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color, "thickness": 0.3},
                            "steps": [
                                {"range": [0, 50], "color": "#3a0a0a"},
                                {"range": [50, 75], "color": "#3a3a0a"},
                                {"range": [75, 100], "color": "#0a3a0a"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "value": 75},
                        },
                    ))
                    fig.update_layout(height=300, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                with col_r:
                    comp = hs.component_scores
                    fig2 = go.Figure(go.Scatterpolar(
                        r=list(comp.values()), theta=list(comp.keys()),
                        fill="toself", line_color=color, fillcolor=f"rgba(46,204,113,0.2)",
                    ))
                    fig2.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 100])),
                        title="Component Health Radar", template="plotly_dark", height=350,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Recommendations")
                for rec in hs.recommendations:
                    st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        uploaded = st.file_uploader("Upload fleet dataset",
                                    type=["csv", "xlsx", "json", "parquet"], key="hs_fleet")
        if uploaded:
            file_bytes = uploaded.read()
            try:
                from backend.file_parser import parse_upload
                from backend.health_score import compute_fleet_health
                df_fleet = parse_upload(uploaded.name, file_bytes)
                fleet_df = compute_fleet_health(df_fleet)

                c1, c2, c3 = st.columns(3)
                c1.metric("Assets", fleet_df["asset_tag"].nunique() if "asset_tag" in fleet_df.columns else len(fleet_df))
                c2.metric("Avg Health", f"{fleet_df['health_score'].mean():.1f}/100")
                c3.metric("Critical Assets", int((fleet_df["health_score"] < 50).sum()))

                st.dataframe(
                    fleet_df.style.background_gradient(subset=["health_score"], cmap="RdYlGn"),
                    use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    import plotly.express as px
                    if "asset_tag" in fleet_df.columns:
                        fig = px.bar(fleet_df.groupby("asset_tag")["health_score"].mean().reset_index(),
                                     x="asset_tag", y="health_score", color="health_score",
                                     color_continuous_scale="RdYlGn", range_color=[0, 100],
                                     title="Fleet Health Scores", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if "machine_type" in fleet_df.columns and "asset_tag" in fleet_df.columns:
                        st.plotly_chart(charts.fleet_health_treemap(fleet_df),
                                        use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TIME-TO-FAILURE
# =============================================================================
elif page == "⏱️  Time-to-Failure":
    st.title("⏱️ Time-to-Failure Prediction")
    st.markdown("Estimates days until breakdown per asset using degradation trend analysis.")

    uploaded = st.file_uploader("Upload historical dataset",
                                type=["csv", "xlsx", "json", "parquet"], key="ttf_file")
    if uploaded:
        file_bytes = uploaded.read()
        if st.button("Estimate TTF", type="primary"):
            with st.spinner("Analysing degradation trends..."):
                try:
                    from backend.file_parser import parse_upload
                    from backend.ttf_predictor import fleet_ttf
                    df_ttf = parse_upload(uploaded.name, file_bytes)
                    results = fleet_ttf(df_ttf)
                    df_res = pd.DataFrame(results)

                    critical = df_res[df_res["urgency"] == "Immediate"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Assets Analysed", len(df_res))
                    c2.metric("Immediate Action", len(critical))
                    c3.metric("Avg Days to Failure", f"{df_res['estimated_days'].mean():.0f}")
                    c4.metric("Min Days to Failure", f"{df_res['estimated_days'].min():.0f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(charts.ttf_gantt(results), use_container_width=True)
                    with col2:
                        import plotly.express as px
                        fig = px.scatter(df_res, x="degradation_rate", y="estimated_days",
                                         color="urgency", text="asset_tag",
                                         color_discrete_map={
                                             "Immediate": "#e74c3c", "This Week": "#e67e22",
                                             "This Month": "#f1c40f", "Routine": "#2ecc71"},
                                         title="Degradation Rate vs TTF", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                    def color_urgency(val):
                        m = {"Immediate": "background-color:#5a0a0a",
                             "This Week": "background-color:#5a3a0a",
                             "This Month": "background-color:#3a3a0a",
                             "Routine": "background-color:#0a3a0a"}
                        return m.get(val, "")

                    st.dataframe(
                        df_res.style.applymap(color_urgency, subset=["urgency"]),
                        use_container_width=True)

                    st.download_button("Download TTF Report",
                                       df_res.to_csv(index=False), "ttf_report.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")


# =============================================================================
# MODEL REGISTRY
# =============================================================================
elif page == "📈  Model Registry":
    st.title("📈 Model Registry & Comparison")

    if st.button("Refresh", type="primary"):
        try:
            from backend.model_registry import get_registry, compare_models, get_active_version
            registry = get_registry()
            comparison = compare_models()
            active = get_active_version()

            st.success(f"Active model: **{active}**")

            if not comparison.empty:
                c1, c2, c3 = st.columns(3)
                best = comparison.loc[comparison["auc"].idxmax()]
                c1.metric("Best AUC", f"{best['auc']:.4f}", f"v{best['version']}")
                c2.metric("Best Accuracy", f"{comparison['accuracy'].max():.4f}")
                c3.metric("Total Versions", len(comparison))

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts.model_comparison_chart(comparison),
                                    use_container_width=True)
                with col2:
                    import plotly.express as px
                    fig = px.bar(comparison, x="version", y="n_train",
                                 title="Training Dataset Size per Version",
                                 template="plotly_dark", color="auc",
                                 color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    comparison.style.highlight_max(subset=["accuracy", "auc", "f1"],
                                                   color="#1a3a1a")
                               .highlight_min(subset=["accuracy", "auc", "f1"],
                                              color="#3a0a0a"),
                    use_container_width=True)
            else:
                st.info("No models trained yet.")

            if registry:
                with st.expander("Full Registry JSON"):
                    st.json(registry)
        except Exception as e:
            st.error(f"Error: {e}")


# =============================================================================
# DOWNTIME CALCULATOR
# =============================================================================
elif page == "💰  Downtime Calculator":
    st.title("💰 Downtime Cost Calculator")
    st.markdown("Calculate financial impact of breakdowns and ROI of predictive maintenance.")

    tab1, tab2 = st.tabs(["Single Machine", "All Machines Comparison"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Machine Type",
                ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
            mttr = st.number_input("Mean Time to Repair (hours)", 1.0, 72.0, 8.0, 0.5)
            annual_bd = st.number_input("Annual Breakdowns", 1, 100, 12)
        with col2:
            hourly_prod = st.number_input("Hourly Production Value (Rs.)", 1000, 500000, 25000, 1000)
            repair_cost = st.number_input("Avg Repair Cost per Event (Rs.)", 1000, 1000000, 80000, 1000)
            pdm_cost = st.number_input("PdM System Annual Cost (Rs.)", 10000, 5000000, 500000, 10000)

        currency = st.radio("Currency", ["INR (Rs.)", "USD ($)"], horizontal=True)

        if st.button("Calculate", type="primary"):
            payload = {
                "machine_type": machine_type, "mttr_hours": mttr,
                "hourly_production_inr": hourly_prod, "repair_cost_inr": repair_cost,
                "annual_breakdowns": annual_bd, "pdm_system_cost_inr": pdm_cost,
            }
            try:
                result = api_client.calc_downtime(payload)
                rate = 83.5 if "USD" in currency else 1
                sym = "$" if "USD" in currency else "Rs."

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Loss / Event", f"{sym}{result['production_loss_inr']/rate:,.0f}")
                c2.metric("Total / Event", f"{sym}{result['total_cost_inr']/rate:,.0f}")
                c3.metric("Annual BD Cost", f"{sym}{result['annual_bd_cost_inr']/rate:,.0f}")
                c4.metric("Savings with PdM", f"{sym}{result['savings_with_pdm_inr']/rate:,.0f}",
                          delta=f"ROI: {result['roi_percent']}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts.downtime_cost_chart(result), use_container_width=True)
                with col2:
                    st.plotly_chart(charts.roi_waterfall(result), use_container_width=True)

                css = "alert-ok" if result["roi_percent"] > 0 else "alert-warning"
                st.markdown(
                    f'<div class="{css}">ROI: {result["roi_percent"]}% – '
                    f'{"PdM investment justified" if result["roi_percent"] > 0 else "Reduce PdM cost"}</div>',
                    unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.plotly_chart(charts.multi_machine_cost_comparison(), use_container_width=True)


# =============================================================================
# AI ADVISOR
# =============================================================================
elif page == "🧠  AI Advisor":
    st.title("🧠 AI Prescriptive Maintenance Advisor")
    st.markdown("LLM-powered root cause analysis and maintenance recommendations.")

    if not model_ok:
        st.warning("Train a model first.")
        st.stop()

    st.info("Configure `OPENAI_API_KEY` or `GROQ_API_KEY` in `.env` for AI recommendations.")

    col1, col2 = st.columns(2)
    with col1:
        asset_tag = st.text_input("Asset Tag", "CNC-001")
        machine_type = st.selectbox("Machine Type",
            ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
        temp_bearing = st.number_input("Bearing Temp (C)", 30.0, 120.0, 78.0)
        temp_motor = st.number_input("Motor Temp (C)", 30.0, 130.0, 88.0)
        vibration_h = st.number_input("H-Vibration (mm/s)", 0.0, 15.0, 5.2)
    with col2:
        vibration_v = st.number_input("V-Vibration (mm/s)", 0.0, 12.0, 4.1)
        oil_pressure = st.number_input("Oil Pressure (bar)", 0.0, 150.0, 4.2)
        load_pct = st.number_input("Load (%)", 0.0, 100.0, 82.0)
        shaft_rpm = st.number_input("Shaft RPM", 0.0, 4000.0, 1450.0)
        power_kw = st.number_input("Power (kW)", 0.0, 100.0, 38.0)

    if st.button("Get AI Advice", type="primary"):
        payload = {
            "asset_tag": asset_tag, "machine_type": machine_type,
            "temp_bearing_degC": temp_bearing, "temp_motor_degC": temp_motor,
            "vibration_h_mms": vibration_h, "vibration_v_mms": vibration_v,
            "oil_pressure_bar": oil_pressure, "load_pct": load_pct,
            "shaft_rpm": shaft_rpm, "power_consumption_kw": power_kw,
        }
        with st.spinner("Consulting AI advisor..."):
            try:
                result = api_client.get_advice(payload)
                pred = result["prediction"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", "BREAKDOWN" if pred["prediction"] else "NORMAL")
                c2.metric("Probability", f"{pred['probability']:.1%}")
                c3.metric("Risk Level", pred["risk_level"])

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(charts.risk_gauge(pred["probability"]),
                                    use_container_width=True)
                with col2:
                    st.subheader("AI Recommendation")
                    st.markdown(result["advice"])
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# AUDIT LOGS
# =============================================================================
elif page == "📋  Audit Logs":
    st.title("📋 Legal Audit Logs")
    st.markdown("All system actions logged to Supabase for compliance and traceability.")

    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.slider("Rows to fetch", 10, 500, 100)
    with col2:
        action_filter = st.multiselect("Filter by action",
            ["model_trained", "batch_predict", "predict", "llm_advise", "ttf_analysis"])

    if st.button("Refresh Logs", type="primary"):
        try:
            result = api_client.get_audit_logs(limit)
            logs = result.get("logs", [])
            if logs:
                df_logs = pd.DataFrame(logs)
                if action_filter:
                    df_logs = df_logs[df_logs["action"].isin(action_filter)]

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Logs", len(df_logs))
                if "action" in df_logs.columns:
                    c2.metric("Unique Actions", df_logs["action"].nunique())
                if "created_at" in df_logs.columns:
                    c3.metric("Latest", str(df_logs["created_at"].max())[:10])

                if "action" in df_logs.columns:
                    import plotly.express as px
                    fig = px.histogram(df_logs, x="action", title="Actions Distribution",
                                       template="plotly_dark", color="action")
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df_logs, use_container_width=True)
                st.download_button("Export CSV", df_logs.to_csv(index=False),
                                   "audit_log.csv", "text/csv")
            else:
                st.info(result.get("message", "No logs found. Configure Supabase in .env"))
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.markdown("""
| Action | Trigger |
|--------|---------|
| `model_trained` | Dataset uploaded + model trained |
| `batch_predict` | Batch prediction file uploaded |
| `predict` | Single sensor reading predicted |
| `llm_advise` | AI advisor consulted |
| `ttf_analysis` | Time-to-failure computed |
    """)
