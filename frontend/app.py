"""
🔧 Industrial Machine Predictive Maintenance System
Streamlit frontend – fully interactive, no static plots.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
import threading
import time
from collections import defaultdict, deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend import api_client, charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredMaint – Industrial AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
.metric-card {
    background: #161b22; border-radius: 10px;
    padding: 16px; border: 1px solid #30363d;
}
.risk-critical { color: #e74c3c; font-weight: bold; font-size: 1.2em; }
.risk-high     { color: #e67e22; font-weight: bold; }
.risk-medium   { color: #f1c40f; font-weight: bold; }
.risk-low      { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/maintenance.png", width=64)
    st.title("PredMaint AI")
    st.caption("Industrial Predictive Maintenance")
    st.divider()

    health = api_client.health_check()
    api_ok = health.get("status") == "ok"
    model_ok = health.get("model_ready", False)

    st.markdown(f"**API:** {'🟢 Online' if api_ok else '🔴 Offline'}")
    st.markdown(f"**Model:** {'✅ Ready' if model_ok else '⚠️ Not trained'}")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📁 Upload & Train",
        "📊 EDA Explorer",
        "🤖 Predict Breakdown",
        "🔴 Live IoT Monitor",
        "🔍 SHAP Explainability",
        "❤️ Machine Health Score",
        "⏱️ Time-to-Failure",
        "📈 Model Registry",
        "💰 Downtime Calculator",
        "🧠 AI Maintenance Advisor",
        "📋 Audit Logs",
    ])

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = None
if "live_history" not in st.session_state:
    st.session_state.live_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=60)))
if "live_running" not in st.session_state:
    st.session_state.live_running = False


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🔧 Industrial Machine Predictive Maintenance")
    st.markdown("**AI-powered breakdown prediction, anomaly detection & prescriptive maintenance**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", "219,000", "3+ years")
    c2.metric("Machine Types", "5", "10 assets")
    c3.metric("Breakdown Rate", "~9.9%", "Class imbalance")
    c4.metric("Target Reduction", "40-60%", "Unplanned downtime")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 System Capabilities")
        st.markdown("""
- 📁 **Universal file upload** – CSV, Excel, JSON, Parquet, TSV
- 🤖 **ML Models** – Random Forest + Isolation Forest
- 🔍 **SHAP Explainability** – Feature importance & beeswarm
- 🔴 **Live IoT Simulation** – Real-time WebSocket sensor stream
- 🧠 **LLM Advisor** – GPT-4o / Llama3 prescriptive recommendations
- 💰 **Downtime Calculator** – INR/USD cost & ROI analysis
- 🚨 **Alerts** – Slack + Email on breakdown prediction
- 📋 **Audit Logs** – Legal-grade Supabase logging
- ❤️ **Health Score** – Per-machine 0-100 health index
- ⏱️ **Time-to-Failure** – Days until predicted breakdown
- 📈 **Model Registry** – Version, compare & track all training runs
        """)
    with col2:
        st.subheader("🏭 Supported Machines")
        machines = {
            "CNC Lathe": "🔩", "Hydraulic Press": "⚙️",
            "Belt Conveyor": "📦", "Screw Compressor": "💨", "EOT Crane": "🏗️"
        }
        for name, icon in machines.items():
            st.markdown(f"{icon} **{name}**")

    st.divider()
    st.subheader("🚀 Quick Start")
    st.info("1. Go to **Upload & Train** → upload your CSV → train the model\n"
            "2. Use **EDA Explorer** for interactive visualizations\n"
            "3. Use **Predict Breakdown** for single or batch predictions\n"
            "4. Monitor **Live IoT** for real-time sensor simulation")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Upload & Train":
    st.title("📁 Upload Dataset & Train Model")
    st.markdown("Supports **CSV, Excel (.xlsx/.xls), JSON, Parquet, TSV, Feather**")

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "feather"],
        help="Must contain sensor columns + breakdown_flag for training"
    )

    if uploaded:
        file_bytes = uploaded.read()
        st.success(f"✅ File received: `{uploaded.name}` ({len(file_bytes)/1024:.1f} KB)")

        col1, col2 = st.columns(2)
        with col1:
            action = st.radio("Action", ["Train Model", "Batch Predict"])
        with col2:
            st.info("**Train Model** – requires `breakdown_flag` column\n\n"
                    "**Batch Predict** – requires sensor columns only")

        if st.button("🚀 Execute", type="primary"):
            with st.spinner("Processing..."):
                try:
                    if action == "Train Model":
                        result = api_client.upload_and_train(file_bytes, uploaded.name)
                        st.session_state.train_metrics = result.get("metrics", {})
                        st.success("✅ Model trained successfully!")

                        m = result["metrics"]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
                        c2.metric("AUC-ROC", f"{m.get('auc', 0):.3f}")
                        c3.metric("Precision (BD)", f"{m.get('precision_breakdown', 0):.3f}")
                        c4.metric("Recall (BD)", f"{m.get('recall_breakdown', 0):.3f}")

                        st.json(m)

                    else:  # Batch Predict
                        result = api_client.upload_predict(file_bytes, uploaded.name)
                        st.success(f"✅ Predictions complete! {result['total']} rows processed.")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Rows", result["total"])
                        c2.metric("Breakdowns Predicted", result["breakdowns_predicted"])
                        c3.metric("Critical Risk", result["critical"])

                        df_res = pd.DataFrame(result["results"])
                        st.dataframe(df_res, use_container_width=True)
                        st.plotly_chart(charts.prediction_results_chart(result["results"]),
                                        use_container_width=True)

                        csv = df_res.to_csv(index=False)
                        st.download_button("⬇️ Download Results CSV", csv,
                                           "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Also load for EDA
        try:
            import io
            from backend.file_parser import parse_upload
            df = parse_upload(uploaded.name, file_bytes)
            st.session_state.df = df
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            c3.metric("Missing Values", int(df.isnull().sum().sum()))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.title("📊 Interactive EDA Explorer")

    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Upload a dataset first (Upload & Train page).")
        st.stop()

    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    if "breakdown_flag" in df.columns and "is_breakdown" not in df.columns:
        df["is_breakdown"] = df["breakdown_flag"].map({0: "Normal", 1: "Breakdown"})
    if "month" not in df.columns and "transaction_date" in df.columns:
        df["month"] = df["transaction_date"].dt.month

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Distributions", "📉 Trends", "🥧 Categorical",
        "📦 Boxplots", "🔥 Correlation", "🌐 3D Scatter", "📅 Monthly"
    ])

    with tab1:
        st.plotly_chart(charts.histogram_grid(df), use_container_width=True)
        st.caption("Frequency distributions with KDE overlay for key sensors.")

    with tab2:
        if "transaction_date" in df.columns:
            st.plotly_chart(charts.lineplot_trends(df), use_container_width=True)
        else:
            st.info("No `transaction_date` column found.")

    with tab3:
        st.plotly_chart(charts.pie_charts(df), use_container_width=True)
        if "machine_type" in df.columns and "is_breakdown" in df.columns:
            st.plotly_chart(charts.breakdown_by_machine(df), use_container_width=True)

    with tab4:
        if "is_breakdown" in df.columns:
            st.plotly_chart(charts.boxplot_sensors(df), use_container_width=True)
        else:
            st.info("Need `breakdown_flag` column for boxplots.")

    with tab5:
        st.plotly_chart(charts.correlation_heatmap(df), use_container_width=True)

    with tab6:
        if all(c in df.columns for c in ["temp_bearing_degC", "vibration_h_mms",
                                          "power_consumption_kw", "is_breakdown"]):
            st.plotly_chart(charts.scatter_3d(df), use_container_width=True)
        else:
            st.info("Need sensor columns + breakdown_flag for 3D scatter.")

    with tab7:
        if "breakdown_flag" in df.columns:
            st.plotly_chart(charts.monthly_breakdown_rate(df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Breakdown":
    st.title("🤖 Predict Machine Breakdown")

    if not model_ok:
        st.warning("⚠️ Train a model first.")
        st.stop()

    st.subheader("Enter Sensor Readings")
    col1, col2 = st.columns(2)

    with col1:
        asset_tag = st.text_input("Asset Tag", "CNC-001")
        machine_type = st.selectbox("Machine Type",
            ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
        temp_bearing = st.slider("Bearing Temp (°C)", 30.0, 120.0, 65.0, 0.5)
        temp_motor = st.slider("Motor Temp (°C)", 30.0, 130.0, 75.0, 0.5)
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

    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Running prediction..."):
            try:
                result = api_client.predict_single(payload)
                prob = result["probability"]
                risk = result["risk_level"]

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", "⚠️ BREAKDOWN" if result["prediction"] else "✅ NORMAL")
                c2.metric("Probability", f"{prob:.1%}")
                c3.metric("Risk Level", risk)

                st.plotly_chart(charts.risk_gauge(prob), use_container_width=True)

                if risk in ("HIGH", "CRITICAL"):
                    st.error(f"🚨 **{risk} RISK** – Alerts sent to Slack/Email")
                elif risk == "MEDIUM":
                    st.warning("⚠️ Medium risk – Schedule inspection soon")
                else:
                    st.success("✅ Low risk – Normal operation")

                with st.expander("Full prediction details"):
                    st.json(result)

            except Exception as e:
                st.error(f"❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE IoT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live IoT Monitor":
    st.title("🔴 Live IoT Sensor Monitor")
    st.markdown("Real-time sensor simulation via WebSocket. Charts auto-refresh every second.")

    selected_asset = st.selectbox("Select Asset", [
        "CNC-001", "HYD-001", "BLT-001", "CMP-001", "EOT-001"
    ])

    col1, col2 = st.columns([1, 3])
    with col1:
        run = st.toggle("▶️ Start Live Feed", value=False)

    placeholder_gauges = st.empty()
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

    if run:
        import websocket as ws_lib
        import threading

        history = defaultdict(lambda: deque(maxlen=60))
        latest = {}
        lock = threading.Lock()

        def on_message(ws, message):
            readings = json.loads(message)
            for r in readings:
                if r["asset_tag"] == selected_asset:
                    with lock:
                        for k, v in r.items():
                            if isinstance(v, (int, float)):
                                history[k].append(v)
                        latest.update(r)

        def on_error(ws, error):
            pass

        def run_ws():
            wsa = ws_lib.WebSocketApp(
                "ws://localhost:8000/ws/live-sensors",
                on_message=on_message, on_error=on_error
            )
            wsa.run_forever()

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

        for _ in range(300):  # 5 minutes max
            if not run:
                break
            time.sleep(1)

            with lock:
                snap = dict(latest)
                hist_snap = {k: list(v) for k, v in history.items()}

            if snap:
                with placeholder_gauges.container():
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Bearing Temp", f"{snap.get('temp_bearing_degC', 0):.1f} °C")
                    g2.metric("H-Vibration", f"{snap.get('vibration_h_mms', 0):.2f} mm/s")
                    g3.metric("Oil Pressure", f"{snap.get('oil_pressure_bar', 0):.1f} bar")
                    g4.metric("Power", f"{snap.get('power_consumption_kw', 0):.1f} kW")

                    deg = snap.get("degradation_index", 0)
                    st.plotly_chart(charts.degradation_gauge(deg, selected_asset),
                                    use_container_width=True)

                with placeholder_chart.container():
                    st.plotly_chart(
                        charts.live_sensor_chart(hist_snap, selected_asset),
                        use_container_width=True
                    )

                with placeholder_table.container():
                    st.dataframe(pd.DataFrame([snap]), use_container_width=True)
    else:
        st.info("Toggle **Start Live Feed** to begin real-time monitoring.\n\n"
                "Make sure the FastAPI backend is running: `uvicorn backend.main:app --reload`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainability":
    st.title("🔍 SHAP Model Explainability")
    st.markdown("Understand **why** the model predicts a breakdown.")

    if not model_ok:
        st.warning("⚠️ Train a model first.")
        st.stop()

    uploaded = st.file_uploader("Upload dataset for SHAP analysis",
                                type=["csv", "xlsx", "json", "parquet"])
    if uploaded:
        file_bytes = uploaded.read()
        if st.button("🔍 Compute SHAP", type="primary"):
            with st.spinner("Computing SHAP values (may take 30–60s)..."):
                try:
                    result = api_client.get_shap(file_bytes, uploaded.name)
                    st.success("✅ SHAP analysis complete!")

                    st.subheader("Feature Importance (Mean |SHAP|)")
                    st.plotly_chart(
                        charts.feature_importance_chart(result["feature_importance"]),
                        use_container_width=True
                    )

                    st.subheader("Raw SHAP Values (first 10 rows)")
                    shap_df = pd.DataFrame(
                        result["shap_values"][:10],
                        columns=result["feature_names"]
                    )
                    st.dataframe(shap_df.style.background_gradient(cmap="RdYlGn", axis=None),
                                 use_container_width=True)

                    with st.expander("Full feature importance dict"):
                        st.json(result["feature_importance"])

                except Exception as e:
                    st.error(f"❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNTIME CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Downtime Calculator":
    st.title("💰 Downtime Cost Calculator")
    st.markdown("Calculate the financial impact of machine breakdowns and ROI of PdM.")

    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox("Machine Type",
            ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
        mttr = st.number_input("Mean Time to Repair (hours)", 1.0, 72.0, 8.0, 0.5)
        annual_bd = st.number_input("Annual Breakdowns (estimated)", 1, 100, 12)

    with col2:
        hourly_prod = st.number_input("Hourly Production Value (₹)", 1000, 500000, 25000, 1000)
        repair_cost = st.number_input("Avg Repair Cost per Event (₹)", 1000, 1000000, 80000, 1000)
        pdm_cost = st.number_input("PdM System Annual Cost (₹)", 10000, 5000000, 500000, 10000)

    currency = st.radio("Display Currency", ["INR (₹)", "USD ($)"], horizontal=True)

    if st.button("💰 Calculate", type="primary"):
        payload = {
            "machine_type": machine_type,
            "mttr_hours": mttr,
            "hourly_production_inr": hourly_prod,
            "repair_cost_inr": repair_cost,
            "annual_breakdowns": annual_bd,
            "pdm_system_cost_inr": pdm_cost,
        }
        try:
            result = api_client.calc_downtime(payload)
            rate = 83.5 if "USD" in currency else 1
            sym = "$" if "USD" in currency else "₹"

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Production Loss / Event", f"{sym}{result['production_loss_inr']/rate:,.0f}")
            c2.metric("Total Cost / Event", f"{sym}{result['total_cost_inr']/rate:,.0f}")
            c3.metric("Annual BD Cost", f"{sym}{result['annual_bd_cost_inr']/rate:,.0f}")
            c4.metric("Savings with PdM", f"{sym}{result['savings_with_pdm_inr']/rate:,.0f}",
                      delta=f"ROI: {result['roi_percent']}%")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts.downtime_cost_chart(result), use_container_width=True)
            with col2:
                st.plotly_chart(charts.roi_waterfall(result), use_container_width=True)

            if result["roi_percent"] > 0:
                st.success(f"✅ Positive ROI of **{result['roi_percent']}%** – PdM investment is justified!")
            else:
                st.warning(f"⚠️ ROI is {result['roi_percent']}% – consider reducing PdM system cost.")

        except Exception as e:
            st.error(f"❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI MAINTENANCE ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Maintenance Advisor":
    st.title("🧠 AI Prescriptive Maintenance Advisor")
    st.markdown("Get **LLM-powered** root cause analysis and maintenance recommendations.")

    if not model_ok:
        st.warning("⚠️ Train a model first.")
        st.stop()

    st.info("💡 Configure `OPENAI_API_KEY` or `GROQ_API_KEY` in `.env` for AI recommendations. "
            "Falls back to rule-based advice if not configured.")

    col1, col2 = st.columns(2)
    with col1:
        asset_tag = st.text_input("Asset Tag", "CNC-001")
        machine_type = st.selectbox("Machine Type",
            ["CNC Lathe", "Hydraulic Press", "Belt Conveyor", "Screw Compressor", "EOT Crane"])
        temp_bearing = st.number_input("Bearing Temp (°C)", 30.0, 120.0, 78.0)
        temp_motor = st.number_input("Motor Temp (°C)", 30.0, 130.0, 88.0)
        vibration_h = st.number_input("H-Vibration (mm/s)", 0.0, 15.0, 5.2)

    with col2:
        vibration_v = st.number_input("V-Vibration (mm/s)", 0.0, 12.0, 4.1)
        oil_pressure = st.number_input("Oil Pressure (bar)", 0.0, 150.0, 4.2)
        load_pct = st.number_input("Load (%)", 0.0, 100.0, 82.0)
        shaft_rpm = st.number_input("Shaft RPM", 0.0, 4000.0, 1450.0)
        power_kw = st.number_input("Power (kW)", 0.0, 100.0, 38.0)

    if st.button("🧠 Get AI Advice", type="primary"):
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

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", "⚠️ BREAKDOWN" if pred["prediction"] else "✅ NORMAL")
                c2.metric("Probability", f"{pred['probability']:.1%}")
                c3.metric("Risk Level", pred["risk_level"])

                st.plotly_chart(charts.risk_gauge(pred["probability"]), use_container_width=True)

                st.subheader("🤖 AI Maintenance Recommendation")
                st.markdown(result["advice"])

            except Exception as e:
                st.error(f"❌ {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT LOGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Audit Logs":
    st.title("📋 Legal Audit Logs")
    st.markdown("All system actions are logged to Supabase for compliance and traceability.")

    limit = st.slider("Number of logs to fetch", 10, 500, 100)

    if st.button("🔄 Refresh Logs"):
        try:
            result = api_client.get_audit_logs(limit)
            logs = result.get("logs", [])
            if logs:
                df_logs = pd.DataFrame(logs)
                st.dataframe(df_logs, use_container_width=True)
                csv = df_logs.to_csv(index=False)
                st.download_button("⬇️ Export Audit Log CSV", csv,
                                   "audit_log.csv", "text/csv")
            else:
                st.info(result.get("message", "No logs found."))
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()
    st.subheader("📌 What gets logged?")
    st.markdown("""
| Action | Trigger |
|--------|---------|
| `model_trained` | Dataset uploaded + model trained |
| `batch_predict` | Batch prediction file uploaded |
| `predict` | Single sensor reading predicted |
| `llm_advise` | AI advisor consulted |
| `alert_sent` | Slack/Email alert fired |
    """)


# =============================================================================
# PAGE: MACHINE HEALTH SCORE
# =============================================================================
if page == "❤️ Machine Health Score":
    st.title("❤️ Machine Health Score")
    st.markdown("Real-time 0-100 health index per asset based on weighted sensor deviation from normal thresholds.")

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
                result = api_client.predict_single(payload)
                # Compute health locally using backend module
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from backend.health_score import compute_health_score
                hs = compute_health_score(payload)

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Health Score", f"{hs.health_score}/100")
                c2.metric("Status", hs.status)
                c3.metric("Breakdown Prob", f"{result['probability']:.1%}")

                # Health gauge
                import plotly.graph_objects as go
                color = "#2ecc71" if hs.health_score >= 75 else "#f39c12" if hs.health_score >= 50 else "#e74c3c"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hs.health_score,
                    title={"text": f"Health Score - {asset_tag}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
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

                # Component scores radar
                import plotly.express as px
                comp = hs.component_scores
                fig2 = go.Figure(go.Scatterpolar(
                    r=list(comp.values()),
                    theta=list(comp.keys()),
                    fill="toself",
                    line_color=color,
                ))
                fig2.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    title="Component Health Radar",
                    template="plotly_dark", height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Recommendations")
                for rec in hs.recommendations:
                    st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        uploaded = st.file_uploader("Upload fleet dataset", type=["csv", "xlsx", "json", "parquet"], key="hs_fleet")
        if uploaded:
            file_bytes = uploaded.read()
            try:
                from backend.file_parser import parse_upload
                from backend.health_score import compute_fleet_health
                df_fleet = parse_upload(uploaded.name, file_bytes)
                fleet_df = compute_fleet_health(df_fleet)

                st.dataframe(fleet_df.style.background_gradient(
                    subset=["health_score"], cmap="RdYlGn"), use_container_width=True)

                import plotly.express as px
                fig = px.bar(fleet_df.groupby("asset_tag")["health_score"].mean().reset_index(),
                             x="asset_tag", y="health_score", color="health_score",
                             color_continuous_scale="RdYlGn", range_color=[0, 100],
                             title="Fleet Health Scores", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# PAGE: TIME-TO-FAILURE
# =============================================================================
elif page == "⏱️ Time-to-Failure":
    st.title("⏱️ Time-to-Failure Prediction")
    st.markdown("Estimates days until breakdown per asset using degradation trend analysis.")

    uploaded = st.file_uploader("Upload historical dataset (needs asset_tag + sensor cols)",
                                type=["csv", "xlsx", "json", "parquet"], key="ttf_file")
    if uploaded:
        file_bytes = uploaded.read()
        if st.button("Estimate TTF for All Assets", type="primary"):
            with st.spinner("Analysing degradation trends..."):
                try:
                    from backend.file_parser import parse_upload
                    from backend.ttf_predictor import fleet_ttf
                    df_ttf = parse_upload(uploaded.name, file_bytes)
                    results = fleet_ttf(df_ttf)
                    df_res = pd.DataFrame(results)

                    st.divider()
                    # Summary metrics
                    critical = df_res[df_res["urgency"] == "Immediate"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Assets Analysed", len(df_res))
                    c2.metric("Immediate Action", len(critical))
                    c3.metric("Avg Days to Failure", f"{df_res['estimated_days'].mean():.0f}")

                    # Color-coded table
                    def color_urgency(val):
                        colors = {"Immediate": "background-color:#5a0a0a",
                                  "This Week": "background-color:#5a3a0a",
                                  "This Month": "background-color:#3a3a0a",
                                  "Routine": "background-color:#0a3a0a"}
                        return colors.get(val, "")

                    st.dataframe(
                        df_res.style.applymap(color_urgency, subset=["urgency"]),
                        use_container_width=True
                    )

                    # TTF bar chart
                    import plotly.express as px
                    fig = px.bar(df_res, x="asset_tag", y="estimated_days",
                                 color="urgency",
                                 color_discrete_map={
                                     "Immediate": "#e74c3c", "This Week": "#e67e22",
                                     "This Month": "#f1c40f", "Routine": "#2ecc71"
                                 },
                                 title="Estimated Days to Failure by Asset",
                                 template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                    # Degradation rate chart
                    fig2 = px.scatter(df_res, x="degradation_rate", y="estimated_days",
                                      color="urgency", text="asset_tag", size_max=15,
                                      title="Degradation Rate vs Time-to-Failure",
                                      template="plotly_dark")
                    st.plotly_chart(fig2, use_container_width=True)

                    csv = df_res.to_csv(index=False)
                    st.download_button("Download TTF Report", csv, "ttf_report.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error: {e}")


# =============================================================================
# PAGE: MODEL REGISTRY
# =============================================================================
elif page == "📈 Model Registry":
    st.title("📈 Model Registry & Comparison")
    st.markdown("Track every training run, compare metrics, and see which model is active.")

    if st.button("Refresh Registry", type="primary"):
        try:
            result = api_client._client().get("/model-registry").json() if hasattr(api_client, "_client") else None
            # Fallback: load locally
            from backend.model_registry import get_registry, compare_models, get_active_version
            registry = get_registry()
            comparison = compare_models()
            active = get_active_version()

            st.success(f"Active model: **{active}**")

            if not comparison.empty:
                st.subheader("Model Comparison Table")
                st.dataframe(
                    comparison.style.highlight_max(subset=["accuracy", "auc", "f1"], color="#1a3a1a")
                               .highlight_min(subset=["accuracy", "auc", "f1"], color="#3a0a0a"),
                    use_container_width=True
                )

                import plotly.express as px
                fig = px.line(comparison, x="version", y=["accuracy", "auc", "f1"],
                              title="Model Performance Across Versions",
                              markers=True, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(comparison, x="version", y="n_train",
                              title="Training Dataset Size per Version",
                              template="plotly_dark", color="auc",
                              color_continuous_scale="Viridis")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No models trained yet. Go to Upload & Train to train your first model.")

            if registry:
                with st.expander("Full Registry JSON"):
                    st.json(registry)

        except Exception as e:
            st.error(f"Error: {e}")
