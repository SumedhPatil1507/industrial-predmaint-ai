"""
Industrial Machine Predictive Maintenance System
Self-contained Streamlit app – works on Streamlit Cloud with NO backend needed.
All ML, live simulation, health scoring run directly in-process.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import random
import io
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
.block-container { padding-top: 1.5rem; }
.kpi-card {
    background: linear-gradient(135deg,#161b22,#1c2128);
    border:1px solid #30363d; border-radius:12px;
    padding:18px 22px; text-align:center;
    transition:transform .2s,border-color .2s;
}
.kpi-card:hover{transform:translateY(-3px);border-color:#58a6ff;}
.kpi-value{font-size:1.9rem;font-weight:700;color:#58a6ff;}
.kpi-label{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;}
.kpi-delta{font-size:.82rem;color:#3fb950;margin-top:4px;}
.alert-critical{background:linear-gradient(135deg,#3d0000,#5a0a0a);
    border:1px solid #e74c3c;border-radius:8px;padding:12px 16px;
    color:#ff6b6b;font-weight:600;animation:pulse 2s infinite;}
.alert-warning{background:linear-gradient(135deg,#2d1a00,#3d2500);
    border:1px solid #e67e22;border-radius:8px;padding:12px 16px;color:#ffa94d;}
.alert-ok{background:linear-gradient(135deg,#001a0d,#002b15);
    border:1px solid #2ecc71;border-radius:8px;padding:12px 16px;color:#69db7c;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(231,76,60,.4);}
    50%{box-shadow:0 0 0 8px rgba(231,76,60,0);}}
.section-hdr{font-size:1.05rem;font-weight:600;color:#58a6ff;
    border-bottom:1px solid #21262d;padding-bottom:6px;margin-bottom:14px;}
div[data-testid="stMetricValue"]{font-size:1.5rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Inline imports (no backend server needed) ─────────────────────────────────
from frontend.charts import (
    histogram_grid, lineplot_trends, breakdown_by_machine, pie_charts,
    boxplot_sensors, violin_sensors, correlation_heatmap, scatter_3d,
    monthly_breakdown_rate, sensor_vs_breakdown_scatter, rolling_anomaly_chart,
    feature_importance_chart, prediction_results_chart, risk_gauge,
    live_sensor_chart, live_all_assets_chart, degradation_gauge,
    live_prediction_timeline, downtime_cost_chart, roi_waterfall,
    multi_machine_cost_comparison, model_comparison_chart,
    fleet_health_treemap, ttf_gantt,
)
from backend.health_score import compute_health_score, compute_fleet_health
from backend.ttf_predictor import fleet_ttf
from backend.downtime_calculator import calculate_downtime, MACHINE_DEFAULTS
from backend.iot_simulator import MACHINES, RANGES, _state, _generate_reading
from backend.ml_engine import (
    engineer_features, get_all_features, train_model,
    predict_single, predict_batch, compute_shap, model_exists,
    FEATURE_COLS, TARGET,
)
from backend.file_parser import parse_upload, validate_columns, infer_schema
from backend.model_registry import get_registry, compare_models, get_active_version

# ── Session state ─────────────────────────────────────────────────────────────
_ss_defaults = {
    "df": None,
    "model_trained": model_exists(),
    "live_ticks": 0,
    "live_history": {tag: defaultdict(lambda: deque(maxlen=80)) for tag in
                     [m["asset_tag"] for m in MACHINES]},
    "live_pred_history": defaultdict(lambda: deque(maxlen=80)),
    "sim_state": {m["asset_tag"]: {"degradation": 0.0, "tick": 0} for m in MACHINES},
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

model_ok = st.session_state.model_trained or model_exists()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 PredMaint AI")
    st.caption("Industrial Predictive Maintenance")
    st.divider()
    st.markdown(f"**Model:** {'✅ Ready' if model_ok else '⚠️ Not trained'}")
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
    st.caption("v3.0 · Streamlit Cloud · Self-contained")


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "🏠  Dashboard":
    st.markdown("## 🔧 Industrial Machine Predictive Maintenance")
    st.markdown("AI-powered breakdown prediction · anomaly detection · prescriptive maintenance")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl,dlt) in zip([c1,c2,c3,c4,c5],[
        ("219K","Dataset Records","3+ years"),
        ("5","Machine Types","10 assets"),
        ("9.9%","Breakdown Rate","Class imbalance"),
        ("40-60%","Downtime Reduction","With PdM"),
        ("12","Features","Sensor+engineered"),
    ]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                     f'<div class="kpi-label">{lbl}</div>'
                     f'<div class="kpi-delta">{dlt}</div></div>', unsafe_allow_html=True)

    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">System Capabilities</div>', unsafe_allow_html=True)
        for icon,name,desc in [
            ("📁","Universal file upload","CSV, Excel, JSON, Parquet, TSV"),
            ("🤖","ML Models","Random Forest + Isolation Forest"),
            ("🔍","SHAP Explainability","Feature importance + beeswarm"),
            ("🔴","Live IoT Simulation","In-browser real-time sensor stream"),
            ("🧠","AI Advisor","Rule-based prescriptive recommendations"),
            ("💰","Downtime Calculator","INR/USD cost + ROI waterfall"),
            ("❤️","Health Score","0-100 per asset with radar chart"),
            ("⏱️","Time-to-Failure","Days until predicted breakdown"),
            ("📈","Model Registry","Version + compare training runs"),
        ]:
            st.markdown(f"**{icon} {name}** — {desc}")
    with col2:
        st.markdown('<div class="section-hdr">Supported Machines</div>', unsafe_allow_html=True)
        for icon,name,desc in [
            ("🔩","CNC Lathe","High-precision cutting"),
            ("⚙️","Hydraulic Press","High-force forming"),
            ("📦","Belt Conveyor","Material transport"),
            ("💨","Screw Compressor","Compressed air supply"),
            ("🏗️","EOT Crane","Heavy lifting"),
        ]:
            st.markdown(f"{icon} **{name}** — {desc}")
        st.divider()
        if model_ok:
            st.markdown('<div class="alert-ok">✅ Model ready. All features unlocked.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">⚠️ Upload a CSV and train the model to unlock predictions.</div>', unsafe_allow_html=True)


# =============================================================================
# UPLOAD & TRAIN
# =============================================================================
elif page == "📁  Upload & Train":
    st.title("📁 Upload Dataset & Train Model")
    st.markdown("Supports **CSV, Excel, JSON, Parquet, TSV**")

    uploaded = st.file_uploader("Drop your dataset here",
        type=["csv","tsv","xlsx","xls","json","parquet","feather"])

    if uploaded:
        file_bytes = uploaded.read()
        st.success(f"File received: `{uploaded.name}` ({len(file_bytes)/1024:.1f} KB)")
        action = st.radio("Action", ["Train Model","Batch Predict"], horizontal=True)

        if st.button("Execute", type="primary"):
            with st.spinner("Processing..."):
                try:
                    df = parse_upload(uploaded.name, file_bytes)
                    st.session_state.df = df

                    if action == "Train Model":
                        ok, missing = validate_columns(df, FEATURE_COLS + [TARGET])
                        if not ok:
                            st.error(f"Missing columns: {missing}")
                        else:
                            metrics = train_model(df)
                            st.session_state.model_trained = True
                            st.success("Model trained successfully!")
                            c1,c2,c3,c4 = st.columns(4)
                            c1.metric("Accuracy", f"{metrics.get('accuracy',0):.3f}")
                            c2.metric("AUC-ROC",  f"{metrics.get('auc',0):.3f}")
                            c3.metric("Precision", f"{metrics.get('precision_breakdown',0):.3f}")
                            c4.metric("Recall",    f"{metrics.get('recall_breakdown',0):.3f}")
                            with st.expander("Full metrics"):
                                st.json(metrics)
                    else:
                        result_df = predict_batch(df)
                        bd = int(result_df["prediction"].sum())
                        crit = int((result_df["risk_level"]=="CRITICAL").sum())
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Total Rows", len(result_df))
                        c2.metric("Breakdowns Predicted", bd)
                        c3.metric("Critical Risk", crit)
                        st.dataframe(result_df.head(500), use_container_width=True)
                        st.plotly_chart(prediction_results_chart(
                            result_df[["probability","risk_level"]].to_dict("records")),
                            use_container_width=True)
                        st.download_button("Download Results CSV",
                            result_df.to_csv(index=False), "predictions.csv","text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

        try:
            df = parse_upload(uploaded.name, file_bytes)
            st.session_state.df = df
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
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
        df["is_breakdown"] = df["breakdown_flag"].map({0:"Normal",1:"Breakdown"})
    if "month" not in df.columns and "transaction_date" in df.columns:
        df["month"] = df["transaction_date"].dt.month

    with st.expander("Filters", expanded=False):
        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            if "machine_type" in df.columns:
                sel = st.multiselect("Machine Type", df["machine_type"].unique(),
                                     default=list(df["machine_type"].unique()))
                df = df[df["machine_type"].isin(sel)]
        with fc2:
            if "transaction_date" in df.columns:
                dr = st.date_input("Date Range",
                    [df["transaction_date"].min().date(), df["transaction_date"].max().date()])
                if len(dr)==2:
                    df = df[(df["transaction_date"].dt.date>=dr[0]) &
                            (df["transaction_date"].dt.date<=dr[1])]
        with fc3:
            if "is_breakdown" in df.columns:
                bf = st.multiselect("Status",["Normal","Breakdown"],
                                    default=["Normal","Breakdown"])
                df = df[df["is_breakdown"].isin(bf)]
    st.caption(f"Showing {len(df):,} rows")

    tabs = st.tabs(["Distributions","Trends","Categorical","Boxplots",
                    "Violins","Correlation","3D Scatter","Scatter Matrix",
                    "Rolling Anomaly","Monthly"])
    with tabs[0]: st.plotly_chart(histogram_grid(df), use_container_width=True)
    with tabs[1]:
        if "transaction_date" in df.columns:
            st.plotly_chart(lineplot_trends(df), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(pie_charts(df), use_container_width=True)
        if "machine_type" in df.columns and "is_breakdown" in df.columns:
            st.plotly_chart(breakdown_by_machine(df), use_container_width=True)
    with tabs[3]:
        if "is_breakdown" in df.columns:
            st.plotly_chart(boxplot_sensors(df), use_container_width=True)
    with tabs[4]:
        if "is_breakdown" in df.columns:
            st.plotly_chart(violin_sensors(df), use_container_width=True)
    with tabs[5]: st.plotly_chart(correlation_heatmap(df), use_container_width=True)
    with tabs[6]:
        if all(c in df.columns for c in ["temp_bearing_degC","vibration_h_mms",
                                          "power_consumption_kw","is_breakdown"]):
            st.plotly_chart(scatter_3d(df), use_container_width=True)
    with tabs[7]:
        if "is_breakdown" in df.columns:
            avail = [c for c in ["temp_bearing_degC","temp_motor_degC",
                                  "vibration_h_mms","oil_pressure_bar",
                                  "power_consumption_kw"] if c in df.columns]
            c1,c2 = st.columns(2)
            xc = c1.selectbox("X axis", avail, index=0)
            yc = c2.selectbox("Y axis", avail, index=2)
            st.plotly_chart(sensor_vs_breakdown_scatter(df,xc,yc), use_container_width=True)
    with tabs[8]:
        if "transaction_date" in df.columns:
            sc = [c for c in ["vibration_h_mms","temp_bearing_degC","oil_pressure_bar"]
                  if c in df.columns]
            col = st.selectbox("Sensor", sc)
            st.plotly_chart(rolling_anomaly_chart(df,col), use_container_width=True)
    with tabs[9]:
        if "breakdown_flag" in df.columns:
            st.plotly_chart(monthly_breakdown_rate(df), use_container_width=True)


# =============================================================================
# PREDICT BREAKDOWN
# =============================================================================
elif page == "🤖  Predict Breakdown":
    st.title("🤖 Predict Machine Breakdown")
    if not model_ok:
        st.warning("Train a model first (Upload & Train page).")
        st.stop()

    tab1,tab2 = st.tabs(["Single Prediction","Batch Predict"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            asset_tag    = st.text_input("Asset Tag","CNC-001")
            machine_type = st.selectbox("Machine Type",
                ["CNC Lathe","Hydraulic Press","Belt Conveyor","Screw Compressor","EOT Crane"])
            temp_bearing = st.slider("Bearing Temp (C)",30.0,120.0,65.0,0.5)
            temp_motor   = st.slider("Motor Temp (C)",30.0,130.0,75.0,0.5)
            vibration_h  = st.slider("H-Vibration (mm/s)",0.0,15.0,2.5,0.1)
            vibration_v  = st.slider("V-Vibration (mm/s)",0.0,12.0,2.0,0.1)
        with col2:
            oil_pressure = st.slider("Oil Pressure (bar)",0.0,150.0,5.5,0.1)
            load_pct     = st.slider("Load (%)",0.0,100.0,65.0,1.0)
            shaft_rpm    = st.slider("Shaft RPM",0.0,4000.0,1200.0,10.0)
            power_kw     = st.slider("Power (kW)",0.0,100.0,25.0,0.5)

        payload = dict(asset_tag=asset_tag, machine_type=machine_type,
            temp_bearing_degC=temp_bearing, temp_motor_degC=temp_motor,
            vibration_h_mms=vibration_h, vibration_v_mms=vibration_v,
            oil_pressure_bar=oil_pressure, load_pct=load_pct,
            shaft_rpm=shaft_rpm, power_consumption_kw=power_kw)

        if st.button("Predict", type="primary"):
            with st.spinner("Running prediction..."):
                try:
                    result = predict_single(payload)
                    prob = result["probability"]
                    risk = result["risk_level"]
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Prediction","BREAKDOWN" if result["prediction"] else "NORMAL")
                    c2.metric("Probability",f"{prob:.1%}")
                    c3.metric("Risk Level",risk)
                    c4.metric("Anomaly Score",f"{result.get('anomaly_score',0):.3f}")
                    st.plotly_chart(risk_gauge(prob), use_container_width=True)
                    css = {"CRITICAL":"alert-critical","HIGH":"alert-warning",
                           "MEDIUM":"alert-warning","LOW":"alert-ok"}.get(risk,"alert-ok")
                    msg = {"CRITICAL":"CRITICAL RISK – Immediate inspection required.",
                           "HIGH":"HIGH RISK – Schedule inspection within 24 hours.",
                           "MEDIUM":"MEDIUM RISK – Monitor closely.",
                           "LOW":"LOW RISK – Normal operation."}.get(risk,"")
                    st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)
                    with st.expander("Full details"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        up2 = st.file_uploader("Upload for batch prediction",
            type=["csv","xlsx","json","parquet"], key="batch_pred")
        if up2:
            fb = up2.read()
            if st.button("Run Batch Predict", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        df2 = parse_upload(up2.name, fb)
                        res = predict_batch(df2)
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Total", len(res))
                        c2.metric("Breakdowns", int(res["prediction"].sum()))
                        c3.metric("Critical", int((res["risk_level"]=="CRITICAL").sum()))
                        st.dataframe(res.head(500), use_container_width=True)
                        st.plotly_chart(prediction_results_chart(
                            res[["probability","risk_level"]].to_dict("records")),
                            use_container_width=True)
                        st.download_button("Download CSV",
                            res.to_csv(index=False),"predictions.csv","text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")


# =============================================================================
# LIVE IoT MONITOR  – fully self-contained, no backend needed
# =============================================================================
elif page == "🔴  Live IoT Monitor":
    import plotly.graph_objects as go
    from backend.iot_simulator import MACHINES, RANGES

    st.title("🔴 Live IoT Sensor Monitor")
    st.markdown("In-browser real-time sensor simulation with live breakdown prediction.")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        selected_asset = st.selectbox("Asset",
            ["CNC-001","HYD-001","BLT-001","CMP-001","EOT-001"])
    with col_ctrl2:
        refresh_rate = st.selectbox("Refresh (s)", [1, 2, 3], index=1)

    run = st.toggle("Start Live Feed", value=False)

    ph_fleet  = st.empty()
    ph_kpis   = st.empty()
    ph_chart  = st.empty()
    ph_pred   = st.empty()

    # Per-asset history stored in session state
    if "iot_hist" not in st.session_state:
        st.session_state.iot_hist = {
            m["asset_tag"]: defaultdict(lambda: deque(maxlen=80)) for m in MACHINES
        }
    if "iot_pred_hist" not in st.session_state:
        st.session_state.iot_pred_hist = defaultdict(lambda: deque(maxlen=80))
    if "iot_sim" not in st.session_state:
        st.session_state.iot_sim = {
            m["asset_tag"]: {"degradation": 0.0, "tick": 0} for m in MACHINES
        }

    def _sim_reading(machine: dict, sim_state: dict) -> dict:
        tag   = machine["asset_tag"]
        mtype = machine["machine_type"]
        r     = RANGES[mtype]
        s     = sim_state[tag]
        s["tick"] += 1
        if random.random() < 0.003:
            s["degradation"] = 0.0
        s["degradation"] = min(1.0, s["degradation"] + random.uniform(0, 0.006))
        deg = s["degradation"]

        def noisy(lo, hi, bias=0.0):
            return round(random.uniform(lo, hi) +
                         random.gauss(0, (hi-lo)*0.05) + bias*(hi-lo), 2)

        tb  = noisy(*r["tb"],  bias=deg*0.4)
        tm  = noisy(*r["tm"],  bias=deg*0.5)
        vh  = max(0, noisy(*r["vh"], bias=deg*0.6))
        vv  = max(0, noisy(*r["vv"], bias=deg*0.5))
        op  = max(0, noisy(*r["op"], bias=-deg*0.3))
        lp  = max(0, min(100, noisy(*r["lp"])))
        rpm = max(0, noisy(*r["rpm"], bias=-deg*0.1))
        pw  = max(0, noisy(*r["pw"],  bias=deg*0.2))
        if random.random() < 0.01*(1+deg*3):
            vh *= random.uniform(1.5, 2.5)
            tb *= random.uniform(1.1, 1.3)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "asset_tag": tag, "machine_type": mtype,
            "temp_bearing_degC": round(tb,2), "temp_motor_degC": round(tm,2),
            "vibration_h_mms": round(vh,2),   "vibration_v_mms": round(vv,2),
            "oil_pressure_bar": round(op,2),   "load_pct": round(lp,1),
            "shaft_rpm": round(rpm,0),         "power_consumption_kw": round(pw,2),
            "degradation_index": round(deg,3),
        }

    if run:
        for tick in range(600):
            time.sleep(refresh_rate)
            all_readings = [_sim_reading(m, st.session_state.iot_sim) for m in MACHINES]
            for r in all_readings:
                tag = r["asset_tag"]
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        st.session_state.iot_hist[tag][k].append(v)

            snap = next((r for r in all_readings if r["asset_tag"]==selected_asset), {})
            hist_snap = {k: list(v) for k,v in st.session_state.iot_hist[selected_asset].items()}

            # Live ML prediction
            pred = {}
            if model_ok and snap:
                try:
                    pred = predict_single({
                        "asset_tag": snap["asset_tag"],
                        "machine_type": snap["machine_type"],
                        "temp_bearing_degC": snap["temp_bearing_degC"],
                        "temp_motor_degC": snap["temp_motor_degC"],
                        "vibration_h_mms": snap["vibration_h_mms"],
                        "vibration_v_mms": snap["vibration_v_mms"],
                        "oil_pressure_bar": snap["oil_pressure_bar"],
                        "load_pct": snap["load_pct"],
                        "shaft_rpm": snap["shaft_rpm"],
                        "power_consumption_kw": snap["power_consumption_kw"],
                    })
                    st.session_state.iot_pred_hist[selected_asset].append(
                        {"probability": pred["probability"], "risk_level": pred["risk_level"]})
                except Exception:
                    pass

            with ph_fleet.container():
                st.plotly_chart(live_all_assets_chart(all_readings), use_container_width=True)

            with ph_kpis.container():
                k1,k2,k3,k4,k5 = st.columns(5)
                k1.metric("Bearing Temp",  f"{snap.get('temp_bearing_degC',0):.1f} C")
                k2.metric("H-Vibration",   f"{snap.get('vibration_h_mms',0):.2f} mm/s")
                k3.metric("Oil Pressure",  f"{snap.get('oil_pressure_bar',0):.1f} bar")
                k4.metric("Power",         f"{snap.get('power_consumption_kw',0):.1f} kW")
                k5.metric("Degradation",   f"{snap.get('degradation_index',0):.1%}")

            with ph_chart.container():
                col_c, col_g = st.columns([3,1])
                with col_c:
                    st.plotly_chart(live_sensor_chart(hist_snap, selected_asset),
                                    use_container_width=True)
                with col_g:
                    st.plotly_chart(degradation_gauge(
                        snap.get("degradation_index",0), selected_asset),
                        use_container_width=True)
                    if pred:
                        st.plotly_chart(risk_gauge(pred.get("probability",0)),
                                        use_container_width=True)

            with ph_pred.container():
                ph = list(st.session_state.iot_pred_hist[selected_asset])
                if ph:
                    st.plotly_chart(live_prediction_timeline(ph), use_container_width=True)
                if pred:
                    risk = pred.get("risk_level","LOW")
                    css = {"CRITICAL":"alert-critical","HIGH":"alert-warning",
                           "MEDIUM":"alert-warning","LOW":"alert-ok"}.get(risk,"alert-ok")
                    st.markdown(
                        f'<div class="{css}">Live: {risk} | Prob: {pred.get("probability",0):.1%}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("Toggle **Start Live Feed** to begin in-browser simulation.")
        st.plotly_chart(live_all_assets_chart([]), use_container_width=True)


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================
elif page == "🔍  SHAP Explainability":
    st.title("🔍 SHAP Model Explainability")
    if not model_ok:
        st.warning("Train a model first.")
        st.stop()

    uploaded = st.file_uploader("Upload dataset for SHAP analysis",
                                type=["csv","xlsx","json","parquet"])
    if uploaded:
        file_bytes = uploaded.read()
        if st.button("Compute SHAP", type="primary"):
            with st.spinner("Computing SHAP values (30-60s)..."):
                try:
                    result = compute_shap(parse_upload(uploaded.name, file_bytes))
                    st.success("SHAP analysis complete!")
                    col1,col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(feature_importance_chart(result["feature_importance"]),
                                        use_container_width=True)
                    with col2:
                        import plotly.graph_objects as go
                        top = sorted(result["feature_importance"].items(),
                                     key=lambda x: x[1], reverse=True)[:5]
                        fig = go.Figure(go.Bar(x=[v for _,v in top], y=[k for k,_ in top],
                            orientation="h", marker_color="#e74c3c"))
                        fig.update_layout(title="Top 5 Risk Drivers",
                                          template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    shap_df = pd.DataFrame(result["shap_values"][:20],
                                           columns=result["feature_names"])
                    st.dataframe(shap_df.style.background_gradient(cmap="RdYlGn", axis=None),
                                 use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")


# =============================================================================
# HEALTH SCORE
# =============================================================================
elif page == "❤️  Health Score":
    import plotly.graph_objects as go
    st.title("❤️ Machine Health Score")
    st.markdown("Weighted 0-100 health index. Radar chart shows which component is degrading.")

    tab1,tab2 = st.tabs(["Single Asset","Fleet Upload"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            asset_tag    = st.text_input("Asset Tag","CNC-001",key="hs_tag")
            machine_type = st.selectbox("Machine Type",
                ["CNC Lathe","Hydraulic Press","Belt Conveyor","Screw Compressor","EOT Crane"],
                key="hs_mtype")
            temp_bearing = st.slider("Bearing Temp (C)",30.0,120.0,68.0,key="hs_tb")
            temp_motor   = st.slider("Motor Temp (C)",30.0,130.0,78.0,key="hs_tm")
            vibration_h  = st.slider("H-Vibration (mm/s)",0.0,15.0,3.2,key="hs_vh")
        with col2:
            vibration_v  = st.slider("V-Vibration (mm/s)",0.0,12.0,2.5,key="hs_vv")
            oil_pressure = st.slider("Oil Pressure (bar)",0.0,150.0,5.2,key="hs_op")
            load_pct     = st.slider("Load (%)",0.0,100.0,65.0,key="hs_lp")
            shaft_rpm    = st.slider("Shaft RPM",0.0,4000.0,1200.0,key="hs_rpm")
            power_kw     = st.slider("Power (kW)",0.0,100.0,24.0,key="hs_pw")

        if st.button("Compute Health Score", type="primary"):
            payload = dict(asset_tag=asset_tag, machine_type=machine_type,
                temp_bearing_degC=temp_bearing, temp_motor_degC=temp_motor,
                vibration_h_mms=vibration_h, vibration_v_mms=vibration_v,
                oil_pressure_bar=oil_pressure, load_pct=load_pct,
                shaft_rpm=shaft_rpm, power_consumption_kw=power_kw)
            try:
                hs = compute_health_score(payload)
                color = "#2ecc71" if hs.health_score>=75 else "#f39c12" if hs.health_score>=50 else "#e74c3c"
                c1,c2,c3 = st.columns(3)
                c1.metric("Health Score", f"{hs.health_score}/100")
                c2.metric("Status", hs.status)
                if model_ok:
                    try:
                        pred = predict_single(payload)
                        c3.metric("Breakdown Prob", f"{pred['probability']:.1%}")
                    except Exception:
                        pass

                col_g,col_r = st.columns(2)
                with col_g:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=hs.health_score,
                        title={"text": f"Health – {asset_tag}"},
                        gauge={"axis":{"range":[0,100]}, "bar":{"color":color,"thickness":0.3},
                               "steps":[{"range":[0,50],"color":"#3a0a0a"},
                                        {"range":[50,75],"color":"#3a3a0a"},
                                        {"range":[75,100],"color":"#0a3a0a"}],
                               "threshold":{"line":{"color":"white","width":3},"value":75}}))
                    fig.update_layout(height=300, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                with col_r:
                    comp = hs.component_scores
                    fig2 = go.Figure(go.Scatterpolar(
                        r=list(comp.values()), theta=list(comp.keys()),
                        fill="toself", line_color=color,
                        fillcolor="rgba(46,204,113,0.15)"))
                    fig2.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                        title="Component Radar", template="plotly_dark", height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Recommendations")
                for rec in hs.recommendations:
                    st.markdown(f"- {rec}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        up_fleet = st.file_uploader("Upload fleet dataset",
            type=["csv","xlsx","json","parquet"], key="hs_fleet")
        if up_fleet:
            try:
                df_fleet = parse_upload(up_fleet.name, up_fleet.read())
                fleet_df = compute_fleet_health(df_fleet)
                c1,c2,c3 = st.columns(3)
                c1.metric("Assets", fleet_df["asset_tag"].nunique()
                           if "asset_tag" in fleet_df.columns else len(fleet_df))
                c2.metric("Avg Health", f"{fleet_df['health_score'].mean():.1f}/100")
                c3.metric("Critical", int((fleet_df["health_score"]<50).sum()))
                st.dataframe(fleet_df.style.background_gradient(
                    subset=["health_score"], cmap="RdYlGn"), use_container_width=True)
                col1,col2 = st.columns(2)
                with col1:
                    import plotly.express as px
                    if "asset_tag" in fleet_df.columns:
                        fig = px.bar(fleet_df.groupby("asset_tag")["health_score"].mean().reset_index(),
                            x="asset_tag", y="health_score", color="health_score",
                            color_continuous_scale="RdYlGn", range_color=[0,100],
                            title="Fleet Health Scores", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if "machine_type" in fleet_df.columns and "asset_tag" in fleet_df.columns:
                        st.plotly_chart(fleet_health_treemap(fleet_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TIME-TO-FAILURE
# =============================================================================
elif page == "⏱️  Time-to-Failure":
    st.title("⏱️ Time-to-Failure Prediction")
    st.markdown("Estimates days until breakdown per asset using degradation trend analysis.")

    up_ttf = st.file_uploader("Upload historical dataset",
        type=["csv","xlsx","json","parquet"], key="ttf_file")
    if up_ttf:
        if st.button("Estimate TTF", type="primary"):
            with st.spinner("Analysing degradation trends..."):
                try:
                    df_ttf = parse_upload(up_ttf.name, up_ttf.read())
                    results = fleet_ttf(df_ttf)
                    df_res = pd.DataFrame(results)
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Assets", len(df_res))
                    c2.metric("Immediate Action",
                              int((df_res["urgency"]=="Immediate").sum()))
                    c3.metric("Avg Days", f"{df_res['estimated_days'].mean():.0f}")
                    c4.metric("Min Days", f"{df_res['estimated_days'].min():.0f}")

                    col1,col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(ttf_gantt(results), use_container_width=True)
                    with col2:
                        import plotly.express as px
                        fig = px.scatter(df_res, x="degradation_rate", y="estimated_days",
                            color="urgency", text="asset_tag",
                            color_discrete_map={"Immediate":"#e74c3c","This Week":"#e67e22",
                                                "This Month":"#f1c40f","Routine":"#2ecc71"},
                            title="Degradation Rate vs TTF", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                    def _color_urgency(val):
                        return {"Immediate":"background-color:#5a0a0a",
                                "This Week":"background-color:#5a3a0a",
                                "This Month":"background-color:#3a3a0a",
                                "Routine":"background-color:#0a3a0a"}.get(val,"")
                    st.dataframe(df_res.style.applymap(_color_urgency, subset=["urgency"]),
                                 use_container_width=True)
                    st.download_button("Download TTF Report",
                        df_res.to_csv(index=False), "ttf_report.csv","text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")


# =============================================================================
# MODEL REGISTRY
# =============================================================================
elif page == "📈  Model Registry":
    st.title("📈 Model Registry & Comparison")
    if st.button("Refresh", type="primary"):
        try:
            registry   = get_registry()
            comparison = compare_models()
            active     = get_active_version()
            st.success(f"Active model: **{active}**")
            if not comparison.empty:
                c1,c2,c3 = st.columns(3)
                best = comparison.loc[comparison["auc"].idxmax()]
                c1.metric("Best AUC",      f"{best['auc']:.4f}")
                c2.metric("Best Accuracy", f"{comparison['accuracy'].max():.4f}")
                c3.metric("Versions",      len(comparison))
                col1,col2 = st.columns(2)
                with col1:
                    st.plotly_chart(model_comparison_chart(comparison), use_container_width=True)
                with col2:
                    import plotly.express as px
                    fig = px.bar(comparison, x="version", y="n_train",
                        title="Training Dataset Size per Version",
                        template="plotly_dark", color="auc",
                        color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    comparison.style
                        .highlight_max(subset=["accuracy","auc","f1"], color="#1a3a1a")
                        .highlight_min(subset=["accuracy","auc","f1"], color="#3a0a0a"),
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
    tab1,tab2 = st.tabs(["Single Machine","All Machines Comparison"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Machine Type",list(MACHINE_DEFAULTS.keys()))
            mttr         = st.number_input("Mean Time to Repair (hours)",1.0,72.0,8.0,0.5)
            annual_bd    = st.number_input("Annual Breakdowns",1,100,12)
        with col2:
            hourly_prod  = st.number_input("Hourly Production Value (Rs.)",1000,500000,25000,1000)
            repair_cost  = st.number_input("Avg Repair Cost per Event (Rs.)",1000,1000000,80000,1000)
            pdm_cost     = st.number_input("PdM System Annual Cost (Rs.)",10000,5000000,500000,10000)
        currency = st.radio("Currency",["INR (Rs.)","USD ($)"], horizontal=True)

        if st.button("Calculate", type="primary"):
            try:
                result = calculate_downtime(machine_type, mttr, hourly_prod,
                                            repair_cost, annual_bd, pdm_cost)
                rate = 83.5 if "USD" in currency else 1
                sym  = "$" if "USD" in currency else "Rs."
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Loss / Event",   f"{sym}{result.production_loss_inr/rate:,.0f}")
                c2.metric("Total / Event",  f"{sym}{result.total_cost_inr/rate:,.0f}")
                c3.metric("Annual BD Cost", f"{sym}{result.annual_bd_cost_inr/rate:,.0f}")
                c4.metric("Savings w/ PdM", f"{sym}{result.savings_with_pdm_inr/rate:,.0f}",
                           delta=f"ROI: {result.roi_percent}%")
                col1,col2 = st.columns(2)
                with col1:
                    st.plotly_chart(downtime_cost_chart(result.__dict__), use_container_width=True)
                with col2:
                    st.plotly_chart(roi_waterfall(result.__dict__), use_container_width=True)
                css = "alert-ok" if result.roi_percent>0 else "alert-warning"
                st.markdown(f'<div class="{css}">ROI: {result.roi_percent}%</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.plotly_chart(multi_machine_cost_comparison(), use_container_width=True)


# =============================================================================
# AI ADVISOR  (rule-based, no LLM key needed on Streamlit Cloud)
# =============================================================================
elif page == "🧠  AI Advisor":
    st.title("🧠 AI Prescriptive Maintenance Advisor")
    st.markdown("Rule-based prescriptive recommendations. Add `OPENAI_API_KEY` in Streamlit secrets for GPT-4o.")
    if not model_ok:
        st.warning("Train a model first.")
        st.stop()

    col1,col2 = st.columns(2)
    with col1:
        asset_tag    = st.text_input("Asset Tag","CNC-001")
        machine_type = st.selectbox("Machine Type",
            ["CNC Lathe","Hydraulic Press","Belt Conveyor","Screw Compressor","EOT Crane"])
        temp_bearing = st.number_input("Bearing Temp (C)",30.0,120.0,78.0)
        temp_motor   = st.number_input("Motor Temp (C)",30.0,130.0,88.0)
        vibration_h  = st.number_input("H-Vibration (mm/s)",0.0,15.0,5.2)
    with col2:
        vibration_v  = st.number_input("V-Vibration (mm/s)",0.0,12.0,4.1)
        oil_pressure = st.number_input("Oil Pressure (bar)",0.0,150.0,4.2)
        load_pct     = st.number_input("Load (%)",0.0,100.0,82.0)
        shaft_rpm    = st.number_input("Shaft RPM",0.0,4000.0,1450.0)
        power_kw     = st.number_input("Power (kW)",0.0,100.0,38.0)

    if st.button("Get Advice", type="primary"):
        payload = dict(asset_tag=asset_tag, machine_type=machine_type,
            temp_bearing_degC=temp_bearing, temp_motor_degC=temp_motor,
            vibration_h_mms=vibration_h, vibration_v_mms=vibration_v,
            oil_pressure_bar=oil_pressure, load_pct=load_pct,
            shaft_rpm=shaft_rpm, power_consumption_kw=power_kw)
        with st.spinner("Analysing..."):
            try:
                pred = predict_single(payload)
                hs   = compute_health_score(payload)
                prob = pred["probability"]
                risk = pred["risk_level"]

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Prediction",   "BREAKDOWN" if pred["prediction"] else "NORMAL")
                c2.metric("Probability",  f"{prob:.1%}")
                c3.metric("Risk Level",   risk)
                c4.metric("Health Score", f"{hs.health_score}/100")

                col1,col2 = st.columns([1,2])
                with col1:
                    st.plotly_chart(risk_gauge(prob), use_container_width=True)
                with col2:
                    st.subheader("Recommendations")
                    for rec in hs.recommendations:
                        st.markdown(f"- {rec}")
                    st.divider()
                    st.markdown(f"""
**Root Cause Analysis**
- Bearing temp: {'⚠️ Elevated' if temp_bearing>75 else '✅ Normal'}
- Vibration: {'⚠️ High' if vibration_h>4 else '✅ Normal'}
- Oil pressure: {'⚠️ Low' if oil_pressure<4 else '✅ Normal'}

**Immediate Actions**
- {'🔴 Shutdown and inspect immediately' if risk=='CRITICAL' else '🟡 Schedule inspection within 24h' if risk=='HIGH' else '🟢 Continue monitoring'}

**Preventive Schedule**
- Daily: Visual inspection + temperature check
- Weekly: Vibration measurement + oil level
- Monthly: Full bearing inspection + alignment
                    """)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# AUDIT LOGS
# =============================================================================
elif page == "📋  Audit Logs":
    st.title("📋 Audit Logs")
    st.markdown("System actions logged for compliance. Connect Supabase in `.env` for persistent storage.")

    try:
        registry = get_registry()
        if registry:
            df_logs = pd.DataFrame([{
                "version": r["version"],
                "trained_at": r["trained_at"][:19],
                "action": "model_trained",
                "accuracy": r["metrics"].get("accuracy",0),
                "auc": r["metrics"].get("auc",0),
                "rows": r["dataset"].get("rows",0),
            } for r in registry])
            c1,c2 = st.columns(2)
            c1.metric("Training Runs", len(df_logs))
            c2.metric("Active Version", get_active_version())
            st.dataframe(df_logs, use_container_width=True)
            st.download_button("Export CSV", df_logs.to_csv(index=False),
                               "audit_log.csv","text/csv")
        else:
            st.info("No training runs yet. Train a model to see logs here.")
    except Exception as e:
        st.info(f"Local logs unavailable: {e}")

    st.divider()
    st.markdown("""
| Action | Trigger |
|--------|---------|
| `model_trained` | Dataset uploaded + model trained |
| `batch_predict` | Batch prediction file uploaded |
| `predict` | Single sensor reading predicted |
| `ttf_analysis` | Time-to-failure computed |
    """)
