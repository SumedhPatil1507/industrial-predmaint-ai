"""
PredMaint AI v3 - Premium Self-Contained Edition
Live data always running. No file upload required. Works on Streamlit Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from collections import defaultdict, deque
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="PredMaint AI", page_icon="🔧",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
[data-testid="stSidebar"]*{color:#c9d1d9!important;}
.block-container{padding-top:1.5rem;}
.kpi{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid #30363d;
     border-radius:12px;padding:16px 20px;text-align:center;
     transition:transform .2s,border-color .2s;}
.kpi:hover{transform:translateY(-3px);border-color:#58a6ff;}
.kv{font-size:1.8rem;font-weight:700;color:#58a6ff;}
.kl{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;}
.kd{font-size:.8rem;color:#3fb950;margin-top:3px;}
.ac{background:linear-gradient(135deg,#3d0000,#5a0a0a);border:1px solid #e74c3c;
    border-radius:8px;padding:10px 14px;color:#ff6b6b;font-weight:600;
    animation:pulse 2s infinite;}
.aw{background:linear-gradient(135deg,#2d1a00,#3d2500);border:1px solid #e67e22;
    border-radius:8px;padding:10px 14px;color:#ffa94d;}
.ao{background:linear-gradient(135deg,#001a0d,#002b15);border:1px solid #2ecc71;
    border-radius:8px;padding:10px 14px;color:#69db7c;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(231,76,60,.4);}
    50%{box-shadow:0 0 0 8px rgba(231,76,60,0);}}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;}
</style>""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
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
from frontend.data_engine import generate_historical_data, generate_live_tick, what_if_sensitivity
from backend.health_score import compute_health_score, compute_fleet_health
from backend.ttf_predictor import fleet_ttf
from backend.downtime_calculator import calculate_downtime, MACHINE_DEFAULTS
from backend.ml_engine import (train_model, predict_single, predict_batch,
                                compute_shap, model_exists, FEATURE_COLS, TARGET)
from backend.file_parser import parse_upload, validate_columns
from backend.model_registry import get_registry, compare_models, get_active_version
from backend.iot_simulator import MACHINES

# ── Auto-bootstrap: generate data + train model on first load ─────────────────
if "df" not in st.session_state or st.session_state.df is None:
    with st.spinner("Generating 2-year synthetic dataset..."):
        st.session_state.df = generate_historical_data(n_days=730)

if "model_trained" not in st.session_state:
    st.session_state.model_trained = model_exists()

if not st.session_state.model_trained:
    with st.spinner("Auto-training model on synthetic data..."):
        try:
            m = train_model(st.session_state.df)
            st.session_state.model_trained = True
            st.session_state.train_metrics = m
        except Exception as e:
            st.warning(f"Auto-train failed: {e}")

model_ok = st.session_state.model_trained

# ── Live data state ───────────────────────────────────────────────────────────
for key, default in [
    ("live_hist", {m["asset_tag"]: defaultdict(lambda: deque(maxlen=100)) for m in MACHINES}),
    ("live_pred_hist", defaultdict(lambda: deque(maxlen=100))),
    ("live_tick", 0),
    ("alert_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 PredMaint AI")
    st.caption("Industrial Predictive Maintenance")
    st.divider()
    df_rows = len(st.session_state.df) if st.session_state.df is not None else 0
    st.markdown(f"**Model:** {'✅ Ready' if model_ok else '⚠️ Training...'}")
    st.markdown(f"**Dataset:** {df_rows:,} rows")
    st.markdown(f"**Live Ticks:** {st.session_state.live_tick}")
    st.divider()
    page = st.radio("", [
        "🏠 Dashboard",
        "🔴 Live Fleet Monitor",
        "📊 EDA Explorer",
        "🤖 Predict Breakdown",
        "🔍 SHAP Explainability",
        "❤️ Health Score",
        "⏱️ Time-to-Failure",
        "💰 Downtime Calculator",
        "🧪 What-If Simulator",
        "📈 Model Registry",
        "📁 Upload Custom Data",
        "📚 References & Citations",
        "ℹ️ About",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("v3.0 · Live Data Always On")
    st.markdown("""
<div style='font-size:0.7rem;color:#8b949e;line-height:1.6'>
Built with<br>
🐍 Python 3.12<br>
⚡ Streamlit 1.35<br>
🤖 scikit-learn 1.4<br>
📊 Plotly 5.22<br>
🔍 SHAP 0.45
</div>""", unsafe_allow_html=True)


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "🏠 Dashboard":
    st.markdown("## 🔧 Industrial Machine Predictive Maintenance")
    st.markdown("**Live data · AI predictions · Zero file upload required**")
    st.divider()

    live_readings = generate_live_tick()
    for r in live_readings:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                st.session_state.live_hist[r["asset_tag"]][k].append(v)
    st.session_state.live_tick += 1

    df = st.session_state.df
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl,dlt) in zip([c1,c2,c3,c4,c5],[
        (f"{len(df):,}", "Dataset Rows", "Auto-generated"),
        ("5", "Machine Types", "10 assets"),
        (f"{df['breakdown_flag'].mean():.1%}", "Breakdown Rate", "Live data"),
        ("40-60%", "Downtime Reduction", "With PdM"),
        (f"{st.session_state.live_tick}", "Live Ticks", "Auto-refresh"),
    ]):
        col.markdown(f'<div class="kpi"><div class="kv">{val}</div>'
                     f'<div class="kl">{lbl}</div><div class="kd">{dlt}</div></div>',
                     unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Fleet Status")
        fleet_rows = []
        for r in live_readings:
            if model_ok:
                try:
                    pred = predict_single(r)
                    hs   = compute_health_score(r)
                    risk = pred["risk_level"]
                    fleet_rows.append({
                        "Asset": r["asset_tag"],
                        "Machine": r["machine_type"],
                        "Health": f"{hs.health_score:.0f}/100",
                        "Breakdown Prob": f"{pred['probability']:.1%}",
                        "Risk": risk,
                        "Degradation": f"{r['degradation_index']:.1%}",
                    })
                    if risk in ("CRITICAL","HIGH"):
                        alert = f"{datetime.now().strftime('%H:%M:%S')} | {r['asset_tag']} | {risk} | {pred['probability']:.1%}"
                        if alert not in st.session_state.alert_log:
                            st.session_state.alert_log.insert(0, alert)
                            st.session_state.alert_log = st.session_state.alert_log[:20]
                except Exception:
                    pass
        if fleet_rows:
            st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, height=280)

    with col2:
        st.subheader("Live Sensor Snapshot")
        st.plotly_chart(live_all_assets_chart(live_readings), use_container_width=True)

    if st.session_state.alert_log:
        st.divider()
        st.subheader("Recent Alerts")
        for a in st.session_state.alert_log[:5]:
            st.markdown(f'<div class="aw">{a}</div>', unsafe_allow_html=True)

    st.info("Live data auto-generated. Go to **Live Fleet Monitor** for real-time charts. "
            "Upload your own CSV in **Upload Custom Data**.")
    time.sleep(2)
    st.rerun()


# =============================================================================
# LIVE FLEET MONITOR
# =============================================================================
elif page == "🔴 Live Fleet Monitor":
    st.title("🔴 Live Fleet Monitor")
    st.markdown("Real-time sensor stream. Auto-refreshes every 2 seconds.")

    selected = st.selectbox("Focus Asset",
        ["CNC-001","HYD-001","BLT-001","CMP-001","EOT-001"])

    live_readings = generate_live_tick()
    for r in live_readings:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                st.session_state.live_hist[r["asset_tag"]][k].append(v)
    st.session_state.live_tick += 1

    snap = next((r for r in live_readings if r["asset_tag"] == selected), {})
    hist = {k: list(v) for k, v in st.session_state.live_hist[selected].items()}

    pred = {}
    if model_ok and snap:
        try:
            pred = predict_single(snap)
            st.session_state.live_pred_hist[selected].append(
                {"probability": pred["probability"], "risk_level": pred["risk_level"]})
        except Exception:
            pass

    st.plotly_chart(live_all_assets_chart(live_readings), use_container_width=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Bearing Temp",  f"{snap.get('temp_bearing_degC',0):.1f} C")
    k2.metric("H-Vibration",   f"{snap.get('vibration_h_mms',0):.2f} mm/s")
    k3.metric("Oil Pressure",  f"{snap.get('oil_pressure_bar',0):.1f} bar")
    k4.metric("Power",         f"{snap.get('power_consumption_kw',0):.1f} kW")
    k5.metric("Degradation",   f"{snap.get('degradation_index',0):.1%}")

    col_c, col_g = st.columns([3, 1])
    with col_c:
        st.plotly_chart(live_sensor_chart(hist, selected), use_container_width=True)
    with col_g:
        st.plotly_chart(degradation_gauge(snap.get("degradation_index",0), selected),
                        use_container_width=True)
        if pred:
            st.plotly_chart(risk_gauge(pred.get("probability",0)), use_container_width=True)

    ph = list(st.session_state.live_pred_hist[selected])
    if ph:
        st.plotly_chart(live_prediction_timeline(ph), use_container_width=True)

    if pred:
        risk = pred.get("risk_level","LOW")
        css = {"CRITICAL":"ac","HIGH":"aw","MEDIUM":"aw","LOW":"ao"}.get(risk,"ao")
        st.markdown(f'<div class="{css}">Live: {risk} | Prob: {pred.get("probability",0):.1%}</div>',
                    unsafe_allow_html=True)

    with st.expander("Raw sensor data"):
        st.dataframe(pd.DataFrame([snap]), use_container_width=True)

    time.sleep(2)
    st.rerun()


# =============================================================================
# EDA EXPLORER  (uses auto-generated data, file upload optional)
# =============================================================================
elif page == "📊 EDA Explorer":
    st.title("📊 Interactive EDA Explorer")

    df = st.session_state.df
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
                sel = st.multiselect("Machine Type", list(df["machine_type"].unique()),
                                     default=list(df["machine_type"].unique()))
                df = df[df["machine_type"].isin(sel)]
        with fc2:
            if "transaction_date" in df.columns:
                dr = st.date_input("Date Range",
                    [df["transaction_date"].min().date(), df["transaction_date"].max().date()])
                if len(dr) == 2:
                    df = df[(df["transaction_date"].dt.date >= dr[0]) &
                            (df["transaction_date"].dt.date <= dr[1])]
        with fc3:
            if "is_breakdown" in df.columns:
                bf = st.multiselect("Status", ["Normal","Breakdown"],
                                    default=["Normal","Breakdown"])
                df = df[df["is_breakdown"].isin(bf)]
    st.caption(f"Showing {len(df):,} rows from auto-generated dataset")

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
# PREDICT BREAKDOWN  (uses live data by default)
# =============================================================================
elif page == "🤖 Predict Breakdown":
    st.title("🤖 Predict Machine Breakdown")
    if not model_ok:
        st.warning("Model training in progress...")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Live Sensor Input","Manual Input","Batch Predict"])

    with tab1:
        st.markdown("Uses latest live sensor reading from selected asset.")
        live_asset = st.selectbox("Asset",
            ["CNC-001","HYD-001","BLT-001","CMP-001","EOT-001"], key="pred_live_asset")
        live_readings = generate_live_tick()
        snap = next((r for r in live_readings if r["asset_tag"] == live_asset), {})
        if snap:
            st.dataframe(pd.DataFrame([snap]), use_container_width=True)
            if st.button("Predict from Live Data", type="primary"):
                try:
                    result = predict_single(snap)
                    hs = compute_health_score(snap)
                    prob = result["probability"]
                    risk = result["risk_level"]
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Prediction", "BREAKDOWN" if result["prediction"] else "NORMAL")
                    c2.metric("Probability", f"{prob:.1%}")
                    c3.metric("Risk Level", risk)
                    c4.metric("Health Score", f"{hs.health_score:.0f}/100")
                    col1,col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(risk_gauge(prob), use_container_width=True)
                    with col2:
                        import plotly.graph_objects as go
                        comp = hs.component_scores
                        color = "#2ecc71" if hs.health_score>=75 else "#f39c12" if hs.health_score>=50 else "#e74c3c"
                        fig = go.Figure(go.Scatterpolar(
                            r=list(comp.values()), theta=list(comp.keys()),
                            fill="toself", line_color=color, fillcolor="rgba(46,204,113,0.15)"))
                        fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                            title="Health Radar", template="plotly_dark", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    css = {"CRITICAL":"ac","HIGH":"aw","MEDIUM":"aw","LOW":"ao"}.get(risk,"ao")
                    msg = {"CRITICAL":"CRITICAL – Immediate shutdown required.",
                           "HIGH":"HIGH – Inspect within 24 hours.",
                           "MEDIUM":"MEDIUM – Monitor closely.",
                           "LOW":"LOW – Normal operation."}.get(risk,"")
                    st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)
                    st.subheader("Recommendations")
                    for rec in hs.recommendations:
                        st.markdown(f"- {rec}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
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
                css = {"CRITICAL":"ac","HIGH":"aw","MEDIUM":"aw","LOW":"ao"}.get(risk,"ao")
                st.markdown(f'<div class="{css}">{risk}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        up2 = st.file_uploader("Upload CSV for batch prediction",
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
# SHAP EXPLAINABILITY  (uses auto-generated data)
# =============================================================================
elif page == "🔍 SHAP Explainability":
    st.title("🔍 SHAP Model Explainability")
    if not model_ok:
        st.warning("Model training in progress...")
        st.stop()

    st.markdown("Using auto-generated dataset. Upload your own below for custom analysis.")
    use_custom = st.checkbox("Upload custom dataset")
    df_shap = st.session_state.df

    if use_custom:
        up = st.file_uploader("Upload dataset", type=["csv","xlsx","json","parquet"])
        if up:
            df_shap = parse_upload(up.name, up.read())

    if st.button("Compute SHAP", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                result = compute_shap(df_shap)
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
                shap_vals = result["shap_values"]
                shap_df = pd.DataFrame(shap_vals[:20], columns=result["feature_names"])
                st.dataframe(shap_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# HEALTH SCORE  (uses live data by default)
# =============================================================================
elif page == "❤️ Health Score":
    import plotly.graph_objects as go
    st.title("❤️ Machine Health Score")

    tab1, tab2 = st.tabs(["Live Asset Score","Fleet Overview"])

    with tab1:
        live_asset = st.selectbox("Asset",
            ["CNC-001","HYD-001","BLT-001","CMP-001","EOT-001"], key="hs_live")
        live_readings = generate_live_tick()
        snap = next((r for r in live_readings if r["asset_tag"] == live_asset), {})

        if snap:
            hs = compute_health_score(snap)
            color = "#2ecc71" if hs.health_score>=75 else "#f39c12" if hs.health_score>=50 else "#e74c3c"
            c1,c2,c3 = st.columns(3)
            c1.metric("Health Score", f"{hs.health_score:.0f}/100")
            c2.metric("Status", hs.status)
            c3.metric("Degradation", f"{snap.get('degradation_index',0):.1%}")

            col_g, col_r = st.columns(2)
            with col_g:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=hs.health_score,
                    title={"text": f"Health – {live_asset}"},
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
                    fill="toself", line_color=color, fillcolor="rgba(46,204,113,0.15)"))
                fig2.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
                    title="Component Radar", template="plotly_dark", height=350)
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Recommendations")
            for rec in hs.recommendations:
                st.markdown(f"- {rec}")

    with tab2:
        live_readings = generate_live_tick()
        fleet_rows = []
        for r in live_readings:
            hs = compute_health_score(r)
            fleet_rows.append({
                "asset_tag": r["asset_tag"],
                "machine_type": r["machine_type"],
                "health_score": hs.health_score,
                "status": hs.status,
            })
        fleet_df = pd.DataFrame(fleet_rows)
        c1,c2,c3 = st.columns(3)
        c1.metric("Avg Health", f"{fleet_df['health_score'].mean():.1f}/100")
        c2.metric("Critical Assets", int((fleet_df["health_score"]<50).sum()))
        c3.metric("Healthy Assets", int((fleet_df["health_score"]>=75).sum()))
        st.dataframe(fleet_df, use_container_width=True)
        import plotly.express as px
        fig = px.bar(fleet_df, x="asset_tag", y="health_score", color="health_score",
            color_continuous_scale="RdYlGn", range_color=[0,100],
            title="Fleet Health Scores", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fleet_health_treemap(fleet_df), use_container_width=True)


# =============================================================================
# TIME-TO-FAILURE  (uses auto-generated data)
# =============================================================================
elif page == "⏱️ Time-to-Failure":
    st.title("⏱️ Time-to-Failure Prediction")
    st.markdown("Using auto-generated 2-year dataset. Upload custom data below.")

    use_custom = st.checkbox("Upload custom dataset", key="ttf_custom")
    df_ttf = st.session_state.df

    if use_custom:
        up = st.file_uploader("Upload dataset", type=["csv","xlsx","json","parquet"], key="ttf_up")
        if up:
            df_ttf = parse_upload(up.name, up.read())

    if st.button("Estimate TTF for All Assets", type="primary"):
        with st.spinner("Analysing degradation trends..."):
            try:
                results = fleet_ttf(df_ttf)
                df_res = pd.DataFrame(results)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Assets", len(df_res))
                c2.metric("Immediate Action", int((df_res["urgency"]=="Immediate").sum()))
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
                def _cu(val):
                    return {"Immediate":"background-color:#5a0a0a",
                            "This Week":"background-color:#5a3a0a",
                            "This Month":"background-color:#3a3a0a",
                            "Routine":"background-color:#0a3a0a"}.get(val,"")
                st.dataframe(df_res.style.map(_cu, subset=["urgency"]),
                             use_container_width=True)
                st.download_button("Download TTF Report",
                    df_res.to_csv(index=False), "ttf_report.csv","text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# DOWNTIME CALCULATOR
# =============================================================================
elif page == "💰 Downtime Calculator":
    st.title("💰 Downtime Cost Calculator")
    tab1, tab2 = st.tabs(["Single Machine","All Machines Comparison"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Machine Type", list(MACHINE_DEFAULTS.keys()))
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
                css = "ao" if result.roi_percent > 0 else "aw"
                st.markdown(f'<div class="{css}">ROI: {result.roi_percent}% – '
                            f'{"PdM investment justified" if result.roi_percent>0 else "Reduce PdM cost"}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.plotly_chart(multi_machine_cost_comparison(), use_container_width=True)


# =============================================================================
# WHAT-IF SIMULATOR  (premium feature)
# =============================================================================
elif page == "🧪 What-If Simulator":
    import plotly.graph_objects as go
    import plotly.express as px
    st.title("🧪 What-If Simulator")
    st.markdown("Vary a sensor value and see how breakdown probability changes. "
                "Uses latest live reading as baseline.")

    if not model_ok:
        st.warning("Model training in progress...")
        st.stop()

    live_asset = st.selectbox("Asset",
        ["CNC-001","HYD-001","BLT-001","CMP-001","EOT-001"], key="wi_asset")
    live_readings = generate_live_tick()
    base = next((r for r in live_readings if r["asset_tag"] == live_asset), {})

    if base:
        st.subheader("Baseline Reading")
        st.dataframe(pd.DataFrame([base]), use_container_width=True)

        sensor = st.selectbox("Vary this sensor", [
            "temp_bearing_degC","temp_motor_degC","vibration_h_mms",
            "vibration_v_mms","oil_pressure_bar","load_pct","power_consumption_kw"])

        pct_range = st.slider("% change range", -80, 200, (-50, 100), 10)
        steps = st.slider("Number of steps", 5, 30, 15)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running sensitivity analysis..."):
                pct_changes = list(np.linspace(pct_range[0], pct_range[1], steps))
                df_wi = what_if_sensitivity(base, sensor, pct_changes)

                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_wi["value"], y=df_wi["probability"],
                        mode="lines+markers", name="Breakdown Prob",
                        line=dict(color="#e74c3c", width=2),
                        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"))
                    fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                                  annotation_text="Alert Threshold")
                    base_val = base.get(sensor, 0)
                    fig.add_vline(x=base_val, line_dash="dot", line_color="#58a6ff",
                                  annotation_text="Current")
                    fig.update_layout(title=f"Breakdown Prob vs {sensor}",
                                      xaxis_title=sensor, yaxis_title="Probability",
                                      yaxis=dict(range=[0,1]),
                                      template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.bar(df_wi, x="change_pct", y="probability",
                        color="probability", color_continuous_scale="RdYlGn_r",
                        range_color=[0,1], title="Probability by % Change",
                        template="plotly_dark")
                    fig2.add_hline(y=0.5, line_dash="dash", line_color="white")
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Sensitivity Table")
                df_wi["risk"] = df_wi["probability"].apply(
                    lambda p: "CRITICAL" if p>=0.75 else "HIGH" if p>=0.5
                              else "MEDIUM" if p>=0.25 else "LOW")
                st.dataframe(df_wi, use_container_width=True)

                safe_range = df_wi[df_wi["probability"] < 0.25]
                if not safe_range.empty:
                    st.markdown(f'<div class="ao">Safe range for {sensor}: '
                                f'{safe_range["value"].min():.2f} – {safe_range["value"].max():.2f}</div>',
                                unsafe_allow_html=True)


# =============================================================================
# MODEL REGISTRY
# =============================================================================
elif page == "📈 Model Registry":
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
# UPLOAD CUSTOM DATA  (optional, not required)
# =============================================================================
elif page == "📁 Upload Custom Data":
    st.title("📁 Upload Custom Data")
    st.markdown("Optional — the app works without uploading anything. "
                "Upload your own dataset to replace the auto-generated data.")

    uploaded = st.file_uploader("Drop your dataset here",
        type=["csv","tsv","xlsx","xls","json","parquet","feather"])

    if uploaded:
        file_bytes = uploaded.read()
        st.success(f"File received: `{uploaded.name}` ({len(file_bytes)/1024:.1f} KB)")
        action = st.radio("Action", ["Replace Dataset + Retrain","Batch Predict Only"],
                          horizontal=True)

        if st.button("Execute", type="primary"):
            with st.spinner("Processing..."):
                try:
                    df = parse_upload(uploaded.name, file_bytes)
                    if action == "Replace Dataset + Retrain":
                        ok, missing = validate_columns(df, FEATURE_COLS + [TARGET])
                        if not ok:
                            st.error(f"Missing columns: {missing}")
                        else:
                            st.session_state.df = df
                            metrics = train_model(df)
                            st.session_state.model_trained = True
                            st.session_state.train_metrics = metrics
                            st.success("Dataset replaced and model retrained!")
                            c1,c2,c3,c4 = st.columns(4)
                            c1.metric("Accuracy", f"{metrics.get('accuracy',0):.3f}")
                            c2.metric("AUC-ROC",  f"{metrics.get('auc',0):.3f}")
                            c3.metric("Precision", f"{metrics.get('precision_breakdown',0):.3f}")
                            c4.metric("Recall",    f"{metrics.get('recall_breakdown',0):.3f}")
                    else:
                        result_df = predict_batch(df)
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Total Rows", len(result_df))
                        c2.metric("Breakdowns", int(result_df["prediction"].sum()))
                        c3.metric("Critical", int((result_df["risk_level"]=="CRITICAL").sum()))
                        st.dataframe(result_df.head(500), use_container_width=True)
                        st.plotly_chart(prediction_results_chart(
                            result_df[["probability","risk_level"]].to_dict("records")),
                            use_container_width=True)
                        st.download_button("Download Results CSV",
                            result_df.to_csv(index=False),"predictions.csv","text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

        try:
            df_preview = parse_upload(uploaded.name, file_bytes)
            st.subheader("Data Preview")
            st.dataframe(df_preview.head(20), use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows", f"{len(df_preview):,}")
            c2.metric("Columns", len(df_preview.columns))
            c3.metric("Missing", int(df_preview.isnull().sum().sum()))
            c4.metric("Breakdown Rate",
                f"{df_preview['breakdown_flag'].mean():.1%}"
                if "breakdown_flag" in df_preview.columns else "N/A")
        except Exception:
            pass


# =============================================================================
# REFERENCES & CITATIONS
# =============================================================================
elif page == "📚 References & Citations":
    from frontend.page_references import render as render_refs
    render_refs()


# =============================================================================
# ABOUT
# =============================================================================
elif page == "ℹ️ About":
    from frontend.page_about import render as render_about
    render_about()
