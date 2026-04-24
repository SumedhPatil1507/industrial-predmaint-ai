"""All interactive Plotly charts – zero static matplotlib. Premium tier."""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = px.colors.qualitative.Plotly
BREAKDOWN_COLORS = {"Normal": "#2ecc71", "Breakdown": "#e74c3c"}
DARK = "plotly_dark"

# ── EDA ───────────────────────────────────────────────────────────────────────

def histogram_grid(df: pd.DataFrame) -> go.Figure:
    cols = ["temp_bearing_degC", "temp_motor_degC", "vibration_h_mms", "oil_pressure_bar"]
    titles = ["Bearing Temp (C)", "Motor Temp (C)", "H-Vibration (mm/s)", "Oil Pressure (bar)"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    colors = ["#f39c12", "#e74c3c", "#3498db", "#27ae60"]
    for i, (col, color) in enumerate(zip(cols, colors)):
        r, c = divmod(i, 2)
        fig.add_trace(go.Histogram(x=df[col], name=col, marker_color=color,
                                   opacity=0.8, nbinsx=60), row=r+1, col=c+1)
    fig.update_layout(title="Key Sensor Distributions", height=500,
                      showlegend=False, template=DARK)
    return fig


def lineplot_trends(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("transaction_date").agg(
        temp_bearing_degC=("temp_bearing_degC", "mean"),
        vibration_h_mms=("vibration_h_mms", "mean"),
        oil_pressure_bar=("oil_pressure_bar", "mean"),
        breakdown_flag=("breakdown_flag", "sum"),
    ).reset_index()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Bearing Temp", "H-Vibration", "Oil Pressure", "Daily Breakdowns"])
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["temp_bearing_degC"],
                             line=dict(color="#f39c12", width=1.5), name="Bearing Temp",
                             fill="tozeroy", fillcolor="rgba(243,156,18,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["vibration_h_mms"],
                             line=dict(color="#3498db", width=1.5), name="Vibration",
                             fill="tozeroy", fillcolor="rgba(52,152,219,0.1)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["oil_pressure_bar"],
                             line=dict(color="#27ae60", width=1.5), name="Oil Pressure",
                             fill="tozeroy", fillcolor="rgba(39,174,96,0.1)"), row=3, col=1)
    fig.add_trace(go.Bar(x=daily["transaction_date"], y=daily["breakdown_flag"],
                         marker_color="#e74c3c", name="Breakdowns"), row=4, col=1)
    fig.update_layout(height=750, template=DARK, title="Daily Sensor Trends (3+ Years)")
    return fig


def breakdown_by_machine(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["machine_type", "is_breakdown"]).size().reset_index(name="count")
    fig = px.bar(grp, x="machine_type", y="count", color="is_breakdown",
                 color_discrete_map=BREAKDOWN_COLORS, barmode="group",
                 title="Breakdowns by Machine Type", template=DARK,
                 text="count")
    fig.update_traces(textposition="outside")
    return fig


def pie_charts(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "pie"}]*2]*2,
                        subplot_titles=["Breakdown Rate", "Machine Types",
                                        "Criticality", "Work Order Types"])
    bd = df["breakdown_flag"].value_counts()
    fig.add_trace(go.Pie(labels=["Normal", "Breakdown"], values=bd.values,
                         marker_colors=["#2ecc71", "#e74c3c"], hole=0.4), row=1, col=1)
    mt = df["machine_type"].value_counts()
    fig.add_trace(go.Pie(labels=mt.index, values=mt.values, hole=0.4), row=1, col=2)
    if "criticality" in df.columns:
        cr = df["criticality"].value_counts()
        fig.add_trace(go.Pie(labels=cr.index, values=cr.values, hole=0.4), row=2, col=1)
    if "wo_type" in df.columns:
        wo = df["wo_type"].dropna().value_counts()
        fig.add_trace(go.Pie(labels=wo.index, values=wo.values, hole=0.4), row=2, col=2)
    fig.update_layout(height=600, template=DARK, title="Categorical Distributions")
    return fig


def boxplot_sensors(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Bearing Temp vs Breakdown", "Vibration by Machine",
                                        "Oil Pressure vs Breakdown", "Power by Criticality"])
    for label, color in [("Normal", "#2ecc71"), ("Breakdown", "#e74c3c")]:
        sub = df[df["is_breakdown"] == label]
        fig.add_trace(go.Box(y=sub["temp_bearing_degC"], name=label,
                             marker_color=color, boxmean=True), row=1, col=1)
    for i, mt in enumerate(df["machine_type"].unique()):
        sub = df[df["machine_type"] == mt]
        fig.add_trace(go.Box(y=sub["vibration_h_mms"], name=mt,
                             marker_color=COLORS[i % len(COLORS)], boxmean=True), row=1, col=2)
    for label, color in [("Normal", "#2ecc71"), ("Breakdown", "#e74c3c")]:
        sub = df[df["is_breakdown"] == label]
        fig.add_trace(go.Box(y=sub["oil_pressure_bar"], name=label,
                             marker_color=color, boxmean=True), row=2, col=1)
    if "criticality" in df.columns:
        for i, cr in enumerate(df["criticality"].unique()):
            sub = df[df["criticality"] == cr]
            fig.add_trace(go.Box(y=sub["power_consumption_kw"], name=str(cr),
                                 marker_color=COLORS[i % len(COLORS)], boxmean=True), row=2, col=2)
    fig.update_layout(height=600, template=DARK, title="Boxplot Analysis")
    return fig


def violin_sensors(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Motor Temp by Breakdown", "Vibration by Machine"])
    for label, color in [("Normal", "#2ecc71"), ("Breakdown", "#e74c3c")]:
        sub = df[df["is_breakdown"] == label]
        fig.add_trace(go.Violin(y=sub["temp_motor_degC"], name=label,
                                box_visible=True, meanline_visible=True,
                                fillcolor=color, opacity=0.6), row=1, col=1)
    for i, mt in enumerate(df["machine_type"].unique()):
        sub = df[df["machine_type"] == mt]
        fig.add_trace(go.Violin(y=sub["vibration_h_mms"], name=mt,
                                box_visible=True, meanline_visible=True,
                                fillcolor=COLORS[i % len(COLORS)], opacity=0.6), row=1, col=2)
    fig.update_layout(height=500, template=DARK, title="Violin Plots")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(title="Correlation Heatmap", height=580, template=DARK)
    return fig


def scatter_3d(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter_3d(sample, x="temp_bearing_degC", y="vibration_h_mms",
                        z="power_consumption_kw", color="is_breakdown",
                        color_discrete_map=BREAKDOWN_COLORS, opacity=0.6,
                        title="3D Sensor Space", template=DARK)
    return fig


def monthly_breakdown_rate(df: pd.DataFrame) -> go.Figure:
    if "month" not in df.columns:
        df = df.copy()
        df["month"] = pd.to_datetime(df["transaction_date"]).dt.month
    mb = df.groupby("month")["breakdown_flag"].mean().reset_index()
    fig = px.bar(mb, x="month", y="breakdown_flag",
                 title="Average Breakdown Rate by Month",
                 labels={"breakdown_flag": "Breakdown Rate"},
                 color="breakdown_flag", color_continuous_scale="Reds",
                 template=DARK, text=mb["breakdown_flag"].round(3))
    fig.update_traces(textposition="outside")
    return fig


def sensor_vs_breakdown_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    sample = df.sample(min(8000, len(df)), random_state=42)
    fig = px.scatter(sample, x=x_col, y=y_col, color="is_breakdown",
                     color_discrete_map=BREAKDOWN_COLORS, opacity=0.5,
                     marginal_x="histogram", marginal_y="histogram",
                     title=f"{x_col} vs {y_col}", template=DARK)
    return fig


def rolling_anomaly_chart(df: pd.DataFrame, col: str = "vibration_h_mms") -> go.Figure:
    daily = df.groupby("transaction_date")[col].mean().reset_index()
    daily["rolling_mean"] = daily[col].rolling(30, min_periods=1).mean()
    daily["rolling_std"] = daily[col].rolling(30, min_periods=1).std().fillna(0)
    daily["upper"] = daily["rolling_mean"] + 2 * daily["rolling_std"]
    daily["lower"] = daily["rolling_mean"] - 2 * daily["rolling_std"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily[col],
                             name="Raw", line=dict(color="#3498db", width=1), opacity=0.4))
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["rolling_mean"],
                             name="30d Mean", line=dict(color="#f39c12", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([daily["transaction_date"], daily["transaction_date"][::-1]]),
        y=pd.concat([daily["upper"], daily["lower"][::-1]]),
        fill="toself", fillcolor="rgba(243,156,18,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="2-Sigma Band"
    ))
    fig.update_layout(title=f"Rolling Mean + Anomaly Band: {col}",
                      template=DARK, height=400)
    return fig


# ── ML Charts ─────────────────────────────────────────────────────────────────

def feature_importance_chart(importance: dict) -> go.Figure:
    # Ensure all values are plain Python floats
    clean = {k: float(v) if not isinstance(v, (int, float)) else v
             for k, v in importance.items()}
    items = sorted(clean.items(), key=lambda x: x[1])
    colors = [f"rgba(26,188,156,{0.4 + 0.6*i/max(len(items)-1,1)})" for i in range(len(items))]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for _, v in items],
        textposition="outside"))
    fig.update_layout(title="SHAP Feature Importance (Mean |SHAP|)", height=420,
                      xaxis_title="Mean |SHAP|", template=DARK)
    return fig


def prediction_results_chart(results: list[dict]) -> go.Figure:
    df = pd.DataFrame(results)
    if "probability" not in df.columns:
        return go.Figure()
    color_map = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e67e22", "CRITICAL": "#e74c3c"}
    fig = px.histogram(df, x="probability", color="risk_level",
                       nbins=40, title="Prediction Probability Distribution",
                       color_discrete_map=color_map, template=DARK)
    fig.add_vline(x=0.5, line_dash="dash", line_color="white",
                  annotation_text="Decision Threshold")
    return fig


def risk_gauge(probability: float) -> go.Figure:
    color = "#e74c3c" if probability > 0.5 else "#f39c12" if probability > 0.25 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Breakdown Probability (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 25], "color": "#0d2b0d"},
                {"range": [25, 50], "color": "#2b2b0d"},
                {"range": [50, 75], "color": "#2b0d0d"},
                {"range": [75, 100], "color": "#4a0505"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": 50},
        },
    ))
    fig.update_layout(height=300, template=DARK, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def roc_curve_chart(fpr: list, tpr: list, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})",
                             line=dict(color="#1abc9c", width=2), fill="tozeroy",
                             fillcolor="rgba(26,188,156,0.1)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                             line=dict(color="gray", dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      template=DARK, height=400)
    return fig


# ── Live IoT Charts ───────────────────────────────────────────────────────────

def live_sensor_chart(history: dict, asset_tag: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Temperature (C)", "Vibration (mm/s)", "Power (kW)"],
                        vertical_spacing=0.08)
    ticks = list(range(len(history.get("temp_bearing_degC", []))))
    fig.add_trace(go.Scatter(x=ticks, y=history.get("temp_bearing_degC", []),
                             name="Bearing", line=dict(color="#f39c12", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("temp_motor_degC", []),
                             name="Motor", line=dict(color="#e74c3c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("vibration_h_mms", []),
                             name="H-Vib", line=dict(color="#3498db", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("vibration_v_mms", []),
                             name="V-Vib", line=dict(color="#9b59b6", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("power_consumption_kw", []),
                             name="Power", line=dict(color="#1abc9c", width=2),
                             fill="tozeroy", fillcolor="rgba(26,188,156,0.1)"), row=3, col=1)
    fig.update_layout(height=520, template=DARK,
                      title=f"Live Sensor Feed - {asset_tag}",
                      legend=dict(orientation="h", y=-0.05))
    return fig


def live_all_assets_chart(all_latest: list[dict]) -> go.Figure:
    """Bar chart comparing latest health across all assets."""
    if not all_latest:
        return go.Figure()
    tags = [r.get("asset_tag", "") for r in all_latest]
    temps = [r.get("temp_bearing_degC", 0) for r in all_latest]
    vibs = [r.get("vibration_h_mms", 0) for r in all_latest]
    degs = [r.get("degradation_index", 0) * 100 for r in all_latest]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Bearing Temp (C)", "H-Vibration (mm/s)", "Degradation (%)"])
    fig.add_trace(go.Bar(x=tags, y=temps, marker_color="#f39c12", name="Temp"), row=1, col=1)
    fig.add_trace(go.Bar(x=tags, y=vibs, marker_color="#3498db", name="Vibration"), row=1, col=2)
    fig.add_trace(go.Bar(x=tags, y=degs, marker_color="#e74c3c", name="Degradation"), row=1, col=3)
    fig.update_layout(height=350, template=DARK, title="Fleet Live Snapshot",
                      showlegend=False)
    return fig


def degradation_gauge(degradation: float, asset_tag: str) -> go.Figure:
    color = "#e74c3c" if degradation > 0.7 else "#f39c12" if degradation > 0.4 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=degradation * 100,
        title={"text": f"Degradation - {asset_tag}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 40], "color": "#0d2b0d"},
                {"range": [40, 70], "color": "#2b2b0d"},
                {"range": [70, 100], "color": "#4a0505"},
            ],
        },
    ))
    fig.update_layout(height=250, template=DARK, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def live_prediction_timeline(pred_history: list[dict]) -> go.Figure:
    """Shows rolling breakdown probability over time for live feed."""
    if not pred_history:
        return go.Figure()
    df = pd.DataFrame(pred_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["probability"],
        mode="lines+markers", name="Breakdown Prob",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                  annotation_text="Alert Threshold")
    fig.update_layout(title="Live Breakdown Probability Trend",
                      yaxis=dict(range=[0, 1]), template=DARK, height=300)
    return fig


# ── Downtime Charts ───────────────────────────────────────────────────────────

def downtime_cost_chart(result: dict) -> go.Figure:
    labels = ["Production Loss", "Repair Cost", "Annual BD Cost", "Savings with PdM"]
    values = [result["production_loss_inr"], result["repair_cost_inr"],
              result["annual_bd_cost_inr"], result["savings_with_pdm_inr"]]
    colors = ["#e74c3c", "#e67e22", "#c0392b", "#2ecc71"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"Rs.{v:,.0f}" for v in values], textposition="outside"))
    fig.update_layout(title="Downtime Cost Analysis (INR)",
                      yaxis_title="INR", template=DARK, height=420)
    return fig


def roi_waterfall(result: dict) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Annual BD Cost", "PdM System Cost", "Savings (50%)", "Net Benefit"],
        y=[result["annual_bd_cost_inr"], -500000, result["savings_with_pdm_inr"], 0],
        connector={"line": {"color": "rgb(63,63,63)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
        text=[f"Rs.{abs(v):,.0f}" for v in
              [result["annual_bd_cost_inr"], -500000, result["savings_with_pdm_inr"], 0]],
        textposition="outside",
    ))
    fig.update_layout(title="ROI Waterfall", template=DARK, height=420)
    return fig


def multi_machine_cost_comparison() -> go.Figure:
    from backend.downtime_calculator import calculate_downtime, MACHINE_DEFAULTS
    machines = list(MACHINE_DEFAULTS.keys())
    costs = [calculate_downtime(m).annual_bd_cost_inr for m in machines]
    savings = [calculate_downtime(m).savings_with_pdm_inr for m in machines]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Annual BD Cost", x=machines, y=costs, marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Savings with PdM", x=machines, y=savings, marker_color="#2ecc71"))
    fig.update_layout(barmode="group", title="Cost Comparison Across Machine Types",
                      template=DARK, height=420, yaxis_title="INR")
    return fig


# ── Model Registry Charts ─────────────────────────────────────────────────────

def model_comparison_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for metric, color in [("accuracy", "#3498db"), ("auc", "#2ecc71"),
                           ("f1", "#f39c12"), ("precision", "#9b59b6"), ("recall", "#e74c3c")]:
        if metric in df.columns:
            fig.add_trace(go.Scatter(x=df["version"], y=df[metric], name=metric.upper(),
                                     mode="lines+markers", line=dict(color=color, width=2),
                                     marker=dict(size=8)))
    fig.update_layout(title="Model Performance Across Versions",
                      yaxis=dict(range=[0, 1.05]), template=DARK, height=420)
    return fig


# ── Fleet Overview ────────────────────────────────────────────────────────────

def fleet_health_treemap(fleet_df: pd.DataFrame) -> go.Figure:
    fig = px.treemap(fleet_df, path=["machine_type", "asset_tag"],
                     values="health_score", color="health_score",
                     color_continuous_scale="RdYlGn", range_color=[0, 100],
                     title="Fleet Health Treemap", template=DARK)
    return fig


def ttf_gantt(results: list[dict]) -> go.Figure:
    """Horizontal bar showing urgency timeline per asset."""
    if not results:
        return go.Figure()
    df = pd.DataFrame(results)
    color_map = {"Immediate": "#e74c3c", "This Week": "#e67e22",
                 "This Month": "#f1c40f", "Routine": "#2ecc71"}
    fig = px.bar(df, x="estimated_days", y="asset_tag", color="urgency",
                 color_discrete_map=color_map, orientation="h",
                 title="Time-to-Failure by Asset", template=DARK,
                 text="estimated_days")
    fig.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
    fig.update_layout(height=350, xaxis_title="Days to Failure")
    return fig
