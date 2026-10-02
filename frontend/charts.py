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


# ── Maintenance Copilot Interactive Charts ────────────────────────────────────

def copilot_langgraph_flow_chart(graph_trace: list[dict] | None = None) -> go.Figure:
    """
    Interactive Sankey-style flow diagram visualising the 6-node LangGraph
    Maintenance Copilot execution.  Completed nodes are highlighted; trace
    summaries appear in hover tooltips.
    """
    nodes = [
        "Telemetry\nIngestion",
        "SHAP\nAttribution",
        "Vector RAG\nRetrieval",
        "Novelty\nGate",
        "Work Order\nDrafting",
        "Escalation\nDispatch",
    ]
    # Build a status map from the trace (step numbers are 1-indexed)
    status_map: dict[int, str] = {}
    summary_map: dict[int, str] = {}
    if graph_trace:
        for t in graph_trace:
            idx = t.get("step", 0) - 1
            status_map[idx] = t.get("status", "PENDING")
            summary_map[idx] = t.get("summary", "")

    # Node positions on an arc
    import math
    n = len(nodes)
    x_pos = [i / (n - 1) for i in range(n)]
    y_pos = [0.5 + 0.18 * math.sin(math.pi * i / (n - 1)) for i in range(n)]

    completed_color = "#2ecc71"
    pending_color   = "#4a5568"
    active_color    = "#3b82f6"

    node_colors = []
    for i in range(n):
        s = status_map.get(i, "PENDING")
        node_colors.append(
            completed_color if s == "COMPLETED" else
            active_color    if s == "RUNNING"   else
            pending_color
        )

    hover_texts = [
        f"<b>Node {i+1}: {nodes[i].replace(chr(10),' ')}</b><br>"
        f"Status: {status_map.get(i, 'PENDING')}<br>"
        f"{summary_map.get(i, 'Awaiting execution...')}"
        for i in range(n)
    ]

    fig = go.Figure()

    # Draw curved edges between consecutive nodes
    for i in range(n - 1):
        mx = (x_pos[i] + x_pos[i + 1]) / 2
        my = max(y_pos[i], y_pos[i + 1]) + 0.12
        edge_color = completed_color if (status_map.get(i, "") == "COMPLETED" and
                                         status_map.get(i + 1, "") != "PENDING") else "#374151"
        # Quadratic bezier approximated with many intermediate points
        t_vals = np.linspace(0, 1, 40)
        bx = (1 - t_vals)**2 * x_pos[i] + 2 * (1 - t_vals) * t_vals * mx + t_vals**2 * x_pos[i + 1]
        by = (1 - t_vals)**2 * y_pos[i] + 2 * (1 - t_vals) * t_vals * my  + t_vals**2 * y_pos[i + 1]
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines",
            line=dict(color=edge_color, width=2.5, dash="dot" if edge_color == "#374151" else "solid"),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Arrowhead marker at destination
        fig.add_trace(go.Scatter(
            x=[x_pos[i + 1]], y=[y_pos[i + 1]],
            mode="markers",
            marker=dict(symbol="arrow", size=12, color=edge_color,
                        angleref="previous", angle=0),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node circles
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        marker=dict(
            size=52,
            color=node_colors,
            line=dict(color="#1f2937", width=3),
            symbol="circle",
        ),
        text=[f"<b>{n_label.replace(chr(10), '<br>')}</b>" for n_label in nodes],
        textposition="middle center",
        textfont=dict(size=9, color="#ffffff"),
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#3b82f6",
                        font=dict(color="#c9d1d9", size=11)),
        showlegend=False,
    ))

    # Step number badges
    for i in range(n):
        fig.add_annotation(
            x=x_pos[i], y=y_pos[i] - 0.14,
            text=f"Step {i + 1}",
            font=dict(size=9, color="#8b949e"),
            showarrow=False,
        )

    # Legend entries
    for label, color in [("Completed ✅", completed_color),
                          ("Running ⚡", active_color),
                          ("Pending ⏳", pending_color)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="<b>🕸️ LangGraph Maintenance Copilot — Agent Execution Flow</b>",
            font=dict(size=14, color="#c9d1d9"),
        ),
        template=DARK,
        height=310,
        xaxis=dict(visible=False, range=[-0.08, 1.08]),
        yaxis=dict(visible=False, range=[0.18, 0.90]),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02,
                    font=dict(size=10, color="#8b949e")),
    )
    return fig


def copilot_shap_waterfall_chart(top_features: list[dict]) -> go.Figure:
    """Horizontal bar chart showing positive/negative SHAP feature contributions for the anomaly."""
    if not top_features:
        return go.Figure()
    
    df = pd.DataFrame(top_features)
    df["color"] = df["shap_impact"].apply(lambda x: "#e74c3c" if x > 0 else "#2ecc71")
    df = df.iloc[::-1]  # reverse for top-down display

    # Rich hover template
    hover_texts = [
        f"<b>{row['display_name']}</b><br>"
        f"SHAP Impact: <b>{row['shap_impact']:+.4f}</b><br>"
        f"Raw Value: <b>{row['raw_value']}</b><br>"
        f"Direction: <b>{row['direction'].upper()}</b><br>"
        f"{'⬆ Drives failure risk' if row['shap_impact'] > 0 else '⬇ Reduces failure risk'}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df["shap_impact"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["color"],
            line=dict(color="#1f2937", width=1),
            opacity=0.85,
        ),
        text=[f"{val:+.3f}" for val in df["shap_impact"]],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#e74c3c",
                        font=dict(color="#c9d1d9", size=11)),
    ))

    # Zero baseline
    fig.add_vline(x=0, line_color="#4a5568", line_width=1.5)

    fig.update_layout(
        title=dict(
            text="<b>SHAP Local Attribution — Failure Drivers</b>",
            font=dict(size=13, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="SHAP Impact (+ = Drives Failure, − = Reduces Risk)",
            title_font=dict(size=10, color="#8b949e"),
            tickfont=dict(size=9, color="#8b949e"),
            gridcolor="#21262d",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10, color="#c9d1d9"),
        ),
        template=DARK,
        height=320,
        margin=dict(l=10, r=50, t=45, b=30),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    return fig


def copilot_confidence_gauge(confidence_score: float, is_novel: bool = False) -> go.Figure:
    """Plotly gauge for RAG SOP match confidence and novel anomaly detection."""
    conf_pct = confidence_score * 100
    color = "#e74c3c" if is_novel or conf_pct < 55 else ("#f39c12" if conf_pct < 75 else "#2ecc71")
    label = "⚠️ NOVEL ANOMALY" if is_novel else ("MODERATE MATCH" if conf_pct < 75 else "HIGH CONFIDENCE")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=conf_pct,
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        delta={"reference": 52, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"},
               "font": {"size": 11}},
        title={
            "text": (
                f"<b>SOP Match Confidence</b><br>"
                f"<span style='font-size:0.75em;color:#8b949e'>{label}</span>"
            ),
            "font": {"size": 13},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#4a5568",
                "tickvals": [0, 52, 75, 100],
                "ticktext": ["0%", "52%\nNovelty", "75%\nL2", "100%"],
                "tickfont": {"size": 8, "color": "#8b949e"},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22",
            "borderwidth": 1,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  52], "color": "rgba(231,76,60,0.20)"},
                {"range": [52, 75], "color": "rgba(243,156,18,0.20)"},
                {"range": [75, 100], "color": "rgba(46,204,113,0.20)"},
            ],
            "threshold": {
                "line": {"color": "#f39c12", "width": 3},
                "thickness": 0.75,
                "value": 52,
            },
        },
    ))

    # Tier annotations
    for x_norm, label_txt, color_txt in [
        (0.18, "L3\nNovel", "#e74c3c"),
        (0.50, "L2\nSenior", "#f39c12"),
        (0.82, "L1\nSOP", "#2ecc71"),
    ]:
        fig.add_annotation(
            x=x_norm, y=0.08, xref="paper", yref="paper",
            text=f"<span style='font-size:8px;color:{color_txt}'>{label_txt}</span>",
            showarrow=False, align="center",
        )

    fig.update_layout(
        template=DARK,
        height=270,
        margin=dict(l=20, r=20, t=55, b=10),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    return fig


def copilot_radar_chart(telemetry: dict, machine_type: str = "CNC Lathe") -> go.Figure:
    """Interactive radar/spider chart comparing current anomaly telemetry against nominal baseline."""
    from backend.copilot.shap_explainer import NOMINAL_BASELINES, FEATURE_DISPLAY_NAMES
    
    categories = []
    current_vals_norm = []
    nominal_baseline_norm = []
    raw_hover = []
    raw_vals = []

    for k, (low, high) in NOMINAL_BASELINES.items():
        if k in telemetry:
            val = float(telemetry.get(k, 0.0))
            display = FEATURE_DISPLAY_NAMES.get(k, k).split(" (")[0]
            categories.append(display)
            
            span = max(high - low, 1.0)
            norm_val = max(0, min(150, ((val - low) / span) * 50 + 50))
            current_vals_norm.append(norm_val)
            nominal_baseline_norm.append(50)
            raw_hover.append(f"{k}: {val:.2f} (Nominal: {low:.1f}–{high:.1f})")
            raw_vals.append(val)

    if not categories:
        return go.Figure()

    # Close polygon loop
    categories_closed     = categories + [categories[0]]
    current_closed        = current_vals_norm + [current_vals_norm[0]]
    nominal_closed        = nominal_baseline_norm + [nominal_baseline_norm[0]]

    fig = go.Figure()

    # Nominal band
    fig.add_trace(go.Scatterpolar(
        r=nominal_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(46,204,113,0.12)",
        line=dict(color="#2ecc71", dash="dash", width=1.5),
        name="Nominal Baseline",
        hoverinfo="skip",
    ))

    # Anomaly reading with rich hover
    fig.add_trace(go.Scatterpolar(
        r=current_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(231,76,60,0.25)",
        line=dict(color="#e74c3c", width=2.5),
        name="Current Anomaly State",
        hovertext=[f"<b>{c}</b><br>{h}" for c, h in zip(categories_closed, raw_hover + [raw_hover[0]])],
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#e74c3c",
                        font=dict(color="#c9d1d9", size=11)),
        mode="lines+markers",
        marker=dict(size=6, color="#e74c3c"),
    ))

    # Upper warning band (>100 = out-of-range)
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(categories_closed),
        theta=categories_closed,
        fill="none",
        line=dict(color="#f39c12", width=1, dash="dot"),
        name="Warning Limit (100%)",
        hoverinfo="skip",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 150],
                showticklabels=True,
                tickvals=[0, 50, 100, 150],
                ticktext=["0", "Normal", "⚠️ High", "🚨 Critical"],
                tickfont=dict(size=8, color="#8b949e"),
                gridcolor="#21262d",
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color="#c9d1d9"),
                gridcolor="#21262d",
            ),
            bgcolor="#0d1117",
        ),
        title=dict(
            text=f"<b>Telemetry Deviation Radar — {machine_type}</b>",
            font=dict(size=13, color="#c9d1d9"),
        ),
        template=DARK,
        height=330,
        margin=dict(l=30, r=30, t=50, b=20),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            font=dict(size=9, color="#8b949e"),
        ),
        paper_bgcolor="#0d1117",
    )
    return fig


def copilot_retrieval_similarity_chart(retrieved_manuals: list[dict], threshold: float = 0.52) -> go.Figure:
    """Bar chart showing RAG vector cosine similarity scores for retrieved SOP manuals."""
    if not retrieved_manuals:
        return go.Figure()

    titles = [f"[{d.get('manual_id')}] {d.get('title', '')[:35]}…" for d in retrieved_manuals]
    sims   = [d.get("similarity_score", 0.0) * 100 for d in retrieved_manuals]
    colors = ["#2ecc71" if s >= threshold * 100 else "#e74c3c" for s in sims]

    hover_texts = [
        f"<b>{d.get('manual_id')}</b><br>"
        f"{d.get('title')}<br>"
        f"Machine: {d.get('machine_type')}<br>"
        f"Subsystem: {d.get('subsystem')}<br>"
        f"Similarity: <b>{s:.1f}%</b><br>"
        f"MTTR: {d.get('estimated_mttr_hours', 'N/A')} hrs<br>"
        f"{'✅ Above threshold — SOP coverage confirmed' if s >= threshold * 100 else '🚨 Below threshold — Novel signature'}"
        for d, s in zip(retrieved_manuals, sims)
    ]

    fig = go.Figure(go.Bar(
        x=sims,
        y=titles,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#1f2937", width=1),
            opacity=0.85,
        ),
        text=[f"{s:.1f}%" for s in sims],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#3b82f6",
                        font=dict(color="#c9d1d9", size=11)),
    ))

    # Threshold line
    fig.add_vline(
        x=threshold * 100,
        line_dash="dot",
        line_color="#f39c12",
        line_width=2,
        annotation_text=f"Novelty Threshold ({threshold * 100:.0f}%)",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#f39c12"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Vector RAG Retrieval — SOP Match Scores</b>",
            font=dict(size=13, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="Cosine Similarity Score (%)",
            range=[0, 110],
            tickfont=dict(size=9, color="#8b949e"),
            gridcolor="#21262d",
        ),
        yaxis=dict(
            tickfont=dict(size=9, color="#c9d1d9"),
        ),
        template=DARK,
        height=270,
        margin=dict(l=10, r=50, t=45, b=30),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    return fig

