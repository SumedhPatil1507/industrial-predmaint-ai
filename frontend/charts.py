"""All interactive Plotly charts – zero static matplotlib."""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = px.colors.qualitative.Plotly
BREAKDOWN_COLORS = {"Normal": "#2ecc71", "Breakdown": "#e74c3c"}


# ── EDA Charts ───────────────────────────────────────────────────────────────

def histogram_grid(df: pd.DataFrame) -> go.Figure:
    cols = ["temp_bearing_degC", "temp_motor_degC", "vibration_h_mms", "oil_pressure_bar"]
    titles = ["Bearing Temp (°C)", "Motor Temp (°C)", "H-Vibration (mm/s)", "Oil Pressure (bar)"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    color_map = ["#f39c12", "#e74c3c", "#3498db", "#27ae60"]
    for i, (col, color) in enumerate(zip(cols, color_map)):
        r, c = divmod(i, 2)
        fig.add_trace(go.Histogram(x=df[col], name=col, marker_color=color,
                                   opacity=0.75, nbinsx=50), row=r+1, col=c+1)
    fig.update_layout(title="Key Sensor Distributions", height=500,
                      showlegend=False, template="plotly_dark")
    return fig


def lineplot_trends(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("transaction_date").agg(
        temp_bearing_degC=("temp_bearing_degC", "mean"),
        vibration_h_mms=("vibration_h_mms", "mean"),
        oil_pressure_bar=("oil_pressure_bar", "mean"),
        breakdown_flag=("breakdown_flag", "sum"),
    ).reset_index()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Bearing Temp", "H-Vibration",
                                        "Oil Pressure", "Daily Breakdowns"])
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["temp_bearing_degC"],
                             line=dict(color="#f39c12"), name="Bearing Temp"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["vibration_h_mms"],
                             line=dict(color="#3498db"), name="Vibration"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["transaction_date"], y=daily["oil_pressure_bar"],
                             line=dict(color="#27ae60"), name="Oil Pressure"), row=3, col=1)
    fig.add_trace(go.Bar(x=daily["transaction_date"], y=daily["breakdown_flag"],
                         marker_color="#e74c3c", name="Breakdowns"), row=4, col=1)
    fig.update_layout(height=700, template="plotly_dark", title="Daily Sensor Trends")
    return fig


def breakdown_by_machine(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["machine_type", "is_breakdown"]).size().reset_index(name="count")
    fig = px.bar(grp, x="machine_type", y="count", color="is_breakdown",
                 color_discrete_map=BREAKDOWN_COLORS, barmode="group",
                 title="Breakdowns by Machine Type", template="plotly_dark")
    return fig


def pie_charts(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "pie"}]*2]*2,
                        subplot_titles=["Breakdown Rate", "Machine Types",
                                        "Criticality", "Work Order Types"])
    bd = df["breakdown_flag"].value_counts()
    fig.add_trace(go.Pie(labels=["Normal", "Breakdown"], values=bd.values,
                         marker_colors=["#2ecc71", "#e74c3c"]), row=1, col=1)
    mt = df["machine_type"].value_counts()
    fig.add_trace(go.Pie(labels=mt.index, values=mt.values), row=1, col=2)
    if "criticality" in df.columns:
        cr = df["criticality"].value_counts()
        fig.add_trace(go.Pie(labels=cr.index, values=cr.values), row=2, col=1)
    if "wo_type" in df.columns:
        wo = df["wo_type"].dropna().value_counts()
        fig.add_trace(go.Pie(labels=wo.index, values=wo.values), row=2, col=2)
    fig.update_layout(height=600, template="plotly_dark", title="Categorical Distributions")
    return fig


def boxplot_sensors(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Bearing Temp vs Breakdown",
                                        "Vibration by Machine",
                                        "Oil Pressure vs Breakdown",
                                        "Power by Criticality"])
    for label, color in [("Normal", "#2ecc71"), ("Breakdown", "#e74c3c")]:
        sub = df[df["is_breakdown"] == label]
        fig.add_trace(go.Box(y=sub["temp_bearing_degC"], name=label,
                             marker_color=color, showlegend=False), row=1, col=1)
    for i, mt in enumerate(df["machine_type"].unique()):
        sub = df[df["machine_type"] == mt]
        fig.add_trace(go.Box(y=sub["vibration_h_mms"], name=mt,
                             marker_color=COLORS[i % len(COLORS)], showlegend=False), row=1, col=2)
    for label, color in [("Normal", "#2ecc71"), ("Breakdown", "#e74c3c")]:
        sub = df[df["is_breakdown"] == label]
        fig.add_trace(go.Box(y=sub["oil_pressure_bar"], name=label,
                             marker_color=color, showlegend=False), row=2, col=1)
    if "criticality" in df.columns:
        for i, cr in enumerate(df["criticality"].unique()):
            sub = df[df["criticality"] == cr]
            fig.add_trace(go.Box(y=sub["power_consumption_kw"], name=str(cr),
                                 marker_color=COLORS[i % len(COLORS)], showlegend=False), row=2, col=2)
    fig.update_layout(height=600, template="plotly_dark", title="Boxplot Analysis")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Heatmap", height=550, template="plotly_dark")
    return fig


def scatter_3d(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter_3d(sample, x="temp_bearing_degC", y="vibration_h_mms",
                        z="power_consumption_kw", color="is_breakdown",
                        color_discrete_map=BREAKDOWN_COLORS, opacity=0.6,
                        title="3D Sensor Space (Breakdown vs Normal)",
                        template="plotly_dark")
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
                 template="plotly_dark")
    return fig


# ── ML Result Charts ──────────────────────────────────────────────────────────

def feature_importance_chart(importance: dict) -> go.Figure:
    items = sorted(importance.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(x=list(v for _, v in items),
                           y=list(k for k, _ in items),
                           orientation="h", marker_color="#1abc9c"))
    fig.update_layout(title="SHAP Feature Importance", height=400,
                      xaxis_title="Mean |SHAP|", template="plotly_dark")
    return fig


def prediction_results_chart(results: list[dict]) -> go.Figure:
    df = pd.DataFrame(results)
    if "probability" not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x="probability", color="risk_level",
                       nbins=40, title="Prediction Probability Distribution",
                       color_discrete_sequence=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
                       template="plotly_dark")
    return fig


def risk_gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": "Breakdown Probability (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if probability > 0.5 else "#f39c12" if probability > 0.25 else "#2ecc71"},
            "steps": [
                {"range": [0, 25], "color": "#1a3a1a"},
                {"range": [25, 50], "color": "#3a3a1a"},
                {"range": [50, 75], "color": "#3a1a1a"},
                {"range": [75, 100], "color": "#5a0a0a"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": 50},
        },
    ))
    fig.update_layout(height=300, template="plotly_dark")
    return fig


# ── Live IoT Charts ───────────────────────────────────────────────────────────

def live_sensor_chart(history: dict[str, list], asset_tag: str) -> go.Figure:
    """history = {metric: [values...]}"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Temperature (°C)", "Vibration (mm/s)", "Power (kW)"])
    ticks = list(range(len(history.get("temp_bearing_degC", []))))
    fig.add_trace(go.Scatter(x=ticks, y=history.get("temp_bearing_degC", []),
                             name="Bearing", line=dict(color="#f39c12")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("temp_motor_degC", []),
                             name="Motor", line=dict(color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("vibration_h_mms", []),
                             name="H-Vib", line=dict(color="#3498db")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("vibration_v_mms", []),
                             name="V-Vib", line=dict(color="#9b59b6")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=history.get("power_consumption_kw", []),
                             name="Power", line=dict(color="#1abc9c")), row=3, col=1)
    fig.update_layout(height=500, template="plotly_dark",
                      title=f"Live Sensor Feed – {asset_tag}")
    return fig


def degradation_gauge(degradation: float, asset_tag: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=degradation * 100,
        title={"text": f"Degradation Index – {asset_tag}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if degradation > 0.7 else "#f39c12" if degradation > 0.4 else "#2ecc71"},
            "steps": [
                {"range": [0, 40], "color": "#1a3a1a"},
                {"range": [40, 70], "color": "#3a3a1a"},
                {"range": [70, 100], "color": "#5a0a0a"},
            ],
        },
    ))
    fig.update_layout(height=250, template="plotly_dark")
    return fig


# ── Downtime Calculator Charts ────────────────────────────────────────────────

def downtime_cost_chart(result: dict) -> go.Figure:
    labels = ["Production Loss", "Repair Cost", "Annual BD Cost", "Savings with PdM"]
    values = [
        result["production_loss_inr"],
        result["repair_cost_inr"],
        result["annual_bd_cost_inr"],
        result["savings_with_pdm_inr"],
    ]
    colors = ["#e74c3c", "#e67e22", "#c0392b", "#2ecc71"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"₹{v:,.0f}" for v in values],
                           textposition="outside"))
    fig.update_layout(title="Downtime Cost Analysis (INR)",
                      yaxis_title="INR", template="plotly_dark", height=400)
    return fig


def roi_waterfall(result: dict) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        name="ROI Analysis",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Annual BD Cost", "PdM System Cost", "Savings (50%)", "Net Benefit"],
        y=[result["annual_bd_cost_inr"], -500000,
           result["savings_with_pdm_inr"], 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(title="ROI Waterfall Chart", template="plotly_dark", height=400)
    return fig
