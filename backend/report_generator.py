"""
PDF Report Generator
Generates a professional maintenance report per asset.
Uses reportlab for PDF generation.
"""
import io
from datetime import datetime


def generate_asset_report(
    asset_tag: str,
    machine_type: str,
    health_result: dict,
    prediction: dict,
    ttf_result: dict,
    sensor_reading: dict,
    recommendations: list,
) -> bytes:
    """Generate a PDF report and return as bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return _fallback_text_report(asset_tag, machine_type, health_result,
                                     prediction, ttf_result, sensor_reading, recommendations)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=18, textColor=colors.HexColor("#1a1a2e"),
                                  spaceAfter=6)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
                               fontSize=13, textColor=colors.HexColor("#16213e"),
                               spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle("body", parent=styles["Normal"],
                                 fontSize=10, spaceAfter=4)

    risk = prediction.get("risk_level", "LOW")
    risk_colors = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22",
                   "MEDIUM": "#f1c40f", "LOW": "#2ecc71"}
    risk_color = colors.HexColor(risk_colors.get(risk, "#2ecc71"))

    story = []

    # Header
    story.append(Paragraph("PredMaint AI — Asset Health Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.3*cm))

    # Asset info
    story.append(Paragraph("Asset Information", h2_style))
    info_data = [
        ["Asset Tag", asset_tag, "Machine Type", machine_type],
        ["Report Date", datetime.now().strftime("%Y-%m-%d"), "Model Version", "v3.0"],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # Health summary
    story.append(Paragraph("Health Summary", h2_style))
    hs = health_result.get("health_score", 0)
    prob = prediction.get("probability", 0)
    ttf_days = ttf_result.get("estimated_days", 0)

    summary_data = [
        ["Metric", "Value", "Status"],
        ["Health Score", f"{hs:.0f} / 100",
         "Good" if hs >= 75 else "Warning" if hs >= 50 else "Critical"],
        ["Breakdown Probability", f"{prob:.1%}", risk],
        ["Estimated Days to Failure", f"{ttf_days:.0f} days",
         ttf_result.get("urgency", "Routine")],
        ["Anomaly Score", f"{prediction.get('anomaly_score', 0):.3f}",
         "Normal" if prediction.get("anomaly_score", 0) < 0.1 else "Anomalous"],
    ]
    summary_table = Table(summary_data, colWidths=[6*cm, 5*cm, 6*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4*cm))

    # Sensor readings
    story.append(Paragraph("Current Sensor Readings", h2_style))
    sensor_keys = ["temp_bearing_degC", "temp_motor_degC", "vibration_h_mms",
                   "vibration_v_mms", "oil_pressure_bar", "load_pct",
                   "shaft_rpm", "power_consumption_kw"]
    sensor_labels = ["Bearing Temp (C)", "Motor Temp (C)", "H-Vibration (mm/s)",
                     "V-Vibration (mm/s)", "Oil Pressure (bar)", "Load (%)",
                     "Shaft RPM", "Power (kW)"]
    sensor_data = [["Sensor", "Value"]]
    for key, label in zip(sensor_keys, sensor_labels):
        val = sensor_reading.get(key, "N/A")
        sensor_data.append([label, f"{val:.2f}" if isinstance(val, float) else str(val)])

    sensor_table = Table(sensor_data, colWidths=[9*cm, 8*cm])
    sensor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ]))
    story.append(sensor_table)
    story.append(Spacer(1, 0.4*cm))

    # Recommendations
    story.append(Paragraph("Maintenance Recommendations", h2_style))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
    story.append(Spacer(1, 0.4*cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "Generated by PredMaint AI v3.0 | github.com/SumedhPatil1507/industrial-predmaint-ai",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


def _fallback_text_report(asset_tag, machine_type, health_result,
                           prediction, ttf_result, sensor_reading, recommendations) -> bytes:
    """Plain text fallback if reportlab not installed."""
    lines = [
        "PredMaint AI - Asset Health Report",
        "=" * 50,
        f"Asset: {asset_tag} | Machine: {machine_type}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "HEALTH SUMMARY",
        f"Health Score: {health_result.get('health_score', 0):.0f}/100",
        f"Breakdown Probability: {prediction.get('probability', 0):.1%}",
        f"Risk Level: {prediction.get('risk_level', 'N/A')}",
        f"Days to Failure: {ttf_result.get('estimated_days', 0):.0f}",
        "",
        "RECOMMENDATIONS",
    ] + [f"- {r}" for r in recommendations]
    return "\n".join(lines).encode("utf-8")
