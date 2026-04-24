"""Appends new page handlers to frontend/app.py"""
import ast

PAGES_CODE = r"""

# =============================================================================
# MAINTENANCE SCHEDULE
# =============================================================================
elif page == "\U0001f4c5 Maintenance Schedule":
    import plotly.express as px
    from backend.ttf_predictor import fleet_ttf
    from backend.maintenance_scheduler import generate_schedule, schedule_to_dataframe
    from backend.health_score import compute_health_score
    from frontend.data_engine import generate_live_tick

    st.title("\U0001f4c5 Maintenance Schedule Generator")
    st.markdown("Optimal maintenance schedule based on TTF predictions and health scores.")

    horizon = st.slider("Planning horizon (days)", 7, 90, 30)

    if st.button("Generate Schedule", type="primary"):
        with st.spinner("Analysing fleet and generating schedule..."):
            try:
                ttf_results = fleet_ttf(st.session_state.df)
                live_readings = generate_live_tick()
                health_scores = {}
                for r in live_readings:
                    hs = compute_health_score(r)
                    health_scores[r["asset_tag"]] = hs.health_score

                work_orders = generate_schedule(ttf_results, health_scores, horizon_days=horizon)
                df_wo = schedule_to_dataframe(work_orders)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Work Orders", len(work_orders))
                c2.metric("Emergency", sum(1 for w in work_orders if w.urgency == "Immediate"))
                c3.metric("Total Est. Hours", f"{sum(w.estimated_hours for w in work_orders):.0f}h")
                c4.metric("Total Est. Cost", f"Rs.{sum(w.estimated_cost_inr for w in work_orders):,.0f}")

                st.subheader("Work Order Schedule")
                st.dataframe(df_wo, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    cost_df = pd.DataFrame([{
                        "Asset": w.asset_tag,
                        "Cost": w.estimated_cost_inr,
                        "Type": w.task_type,
                    } for w in work_orders])
                    fig = px.bar(cost_df, x="Asset", y="Cost", color="Type",
                                 title="Estimated Cost by Asset", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    urgency_counts = pd.DataFrame([{
                        "Urgency": w.urgency, "Count": 1
                    } for w in work_orders]).groupby("Urgency").sum().reset_index()
                    fig2 = px.pie(urgency_counts, names="Urgency", values="Count",
                                  color="Urgency",
                                  color_discrete_map={
                                      "Immediate": "#e74c3c", "This Week": "#e67e22",
                                      "This Month": "#f1c40f", "Routine": "#2ecc71"},
                                  title="Work Orders by Urgency", template="plotly_dark",
                                  hole=0.4)
                    st.plotly_chart(fig2, use_container_width=True)

                st.download_button("Download Schedule CSV",
                    df_wo.to_csv(index=False), "maintenance_schedule.csv", "text/csv")

                st.subheader("Work Order Details")
                for wo in work_orders:
                    label = f"{wo.wo_id} | {wo.asset_tag} | {wo.urgency} | {wo.scheduled_date}"
                    with st.expander(label):
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"**Task Type:** {wo.task_type}")
                        c2.markdown(f"**Est. Hours:** {wo.estimated_hours}h")
                        c3.markdown(f"**Est. Cost:** Rs.{wo.estimated_cost_inr:,.0f}")
                        st.markdown("**Tasks:**")
                        for t in wo.tasks:
                            st.markdown(f"  - {t}")
                        st.markdown("**Spare Parts:** " + ", ".join(wo.spare_parts))
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# DRIFT DETECTION
# =============================================================================
elif page == "\U0001f4c9 Drift Detection":
    import plotly.express as px
    from backend.drift_detector import detect_drift, save_baseline, load_baseline

    st.title("\U0001f4c9 Model Drift Detection")
    st.markdown("Compares current sensor distributions against training baseline using PSI and KS test.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Current Data as Baseline", type="secondary"):
            try:
                save_baseline(st.session_state.df)
                st.success("Baseline saved.")
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        baseline = load_baseline()
        st.markdown(f"**Baseline:** {'OK' if baseline else 'Not set'}")

    st.info("PSI < 0.1 = Stable | 0.1-0.2 = Warning | > 0.2 = Critical drift")

    use_custom = st.checkbox("Use custom dataset for drift check")
    df_drift = st.session_state.df
    if use_custom:
        up = st.file_uploader("Upload current data", type=["csv", "xlsx", "json", "parquet"])
        if up:
            df_drift = parse_upload(up.name, up.read())

    if st.button("Run Drift Analysis", type="primary"):
        with st.spinner("Analysing distributions..."):
            try:
                report = detect_drift(df_drift)
                c1, c2, c3 = st.columns(3)
                c1.metric("Drift Score", f"{report.drift_score:.4f}")
                c2.metric("Features Drifted", len(report.features_drifted))
                c3.metric("Status", "Drift Detected" if report.overall_drift else "Stable")

                css = "ac" if len(report.features_drifted) >= 4 else "aw" if report.overall_drift else "ao"
                st.markdown(f'<div class="{css}">{report.recommendation}</div>',
                            unsafe_allow_html=True)

                if report.results:
                    df_r = pd.DataFrame([{
                        "Feature": r.feature, "PSI": r.psi,
                        "KS Stat": r.ks_stat, "KS p-value": r.ks_pvalue,
                        "Drift": "Yes" if r.drift_detected else "No",
                        "Severity": r.severity,
                        "Baseline Mean": r.baseline_mean,
                        "Current Mean": r.current_mean,
                        "Mean Shift %": r.mean_shift_pct,
                    } for r in report.results])

                    st.dataframe(df_r, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(df_r, x="Feature", y="PSI", color="Severity",
                            color_discrete_map={"None": "#2ecc71",
                                                "Warning": "#f1c40f",
                                                "Critical": "#e74c3c"},
                            title="PSI by Feature", template="plotly_dark")
                        fig.add_hline(y=0.1, line_dash="dash", line_color="yellow",
                                      annotation_text="Warning")
                        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                                      annotation_text="Critical")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig2 = px.bar(df_r, x="Feature", y="Mean Shift %",
                            color="Mean Shift %", color_continuous_scale="RdYlGn_r",
                            title="Mean Shift % vs Baseline", template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# PDF REPORT
# =============================================================================
elif page == "\U0001f4c4 PDF Report":
    from backend.report_generator import generate_asset_report
    from backend.health_score import compute_health_score
    from backend.ttf_predictor import estimate_ttf
    from frontend.data_engine import generate_live_tick

    st.title("\U0001f4c4 Asset Health Report")
    st.markdown("Generate a professional PDF report for any asset.")

    asset_tag = st.selectbox("Select Asset",
        ["CNC-001", "HYD-001", "BLT-001", "CMP-001", "EOT-001"])

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                live_readings = generate_live_tick()
                snap = next((r for r in live_readings if r["asset_tag"] == asset_tag), {})
                if not snap:
                    st.error("No live data available.")
                else:
                    hs = compute_health_score(snap)
                    pred = predict_single(snap) if model_ok else {
                        "prediction": 0, "probability": 0.1,
                        "risk_level": "LOW", "anomaly_score": 0.0}
                    try:
                        ttf_r = estimate_ttf(st.session_state.df, asset_tag).__dict__
                    except Exception:
                        ttf_r = {"estimated_days": 30, "urgency": "Routine"}

                    pdf_bytes = generate_asset_report(
                        asset_tag=asset_tag,
                        machine_type=snap.get("machine_type", "Unknown"),
                        health_result=hs.__dict__,
                        prediction=pred,
                        ttf_result=ttf_r,
                        sensor_reading=snap,
                        recommendations=hs.recommendations,
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Health Score", f"{hs.health_score:.0f}/100")
                    c2.metric("Breakdown Prob", f"{pred['probability']:.1%}")
                    c3.metric("Days to Failure", f"{ttf_r.get('estimated_days', 0):.0f}")
                    st.plotly_chart(risk_gauge(pred["probability"]), use_container_width=True)

                    ext = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
                    fname = f"predmaint_{asset_tag}_{datetime.now().strftime('%Y%m%d')}.{ext}"
                    st.download_button(
                        label=f"Download {asset_tag} Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf" if ext == "pdf" else "text/plain",
                    )
                    st.success("Report ready for download!")
            except Exception as e:
                st.error(f"Error: {e}")
"""

with open("frontend/app.py", "a", encoding="utf-8") as f:
    f.write(PAGES_CODE)

print("Written. Checking syntax...")
src = open("frontend/app.py", encoding="utf-8").read()
try:
    ast.parse(src)
    print(f"Syntax OK — {len(src.splitlines())} lines")
except SyntaxError as e:
    print(f"ERR line {e.lineno}: {e.msg}")
    lines = src.splitlines()
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
        print(f"  {i+1}: {lines[i]}")
