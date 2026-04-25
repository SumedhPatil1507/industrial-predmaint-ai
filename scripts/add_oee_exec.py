"""Appends OEE and Executive Summary pages to frontend/app.py"""
import ast

OEE_PAGE = r"""

# =============================================================================
# OEE CALCULATOR
# =============================================================================
elif page == "\U0001f4d0 OEE Calculator":
    import plotly.graph_objects as go
    import plotly.express as px
    from backend.oee_calculator import calculate_oee, oee_from_health_score, WORLD_CLASS_OEE
    from backend.health_score import compute_health_score
    from frontend.data_engine import generate_live_tick

    st.title("\U0001f4d0 OEE Calculator")
    st.markdown("Overall Equipment Effectiveness = Availability x Performance x Quality")
    st.markdown("**World-class benchmark: 85%**")
    st.divider()

    tab1, tab2 = st.tabs(["Manual Input", "Live Fleet OEE"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Machine Type", list(
                ["CNC Lathe","Hydraulic Press","Belt Conveyor","Screw Compressor","EOT Crane"]))
            asset_tag = st.text_input("Asset Tag", "CNC-001")
            planned_hours = st.number_input("Planned Production Hours", 1.0, 24.0, 8.0, 0.5)
            downtime_hours = st.number_input("Unplanned Downtime Hours", 0.0, 24.0, 0.5, 0.1)
        with col2:
            ideal_cycle = st.number_input("Ideal Cycle Time (min/part)", 0.1, 60.0, 1.0, 0.1)
            actual_cycle = st.number_input("Actual Cycle Time (min/part)", 0.1, 60.0, 1.1, 0.1)
            total_parts = st.number_input("Total Parts Produced", 1, 10000, 400)
            good_parts = st.number_input("Good Parts (no defects)", 1, 10000, 392)
            hourly_rate = st.number_input("Hourly Production Value (Rs.)", 1000, 500000, 25000, 1000)

        if st.button("Calculate OEE", type="primary"):
            try:
                result = calculate_oee(
                    machine_type=machine_type, asset_tag=asset_tag,
                    planned_hours=planned_hours, downtime_hours=downtime_hours,
                    ideal_cycle_time=ideal_cycle, actual_cycle_time=actual_cycle,
                    total_parts=int(total_parts), good_parts=int(good_parts),
                    hourly_production_inr=hourly_rate,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("OEE", f"{result.oee:.1f}%", delta=f"{result.oee - WORLD_CLASS_OEE:.1f}% vs world class")
                c2.metric("Availability", f"{result.availability:.1f}%")
                c3.metric("Performance", f"{result.performance:.1f}%")
                c4.metric("Quality", f"{result.quality:.1f}%")

                color = "#2ecc71" if result.oee >= 85 else "#f39c12" if result.oee >= 75 else "#e74c3c"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result.oee,
                    title={"text": f"OEE - {asset_tag}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color, "thickness": 0.3},
                        "steps": [
                            {"range": [0, 65], "color": "#3a0a0a"},
                            {"range": [65, 75], "color": "#3a3a0a"},
                            {"range": [75, 85], "color": "#1a3a1a"},
                            {"range": [85, 100], "color": "#0a4a0a"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": 85},
                    },
                ))
                fig.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                css = "ao" if result.oee >= 85 else "aw" if result.oee >= 75 else "ac"
                st.markdown(f'<div class="{css}"><b>{result.rating}</b> — {result.improvement_potential}</div>',
                            unsafe_allow_html=True)
                st.metric("Annual Loss from OEE Gap", f"Rs.{result.annual_loss_inr:,.0f}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.markdown("OEE estimated from live health scores for all assets.")
        if st.button("Calculate Fleet OEE", type="primary"):
            try:
                live_readings = generate_live_tick()
                oee_results = []
                for r in live_readings:
                    hs = compute_health_score(r)
                    oee_r = oee_from_health_score(
                        hs.health_score, r["machine_type"], r["asset_tag"])
                    oee_results.append(oee_r)

                df_oee = pd.DataFrame([{
                    "Asset": r.asset_tag, "Machine": r.machine_type,
                    "OEE %": r.oee, "Availability %": r.availability,
                    "Performance %": r.performance, "Quality %": r.quality,
                    "Rating": r.rating,
                    "Annual Loss (Rs.)": f"{r.annual_loss_inr:,.0f}",
                } for r in oee_results])

                avg_oee = sum(r.oee for r in oee_results) / len(oee_results)
                total_loss = sum(r.annual_loss_inr for r in oee_results)
                c1, c2, c3 = st.columns(3)
                c1.metric("Fleet Avg OEE", f"{avg_oee:.1f}%")
                c2.metric("Below World Class", sum(1 for r in oee_results if r.oee < 85))
                c3.metric("Total Annual Loss", f"Rs.{total_loss:,.0f}")

                st.dataframe(df_oee, use_container_width=True)

                fig = px.bar(df_oee, x="Asset", y="OEE %", color="OEE %",
                    color_continuous_scale="RdYlGn", range_color=[60, 100],
                    title="Fleet OEE Comparison", template="plotly_dark")
                fig.add_hline(y=85, line_dash="dash", line_color="white",
                              annotation_text="World Class (85%)")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                for r in oee_results:
                    fig2.add_trace(go.Bar(
                        name=r.asset_tag,
                        x=["Availability", "Performance", "Quality"],
                        y=[r.availability, r.performance, r.quality],
                    ))
                fig2.update_layout(barmode="group", title="OEE Components by Asset",
                                   template="plotly_dark", height=400)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
elif page == "\U0001f4cb Executive Summary":
    import plotly.graph_objects as go
    import plotly.express as px
    from backend.executive_summary import generate_summary
    from backend.health_score import compute_health_score, compute_fleet_health
    from backend.ttf_predictor import fleet_ttf
    from backend.oee_calculator import oee_from_health_score
    from frontend.data_engine import generate_live_tick

    st.title("\U0001f4cb Executive Summary")
    st.markdown("Decision-ready overview for plant managers. No ML jargon.")

    if st.button("Generate Summary", type="primary") or True:
        with st.spinner("Analysing fleet..."):
            try:
                live_readings = generate_live_tick()
                fleet_health = []
                health_scores = {}
                oee_results = []

                for r in live_readings:
                    hs = compute_health_score(r)
                    fleet_health.append({
                        "asset_tag": r["asset_tag"],
                        "machine_type": r["machine_type"],
                        "health_score": hs.health_score,
                        "status": hs.status,
                    })
                    health_scores[r["asset_tag"]] = hs.health_score
                    oee_r = oee_from_health_score(hs.health_score, r["machine_type"], r["asset_tag"])
                    oee_results.append(oee_r)

                ttf_results = fleet_ttf(st.session_state.df)
                summary = generate_summary(fleet_health, ttf_results, oee_results)

                # Header banner
                css = "ac" if summary.critical_assets else "aw" if summary.warning_assets else "ao"
                status_text = (f"CRITICAL: {len(summary.critical_assets)} machine(s) need immediate attention"
                               if summary.critical_assets
                               else f"WARNING: {len(summary.warning_assets)} machine(s) need attention"
                               if summary.warning_assets
                               else "All machines operating normally")
                st.markdown(f'<div class="{css}" style="font-size:1.1rem;padding:14px 18px;">'
                            f'{status_text}</div>', unsafe_allow_html=True)
                st.markdown(f"*Generated: {summary.generated_at}*")
                st.divider()

                # KPI row
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Assets", summary.total_assets)
                c2.metric("Critical", len(summary.critical_assets),
                          delta=None if not summary.critical_assets else "Action needed",
                          delta_color="inverse")
                c3.metric("Warning", len(summary.warning_assets))
                c4.metric("Healthy", len(summary.healthy_assets))
                c5.metric("Risk Exposure", f"Rs.{summary.total_risk_exposure_inr:,.0f}")

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Fleet Health Status")
                    df_fleet = pd.DataFrame(fleet_health)
                    for _, row in df_fleet.iterrows():
                        ttf = next((t for t in ttf_results
                                    if t.get("asset_tag") == row["asset_tag"]), {})
                        hs = row["health_score"]
                        color = "#e74c3c" if hs < 50 else "#f39c12" if hs < 75 else "#2ecc71"
                        st.markdown(
                            f"<div style='background:#161b22;border-left:4px solid {color};"
                            f"border-radius:6px;padding:10px 14px;margin-bottom:8px;'>"
                            f"<b>{row['asset_tag']}</b> — {row['machine_type']}<br>"
                            f"Health: <b>{hs:.0f}/100</b> | "
                            f"TTF: <b>{ttf.get('estimated_days', 90):.0f} days</b> | "
                            f"Status: <b>{row['status']}</b>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with col2:
                    st.subheader("OEE Summary")
                    df_oee = pd.DataFrame([{
                        "Asset": r.asset_tag, "OEE %": r.oee, "Rating": r.rating,
                        "Annual Loss": f"Rs.{r.annual_loss_inr:,.0f}",
                    } for r in oee_results])
                    st.dataframe(df_oee, use_container_width=True)

                    fig = px.bar(df_oee, x="Asset", y="OEE %", color="OEE %",
                        color_continuous_scale="RdYlGn", range_color=[60, 100],
                        template="plotly_dark", height=280)
                    fig.add_hline(y=85, line_dash="dash", line_color="white",
                                  annotation_text="World Class")
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Prioritised Action List")
                if summary.actions:
                    for action in summary.actions:
                        css_a = "ac" if action["priority"] == "CRITICAL" else "aw"
                        st.markdown(
                            f'<div class="{css_a}" style="margin-bottom:8px;">'
                            f'<b>[{action["priority"]}]</b> {action["asset"]} — '
                            f'{action["action"]} | Deadline: {action["deadline"]} | '
                            f'Est. Cost: Rs.{action["est_cost_inr"]:,}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown('<div class="ao">No immediate actions required.</div>',
                                unsafe_allow_html=True)

                st.divider()
                st.subheader("Shift Handover Recommendation")
                st.info(summary.shift_recommendation)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Risk Exposure", f"Rs.{summary.total_risk_exposure_inr:,.0f}")
                with col2:
                    st.metric("Potential Savings with PdM",
                              f"Rs.{summary.potential_savings_inr:,.0f}")

                # Download shift report
                report_lines = [
                    f"SHIFT HANDOVER REPORT — {summary.generated_at}",
                    "=" * 50,
                    f"Status: {status_text}",
                    "",
                    "FLEET HEALTH",
                ]
                for a in fleet_health:
                    ttf = next((t for t in ttf_results
                                if t.get("asset_tag") == a["asset_tag"]), {})
                    report_lines.append(
                        f"  {a['asset_tag']} ({a['machine_type']}): "
                        f"Health {a['health_score']:.0f}/100, "
                        f"TTF {ttf.get('estimated_days', 90):.0f} days"
                    )
                report_lines += ["", "ACTIONS REQUIRED"]
                for action in summary.actions:
                    report_lines.append(f"  [{action['priority']}] {action['action']}")
                report_lines += ["", f"RECOMMENDATION: {summary.shift_recommendation}"]

                st.download_button(
                    "Download Shift Report",
                    "\n".join(report_lines),
                    f"shift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain",
                )
            except Exception as e:
                st.error(f"Error: {e}")
"""

with open("frontend/app.py", "a", encoding="utf-8") as f:
    f.write(OEE_PAGE)

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
