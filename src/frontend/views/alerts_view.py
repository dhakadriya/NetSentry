"""Threat alerts center view with alert filtering, packet inspection, and analyst triage."""

import streamlit as st

from src.database.crud.alert_repo import get_threat_stats, list_alerts, update_alert_review
from src.frontend.components.modal import render_alert_inspection_dialog


def render_alerts_view(db_session) -> None:
    """Renders the cybersecurity threat triage and alert management dashboard."""
    st.markdown("### 🚨 Threat Alerts & Incident Response Center")
    st.markdown("Filter, inspect suspicious packet headers, and triage active intrusion alerts.")

    stats = get_threat_stats(db_session)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical Threats", stats["critical_alerts"])
    c2.metric("High Severity", stats["high_alerts"])
    c3.metric("Medium Severity", stats["medium_alerts"])
    c4.metric("Triaged / Reviewed", stats["reviewed_count"])

    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns(3)
    sev_filter = f1.selectbox("Filter Severity", ["All", "critical", "high", "medium", "low"], index=0)
    rev_filter = f2.selectbox("Triage Status", ["All", "Pending Review", "Reviewed"], index=0)

    severity_arg = None if sev_filter == "All" else sev_filter
    is_rev_arg = None
    if rev_filter == "Pending Review":
        is_rev_arg = False
    elif rev_filter == "Reviewed":
        is_rev_arg = True

    alerts = list_alerts(db_session, severity=severity_arg, is_reviewed=is_rev_arg, limit=50)

    if alerts:
        st.markdown(f"**Showing {len(alerts)} Alerts**")
        for al in alerts:
            with st.container():
                cols = st.columns([1, 2, 2, 2, 1, 2])
                cols[0].markdown(f"**#{al.id}**")
                cols[1].markdown(f"**{al.attack_type}**")
                cols[2].markdown(f"`{al.source_ip}` → `{al.destination_ip}`")
                cols[3].markdown(f"Conf: **{al.confidence * 100:.1f}%**")

                sev_badge = f"<span style='color: {'#fb7185' if al.severity == 'critical' else '#fcd34d'}; font-weight:700;'>{al.severity.upper()}</span>"
                cols[4].markdown(sev_badge, unsafe_allow_html=True)

                if cols[5].button(f"Triage #{al.id}", key=f"triage_{al.id}"):
                    st.session_state["inspect_alert_id"] = al.id

        if "inspect_alert_id" in st.session_state:
            inspect_id = st.session_state["inspect_alert_id"]
            target = next((a for a in alerts if a.id == inspect_id), None)
            if target:
                st.markdown("---")
                render_alert_inspection_dialog(target.to_dict())

                notes = st.text_input("Analyst Triage Annotations", value=target.analyst_notes or "Investigated and confirmed intrusion signature.")
                if st.button("Mark Triaged / Reviewed", type="primary"):
                    update_alert_review(db_session, target.id, is_reviewed=True, notes=notes)
                    st.success(f"Alert #{target.id} marked as triaged!")
                    del st.session_state["inspect_alert_id"]
                    st.rerun()
    else:
        st.info("No threat alerts matching selected filter criteria.")
