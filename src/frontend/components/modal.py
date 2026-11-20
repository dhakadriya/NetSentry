"""Modal and detail dialog components for threat inspection."""

import streamlit as st
from typing import Any, Dict


def render_alert_inspection_dialog(alert: Dict[str, Any]) -> None:
    """Renders a detailed inspection card for a flagged intrusion packet."""
    with st.expander(f"🔍 Inspect Threat Event #{alert.get('id', 'N/A')} - {alert.get('attack_type', 'Attack')}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attack Type", alert.get("attack_type", "Unknown"))
        c2.metric("Confidence", f"{alert.get('confidence', 0.0) * 100:.1f}%")
        c3.metric("Severity", str(alert.get("severity", "medium")).upper())
        c4.metric("Protocol", alert.get("protocol", "TCP").upper())

        st.markdown("---")
        col_net1, col_net2 = st.columns(2)
        with col_net1:
            st.markdown(f"**Source IP:** `{alert.get('source_ip', 'unknown')}`")
            st.markdown(f"**Source Bytes:** `{alert.get('source_bytes', 0):,}`")
            st.markdown(f"**Flow Duration:** `{alert.get('flow_duration', 0.0):.4f}s`")
        with col_net2:
            st.markdown(f"**Destination IP:** `{alert.get('destination_ip', 'unknown')}`")
            st.markdown(f"**Destination Bytes:** `{alert.get('destination_bytes', 0):,}`")
            st.markdown(f"**Service:** `{alert.get('service', '-')}`")

        if alert.get("analyst_notes"):
            st.info(f"**Analyst Notes:** {alert.get('analyst_notes')}")
