"""Reusable card primitives for metrics, alerts, and analytics."""

import streamlit as st
from typing import Optional


def render_metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    color: str = "#38bdf8",
) -> None:
    """Renders an analytical metric card with glassmorphism aesthetics."""
    delta_html = ""
    if delta:
        delta_color = "#34d399" if delta.startswith("+") or "low" in delta.lower() else "#fb7185"
        delta_html = f"""<span style="font-size: 0.8rem; font-weight: 600; color: {delta_color}; margin-left: 0.5rem;">{delta}</span>"""

    sub_html = f"""<div style="font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">{subtitle}</div>""" if subtitle else ""

    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(8px);
                    border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px;
                    padding: 1.15rem; margin-bottom: 0.75rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">
                {title}
            </div>
            <div style="display: flex; align-items: baseline; margin-top: 0.35rem;">
                <span style="font-size: 1.85rem; font-weight: 800; color: {color}; letter-spacing: -0.025em;">
                    {value}
                </span>
                {delta_html}
            </div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_threat_card(
    attack_type: str,
    severity: str,
    count: int,
    percentage: float,
) -> None:
    """Renders a threat intelligence category summary card."""
    sev_colors = {
        "critical": ("#fb7185", "rgba(244, 63, 94, 0.2)"),
        "high": ("#fcd34d", "rgba(245, 158, 11, 0.2)"),
        "medium": ("#60a5fa", "rgba(59, 130, 246, 0.2)"),
        "low": ("#34d399", "rgba(16, 185, 129, 0.2)"),
    }
    text_color, bg_color = sev_colors.get(severity.lower(), ("#38bdf8", "rgba(56, 189, 248, 0.2)"))

    st.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(56, 189, 248, 0.15);
                    border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
                    display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 700; color: #f1f5f9; font-size: 0.95rem;">{attack_type}</div>
                <div style="font-size: 0.78rem; color: #64748b;">{count} occurrences ({percentage:.1f}%)</div>
            </div>
            <span style="background: {bg_color}; color: {text_color}; border: 1px solid {text_color}40;
                         padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                {severity}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
