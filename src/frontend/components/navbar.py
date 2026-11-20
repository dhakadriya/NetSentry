"""Navbar header component with status badges and system diagnostics."""

import streamlit as st


def render_navbar(
    app_name: str = "NetSentry",
    active_model: str = "RandomForest-Optimized",
    api_online: bool = True,
) -> None:
    """Renders the top branding navigation bar with live status indicators."""
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; 
                    padding: 1rem 1.5rem; background: rgba(17, 24, 39, 0.85); 
                    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.85rem;">🛡️</span>
                <div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #38bdf8; letter-spacing: -0.025em;">
                        {app_name} <span style="font-size: 0.8rem; font-weight: 500; color: #94a3b8;">NIDS v1.0</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b;">
                        Intelligent Machine Learning Network Intrusion Detection & Analysis
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 0.75rem; align-items: center;">
                <span style="background: {'rgba(16, 185, 129, 0.15)' if api_online else 'rgba(244, 63, 94, 0.15)'}; 
                             color: {'#34d399' if api_online else '#fb7185'}; 
                             border: 1px solid {'rgba(16, 185, 129, 0.3)' if api_online else 'rgba(244, 63, 94, 0.3)'}; 
                             padding: 0.3rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    {'● API Online' if api_online else '○ API Offline'}
                </span>
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; 
                             border: 1px solid rgba(56, 189, 248, 0.3); 
                             padding: 0.3rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    Active Model: {active_model}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
