"""Empty state visual representations for zero-data views."""

import streamlit as st
from typing import Optional


def render_empty_state(
    title: str = "No Data Found",
    description: str = "Upload a dataset or train a model to populate analytics.",
    icon: str = "📡",
    action_button_label: Optional[str] = None,
) -> bool:
    """Renders a zero-data illustration card with optional call-to-action button."""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 1.5rem; background: rgba(17, 24, 39, 0.4);
                    border: 2px dashed rgba(56, 189, 248, 0.2); border-radius: 16px; margin: 1.5rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.75rem;">{icon}</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 0.9rem; color: #64748b; max-width: 450px; margin: 0 auto 1.5rem auto;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if action_button_label:
        return st.button(action_button_label, type="primary", use_container_width=True)
    return False
