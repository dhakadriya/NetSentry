"""Action button primitives with styled variants."""

import streamlit as st
from typing import Optional


def render_action_button(
    label: str,
    key: str,
    button_type: str = "primary",
    icon: Optional[str] = None,
    help_text: Optional[str] = None,
    use_container_width: bool = True,
) -> bool:
    """Renders a styled action button supporting primary, secondary, and danger styles."""
    display_label = f"{icon} {label}" if icon else label
    btn_type = "primary" if button_type in ("primary", "danger") else "secondary"

    return st.button(
        label=display_label,
        key=key,
        type=btn_type,
        help=help_text,
        use_container_width=use_container_width,
    )
