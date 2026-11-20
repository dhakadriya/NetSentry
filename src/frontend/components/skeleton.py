"""Loading skeletons and progressive animation placeholders."""

import streamlit as st


def render_loading_skeleton(message: str = "Analyzing network traffic...", height_px: int = 120) -> None:
    """Renders a pulsing animated skeleton placeholder."""
    st.markdown(
        f"""
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.35; }}
            50% {{ opacity: 0.8; }}
        }}
        .skeleton-box {{
            width: 100%;
            height: {height_px}px;
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.6) 0%, rgba(56, 189, 248, 0.15) 50%, rgba(30, 41, 59, 0.6) 100%);
            border-radius: 10px;
            animation: pulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            border: 1px dashed rgba(56, 189, 248, 0.3);
        }}
        </style>
        <div class="skeleton-box">
            <span>⚡ {message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
