"""NetSentry - Intelligent Machine Learning-Based Network Intrusion Detection and Analysis System
Main Streamlit Application Controller and View Router
"""

import os
import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from src.database.connection import get_session_factory
from src.database.init_db import init_db
from src.database.crud.experiment_repo import get_best_model
from src.frontend.components.navbar import render_navbar

from src.frontend.views.overview_view import render_overview_view
from src.frontend.views.dataset_view import render_dataset_view
from src.frontend.views.training_view import render_training_view
from src.frontend.views.comparison_view import render_comparison_view
from src.frontend.views.detection_view import render_detection_view
from src.frontend.views.alerts_view import render_alerts_view
from src.frontend.views.reports_view import render_reports_view


def load_custom_css() -> None:
    """Injects custom cyber-dark glassmorphism styling."""
    css_file = Path(__file__).resolve().parent / "styles" / "main.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            custom_css = f.read()
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


def init_app_state() -> None:
    """Initializes database schema and session state variables."""
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True


def main() -> None:
    """Main application entrypoint."""
    st.set_page_config(
        page_title="NetSentry | Intelligent NIDS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()
    init_app_state()

    SessionFactory = get_session_factory()
    db = SessionFactory()

    try:
        best_model = get_best_model(db)
        active_model_name = best_model.model_name if best_model else "RandomForest-Optimized"

        # Render Top Navigation Header
        render_navbar(
            app_name="NetSentry",
            active_model=active_model_name,
            api_online=True,
        )

        # Sidebar Navigation
        with st.sidebar:
            st.markdown(
                """
                <div style="text-align: center; padding: 0.5rem 0 1.25rem 0;">
                    <span style="font-size: 2.5rem;">🛡️</span>
                    <h2 style="margin: 0.25rem 0 0 0; color: #38bdf8; font-size: 1.5rem; font-weight: 800;">NetSentry</h2>
                    <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Autonomous ML-NIDS Platform</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            menu_options = [
                "📊 Overview",
                "📁 Datasets & EDA",
                "⚙️ Model Training",
                "⚖️ Benchmark & Comparison",
                "🔍 Live Traffic Detection",
                "🚨 Security Alerts",
                "📑 Reports & Export",
            ]

            selected_page = st.radio(
                "Navigation Console",
                menu_options,
                index=0,
                label_visibility="collapsed",
            )

            st.markdown("---")
            st.markdown("### 📡 System Telemetry")
            st.markdown(
                """
                - **Engine:** Scikit-Learn 1.6+
                - **Protocol:** UNSW-NB15 / CIC-IDS
                - **Database:** SQLite3 / SQLAlchemy 2.0
                - **Pipeline:** Automated Cleaning + Feature Selection
                - **API Gateway:** FastAPI / ASGI
                """
            )
            st.caption("NetSentry Security Suite v1.0.0")

        # View Routing
        if selected_page == "📊 Overview":
            render_overview_view(db)
        elif selected_page == "📁 Datasets & EDA":
            render_dataset_view(db)
        elif selected_page == "⚙️ Model Training":
            render_training_view(db)
        elif selected_page == "⚖️ Benchmark & Comparison":
            render_comparison_view(db)
        elif selected_page == "🔍 Live Traffic Detection":
            render_detection_view(db)
        elif selected_page == "🚨 Security Alerts":
            render_alerts_view(db)
        elif selected_page == "📑 Reports & Export":
            render_reports_view(db)

    finally:
        db.close()


if __name__ == "__main__":
    main()
