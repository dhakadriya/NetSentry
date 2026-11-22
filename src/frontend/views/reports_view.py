"""Detection reports view providing downloadable CSV analysis files and executive summaries."""

from pathlib import Path

import streamlit as st

from src.database.crud.prediction_repo import list_prediction_jobs


def render_reports_view(db_session) -> None:
    """Renders generated batch intrusion detection reports and data export view."""
    st.markdown("### 📑 Detection Reports & Data Export Center")
    st.markdown("Download generated network classification CSV reports and review historical inference jobs.")

    jobs = list_prediction_jobs(db_session)

    if jobs:
        st.markdown(f"**Historical Prediction Scans ({len(jobs)} Completed)**")
        for j in jobs:
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])
                c1.markdown(f"**Job #{j.id}**")
                c2.markdown(f"Records: **{j.total_records:,}**")
                c3.markdown(f"Threats: **{j.malicious_count:,}** ({j.detection_rate * 100:.1f}%)")
                c4.markdown(f"Latency: **{j.execution_time_seconds:.3f}s**")

                if j.report_csv_path and Path(j.report_csv_path).exists():
                    report_bytes = Path(j.report_csv_path).read_bytes()
                    c5.download_button(
                        label="📥 Download CSV",
                        data=report_bytes,
                        file_name=Path(j.report_csv_path).name,
                        mime="text/csv",
                        key=f"dl_job_{j.id}",
                    )
                else:
                    c5.caption("Generated internally")
            st.markdown("---")
    else:
        st.info("No batch prediction scans have been executed yet. Run a scan from the Live Detection tab to generate downloadable reports.")
