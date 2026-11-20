"""Dataset management and Exploratory Data Analysis (EDA) dashboard view."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from src.core.security import compute_sha256, sanitize_filename
from src.database.crud.dataset_repo import create_dataset, list_datasets, get_dataset_by_id
from src.ml.validators.dataset_validator import DatasetValidator


def render_dataset_view(db_session) -> None:
    """Renders the dataset upload, catalog, and EDA view."""
    st.markdown("### 📁 Network Traffic Datasets & Exploratory Analysis")
    st.markdown("Upload, validate, inspect schemas, and analyze network flow distributions.")

    tab_catalog, tab_upload, tab_eda = st.tabs(["📚 Dataset Catalog", "⬆️ Upload & Validate CSV", "🔬 Exploratory Analysis (EDA)"])

    datasets = list_datasets(db_session)

    with tab_catalog:
        if datasets:
            st.markdown(f"**{len(datasets)} Datasets Cataloged**")
            ds_data = []
            for d in datasets:
                ds_data.append({
                    "ID": d.id,
                    "Filename": d.filename,
                    "Type": d.dataset_type.upper(),
                    "Total Records": f"{d.total_rows:,}",
                    "Normal Flows": f"{d.normal_count:,}",
                    "Attack Flows": f"{d.attack_count:,}",
                    "Attack Ratio": f"{(d.attack_count / max(d.total_rows, 1)) * 100:.1f}%",
                    "Uploaded": d.created_at.strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(ds_data), use_container_width=True, hide_index=True)
        else:
            st.info("No datasets cataloged. Upload a dataset or load the bundled sample.")

    with tab_upload:
        st.markdown("#### Upload Network Flow Dataset (CSV)")
        uploaded_file = st.file_uploader("Select UNSW-NB15 or CIC-IDS2017 compliant CSV", type=["csv"])
        description = st.text_input("Dataset Description (optional)", "UNSW-NB15 sample evaluation traffic")

        if uploaded_file and st.button("Validate & Register Dataset", type="primary"):
            with st.spinner("Validating dataset structure and security integrity..."):
                content = uploaded_file.read()
                safe_name = sanitize_filename(uploaded_file.name)
                sha = compute_sha256(content)

                save_path = Path("data/uploads") / f"{sha[:10]}_{safe_name}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(content)

                validator = DatasetValidator()
                report, df = validator.validate_file(save_path)

                if report.is_valid:
                    normal_cnt = int((df["label"] == 0).sum()) if "label" in df.columns else 0
                    attack_cnt = int((df["label"] == 1).sum()) if "label" in df.columns else 0

                    record = create_dataset(
                        db=db_session,
                        filename=safe_name,
                        stored_path=str(save_path),
                        dataset_type=report.detected_type,
                        total_rows=report.total_rows,
                        feature_count=report.total_columns,
                        normal_count=normal_cnt,
                        attack_count=attack_cnt,
                        checksum_sha256=sha,
                        columns=list(df.columns),
                        description=description,
                    )
                    st.success(f"✅ Successfully validated and registered '{safe_name}'! ID: #{record.id} ({record.total_rows:,} records)")
                else:
                    st.error(f"❌ Validation failed: {'; '.join(report.errors)}")

    with tab_eda:
        st.markdown("#### Dataset Profiling & Distribution Inspector")
        if not datasets:
            st.warning("Please upload or seed a dataset first.")
            return

        ds_names = {f"#{d.id} - {d.filename} ({d.total_rows:,} rows)": d.id for d in datasets}
        selected_label = st.selectbox("Select Dataset for Profiling", list(ds_names.keys()))
        selected_id = ds_names[selected_label]
        target_ds = get_dataset_by_id(db_session, selected_id)

        if target_ds and Path(target_ds.stored_path).exists():
            df = pd.read_csv(target_ds.stored_path, nrows=50000)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Feature Columns", len(df.columns))
            normal_cnt = int((df["label"] == 0).sum()) if "label" in df.columns else 0
            attack_cnt = int((df["label"] == 1).sum()) if "label" in df.columns else 0
            c3.metric("Attack Ratio", f"{(attack_cnt / max(len(df), 1)) * 100:.1f}%")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if "proto" in df.columns:
                    top_proto = df["proto"].value_counts().head(6).reset_index()
                    top_proto.columns = ["Protocol", "Count"]
                    fig_proto = px.bar(top_proto, x="Protocol", y="Count", title="Top Protocols", color="Protocol",
                                       color_discrete_sequence=px.colors.qualitative.Prism)
                    fig_proto.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), height=320)
                    st.plotly_chart(fig_proto, use_container_width=True)

            with col_p2:
                if "service" in df.columns:
                    top_svc = df["service"].value_counts().head(6).reset_index()
                    top_svc.columns = ["Service", "Count"]
                    fig_svc = px.bar(top_svc, x="Service", y="Count", title="Top Services", color="Service",
                                     color_discrete_sequence=px.colors.qualitative.Safe)
                    fig_svc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), height=320)
                    st.plotly_chart(fig_svc, use_container_width=True)

            st.markdown("##### Numerical Feature Summary Statistics")
            num_cols = ["dur", "sbytes", "dbytes", "sttl", "dttl", "Spkts", "Dpkts"]
            active_nums = [c for c in num_cols if c in df.columns]
            if active_nums:
                st.dataframe(df[active_nums].describe().round(3).T, use_container_width=True)
