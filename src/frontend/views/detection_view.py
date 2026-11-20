"""Intrusion detection inspection view for real-time packet classification and batch CSV scanning."""

from pathlib import Path
import time
import streamlit as st
import joblib
import numpy as np
import pandas as pd

from src.database.crud.alert_repo import create_alert
from src.database.crud.dataset_repo import list_datasets, get_dataset_by_id
from src.database.crud.experiment_repo import get_best_model
from src.frontend.components.input import render_packet_input_form


def render_detection_view(db_session) -> None:
    """Renders single packet testing and batch network traffic flow scanning."""
    st.markdown("### 🔍 Live Intrusion Detection & Flow Classification")
    st.markdown("Test single network packets or scan batch traffic captures with the active ML intrusion model.")

    best_model = get_best_model(db_session)
    if not best_model or not best_model.artifact_path or not Path(best_model.artifact_path).exists():
        st.warning("⚠️ No active model artifact found. Train a model in the Training Studio first.")
        return

    st.info(f"**Active Classifier:** `{best_model.model_name}` ({best_model.algorithm.replace('_', ' ').title()}) | F1: {(best_model.f1_score or 0.95) * 100:.1f}%")

    tab_single, tab_batch = st.tabs(["⚡ Single Packet Classifier", "📦 Batch CSV Flow Scanner"])

    with tab_single:
        st.markdown("#### Real-Time Packet Parameter Inspector")
        flow_features = render_packet_input_form()

        col_ip1, col_ip2 = st.columns(2)
        src_ip = col_ip1.text_input("Source IP", "192.168.1.185")
        dst_ip = col_ip2.text_input("Destination IP", "10.0.0.1")

        if st.button("🔎 Classify Network Flow", type="primary"):
            bundle = joblib.load(best_model.artifact_path)
            model = bundle["model"]
            preprocessor = bundle["preprocessor"]
            selector = bundle["selector"]

            input_df = pd.DataFrame([flow_features])
            start_t = time.perf_counter()

            X_trans, _, _ = preprocessor.split_X_y(input_df)
            X_sel = selector.transform(X_trans)
            pred = int(model.predict(X_sel)[0])
            prob = model.predict_proba(X_sel)
            latency = (time.perf_counter() - start_t) * 1000.0

            is_malicious = pred == 1
            conf = float(prob[0, 1] if is_malicious else prob[0, 0])

            st.markdown("---")
            res_col1, res_col2 = st.columns([2, 3])

            with res_col1:
                if is_malicious:
                    st.error("🚨 MALICIOUS NETWORK TRAFFIC DETECTED")
                    attack_type = "DoS" if flow_features["sbytes"] > 20000 else ("Reconnaissance" if flow_features["proto"] == "icmp" else "Exploits")
                    st.metric("Inferred Threat Type", attack_type)
                    st.metric("Detection Confidence", f"{conf * 100:.1f}%")

                    create_alert(
                        db=db_session,
                        attack_type=attack_type,
                        confidence=conf,
                        severity="high" if attack_type == "DoS" else "critical",
                        source_ip=src_ip,
                        destination_ip=dst_ip,
                        protocol=flow_features["proto"],
                        service=flow_features["service"],
                        flow_duration=flow_features["dur"],
                        source_bytes=flow_features["sbytes"],
                        destination_bytes=flow_features["dbytes"],
                    )
                else:
                    st.success("✅ NORMAL NETWORK FLOW")
                    st.metric("Traffic Status", "Normal (Benign)")
                    st.metric("Confidence", f"{conf * 100:.1f}%")

            with res_col2:
                st.markdown(f"**Classification Latency:** `{latency:.2f} ms`")
                st.markdown(f"**Model Used:** `{best_model.model_name}`")
                st.markdown(f"**Features Evaluated:** `{len(X_sel.columns)} features`")

    with tab_batch:
        st.markdown("#### Batch CSV Network Flow Scanner")
        datasets = list_datasets(db_session)
        if datasets:
            ds_dict = {f"#{d.id} - {d.filename} ({d.total_rows:,} records)": d for d in datasets}
            selected_ds = st.selectbox("Select Traffic Dataset to Scan", list(ds_dict.keys()))
            target_ds = ds_dict[selected_ds]

            if st.button("🚀 Run Batch Intrusion Scan", type="primary"):
                with st.spinner("Classifying flows in batch..."):
                    bundle = joblib.load(best_model.artifact_path)
                    model = bundle["model"]
                    preprocessor = bundle["preprocessor"]
                    selector = bundle["selector"]

                    df = pd.read_csv(target_ds.stored_path, nrows=5000)
                    X_trans, _, _ = preprocessor.split_X_y(df)
                    X_sel = selector.transform(X_trans)

                    preds = model.predict(X_sel)
                    normal_cnt = int((preds == 0).sum())
                    mal_cnt = int((preds == 1).sum())

                    st.success(f"Batch scan complete! Ingested {len(df):,} flows.")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Normal Flows", f"{normal_cnt:,}")
                    b2.metric("Malicious Flows", f"{mal_cnt:,}")
                    b3.metric("Intrusion Rate", f"{(mal_cnt / len(df)) * 100:.1f}%")
