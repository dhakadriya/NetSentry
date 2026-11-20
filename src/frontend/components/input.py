"""Input controls and parameter configuration form primitives."""

import streamlit as st
from typing import Any, Dict


def render_hyperparameter_controls(algorithm: str) -> Dict[str, Any]:
    """Renders dynamic parameter tuning widgets for selected ML algorithm."""
    st.markdown("#### Hyperparameter Tuning")
    params: Dict[str, Any] = {}

    if algorithm == "random_forest":
        c1, c2 = st.columns(2)
        params["n_estimators"] = c1.slider("Number of Trees (n_estimators)", 10, 200, 100, 10)
        params["max_depth"] = c2.slider("Max Tree Depth", 4, 32, 16, 2)
        params["class_weight"] = "balanced"

    elif algorithm == "decision_tree":
        c1, c2 = st.columns(2)
        params["criterion"] = c1.selectbox("Split Criterion", ["gini", "entropy"], index=0)
        params["max_depth"] = c2.slider("Max Depth", 2, 25, 12, 1)
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 5, 1)

    elif algorithm == "logistic_regression":
        c1, c2 = st.columns(2)
        params["C"] = c1.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1)
        params["max_iter"] = c2.slider("Max Iterations", 200, 2000, 1000, 100)
        params["penalty"] = "l2"

    elif algorithm == "knn":
        c1, c2 = st.columns(2)
        params["n_neighbors"] = c1.slider("Number of Neighbors (K)", 1, 25, 5, 1)
        params["weights"] = c2.selectbox("Weight Function", ["distance", "uniform"], index=0)

    return params


def render_packet_input_form() -> Dict[str, Any]:
    """Renders interactive form controls to test single network flow records."""
    c1, c2, c3 = st.columns(3)
    proto = c1.selectbox("Protocol (proto)", ["tcp", "udp", "icmp"], index=0)
    service = c2.selectbox("Service (service)", ["http", "dns", "ssh", "ftp", "smtp", "-"], index=0)
    state = c3.selectbox("Connection State (state)", ["FIN", "CON", "INT", "REQ", "RST"], index=0)

    c4, c5, c6 = st.columns(3)
    dur = c4.number_input("Duration (seconds)", 0.0, 100.0, 0.12, 0.01)
    sbytes = c5.number_input("Source Bytes (sbytes)", 0, 10000000, 1250, 50)
    dbytes = c6.number_input("Destination Bytes (dbytes)", 0, 10000000, 3500, 50)

    c7, c8, c9 = st.columns(3)
    sttl = c7.slider("Source TTL (sttl)", 0, 255, 64)
    dttl = c8.slider("Dest TTL (dttl)", 0, 255, 62)
    spkts = c9.number_input("Source Packets (Spkts)", 1, 100000, 12, 1)

    return {
        "proto": proto,
        "service": service,
        "state": state,
        "dur": dur,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "sttl": sttl,
        "dttl": dttl,
        "Spkts": spkts,
        "Dpkts": max(1, int(spkts * 0.8)),
        "Sload": (sbytes * 8) / max(dur, 0.001),
        "Dload": (dbytes * 8) / max(dur, 0.001),
        "sloss": 0,
        "dloss": 0,
        "swin": 255,
        "dwin": 255,
        "stcpb": 100000,
        "dtcpb": 100000,
        "smeansz": int(sbytes / max(spkts, 1)),
        "dmeansz": int(dbytes / max(int(spkts * 0.8), 1)),
        "trans_depth": 0,
        "res_bdy_len": 0,
        "sjit": 1.2,
        "djit": 1.5,
        "sintpkt": 0.01,
        "dintpkt": 0.01,
        "tcprtt": 0.005,
        "synack": 0.002,
        "ackdat": 0.003,
        "is_sm_ips_ports": 0,
        "ct_state_ttl": 0,
        "ct_flw_http_mthd": 0,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_srv_src": 5,
        "ct_srv_dst": 5,
        "ct_dst_ltm": 5,
        "ct_src_ltm": 5,
        "ct_src_dport_ltm": 5,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 5,
    }
