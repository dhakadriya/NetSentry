"""Plotly visualization charts for ROC curves, confusion matrices, and model comparisons."""

from typing import Any, Dict, List
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_confusion_matrix_chart(cm: List[List[int]], labels: List[str] = ["Normal", "Malicious"]) -> None:
    """Renders an interactive heatmap for the binary confusion matrix."""
    arr = np.array(cm)
    # Calculate percentages
    row_sums = arr.sum(axis=1, keepdims=True)
    percentages = np.divide(arr, row_sums, where=row_sums != 0) * 100

    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{arr[i, j]:,}</b><br>({percentages[i, j]:.1f}%)",
                    font=dict(color="#f8fafc", size=14),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=labels,
            y=labels,
            colorscale=[[0, "#0f172a"], [0.5, "#0284c7"], [1, "#38bdf8"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix (Normalized Recall Diagnostics)",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_algorithm_comparison_chart(comparison_data: Dict[str, List[Any]]) -> None:
    """Renders side-by-side metric comparison bar chart across the 4 ML models."""
    algorithms = comparison_data.get("algorithms", [])
    f1_scores = [round(x * 100, 2) for x in comparison_data.get("f1_score", [])]
    accuracies = [round(x * 100, 2) for x in comparison_data.get("accuracy", [])]
    fprs = [round(x * 100, 2) for x in comparison_data.get("fpr", [])]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=algorithms,
        y=f1_scores,
        name="F1-Score (%)",
        marker_color="#38bdf8",
    ))
    fig.add_trace(go.Bar(
        x=algorithms,
        y=accuracies,
        name="Accuracy (%)",
        marker_color="#10b981",
    ))
    fig.add_trace(go.Bar(
        x=algorithms,
        y=fprs,
        name="False Positive Rate (%)",
        marker_color="#f43f5e",
    ))

    fig.update_layout(
        title="Model Benchmark: F1 vs Accuracy vs False Positive Rate (FPR)",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Percentage (%)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_threat_distribution_pie(attack_counts: Dict[str, int]) -> None:
    """Renders a doughnut chart displaying the breakdown of detected attack categories."""
    labels = list(attack_counts.keys())
    values = list(attack_counts.values())

    colors = ["#38bdf8", "#f43f5e", "#f59e0b", "#a855f7", "#ec4899", "#10b981", "#64748b"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        insidetextorientation="radial",
    )])

    fig.update_layout(
        title="Detected Threat Distribution by Attack Category",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
