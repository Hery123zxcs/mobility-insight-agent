"""Plotly chart builders for the agent workflow dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import FieldRoles, aggregate_metric, metric_aggregation


def empty_figure(message: str) -> go.Figure:
    """Create an empty Plotly figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=360, margin=dict(l=30, r=30, t=50, b=30))
    return fig


def create_time_trend_chart(df: pd.DataFrame, roles: FieldRoles, metric_columns: list[str]) -> go.Figure:
    """Create a time trend chart for selected numeric metrics."""
    if not roles.time_column or roles.time_column not in df.columns:
        return empty_figure("未识别到可用时间列")
    selected_metrics = [column for column in metric_columns if column in df.columns][:3]
    if not selected_metrics:
        return empty_figure("未识别到可用于趋势分析的数值列")

    trend_df = df.dropna(subset=[roles.time_column]).copy()
    if trend_df.empty:
        return empty_figure("时间列没有可分析的数据")
    trend_df["_time_bucket"] = trend_df[roles.time_column].dt.floor("D")
    aggregated = trend_df.groupby("_time_bucket")[selected_metrics].mean().reset_index()
    melted = aggregated.melt("_time_bucket", var_name="指标", value_name="数值")

    fig = px.line(melted, x="_time_bucket", y="数值", color="指标", markers=True, title="时间趋势")
    fig.update_layout(xaxis_title="时间", yaxis_title="数值", height=420, margin=dict(l=30, r=30, t=60, b=40))
    return fig


def create_region_comparison_chart(df: pd.DataFrame, roles: FieldRoles, metric_column: str | None) -> go.Figure:
    """Create a regional comparison chart for one selected metric."""
    if not roles.region_column or roles.region_column not in df.columns:
        return empty_figure("未识别到可用区域列")
    if not metric_column or metric_column not in df.columns:
        return empty_figure("未选择可用于区域对比的指标")

    agg = metric_aggregation(metric_column, roles)
    region_df = aggregate_metric(df, roles.region_column, metric_column, agg).head(15)
    agg_label = "总量" if agg == "sum" else "均值"
    fig = px.bar(
        region_df,
        x=roles.region_column,
        y=metric_column,
        title=f"区域对比：{metric_column}（{agg_label}）",
        text_auto=".2s",
    )
    fig.update_layout(xaxis_title="区域", yaxis_title=metric_column, height=420, margin=dict(l=30, r=30, t=60, b=80))
    return fig


def create_anomaly_chart(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    roles: FieldRoles,
    metric_column: str | None,
) -> go.Figure:
    """Create a scatter chart and highlight detected anomaly points."""
    if not metric_column or metric_column not in df.columns:
        return empty_figure("未选择可用于异常检测的指标")

    plot_df = df.copy()
    plot_df["_row_id"] = range(1, len(plot_df) + 1)
    x_column = roles.time_column if roles.time_column and roles.time_column in plot_df.columns else "_row_id"
    color_column = roles.region_column if roles.region_column and roles.region_column in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=x_column,
        y=metric_column,
        color=color_column,
        title=f"异常点检测：{metric_column}",
        hover_data=[column for column in [roles.region_column, roles.time_column] if column and column in plot_df.columns],
    )

    if not anomalies.empty and "_anomaly_metric" in anomalies.columns:
        anomaly_points = anomalies[anomalies["_anomaly_metric"] == metric_column].copy()
        if not anomaly_points.empty:
            if x_column == "_row_id":
                anomaly_points["_row_id"] = anomaly_points["_source_index"] + 1
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points[x_column],
                    y=anomaly_points[metric_column],
                    mode="markers",
                    marker=dict(size=14, color="#e11d48", symbol="x"),
                    name="异常点",
                )
            )

    fig.update_layout(xaxis_title="时间" if x_column != "_row_id" else "记录序号", yaxis_title=metric_column)
    fig.update_layout(height=420, margin=dict(l=30, r=30, t=60, b=50))
    return fig


def create_numeric_distribution_chart(df: pd.DataFrame, metric_column: str | None) -> go.Figure:
    """Create a histogram and boxplot for one numeric metric."""
    if not metric_column or metric_column not in df.columns:
        return empty_figure("未选择可用于分布分析的指标")
    fig = px.histogram(df, x=metric_column, marginal="box", title=f"指标分布：{metric_column}")
    fig.update_layout(xaxis_title=metric_column, yaxis_title="记录数", height=420, margin=dict(l=30, r=30, t=60, b=40))
    return fig
