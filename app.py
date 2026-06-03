"""Streamlit entrypoint for the Mobility Insight Agent demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Fallback when python-dotenv is not installed in the active environment."""
        return False

from src.agents import (
    AnalysisAgent,
    AnalysisInput,
    DataProfilerAgent,
    DataProfilerInput,
    InsightAgent,
    InsightInput,
    ProductSuggestionAgent,
    ProductSuggestionInput,
    ReportAgent,
    ReportInput,
)
from src.data_processing import FieldRoles, infer_field_roles
from src.visualization import (
    create_anomaly_chart,
    create_numeric_distribution_chart,
    create_region_comparison_chart,
    create_time_trend_chart,
)
from style import apply_global_styles, render_hero, render_suggestion_card


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = BASE_DIR / "sample_data.csv"


def read_csv_source(uploaded_file, use_sample: bool) -> tuple[pd.DataFrame | None, str | None]:
    """Read a user-uploaded CSV file or the bundled sample dataset."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    if use_sample and SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH), SAMPLE_DATA_PATH.name
    return None, None


def select_optional_column(label: str, columns: list[str], default_value: str | None, key: str) -> str | None:
    """Render a selectbox that can return either a column name or None."""
    options = ["未选择"] + columns
    default_index = options.index(default_value) if default_value in options else 0
    selected = st.selectbox(label, options=options, index=default_index, key=key)
    return None if selected == "未选择" else selected


def build_role_overrides(default_roles: FieldRoles, columns: list[str]) -> dict[str, str | None]:
    """Render role override controls and return a structured override dictionary."""
    with st.expander("字段角色设置", expanded=False):
        st.caption("系统会自动识别字段，你也可以手动覆盖，用于演示 Agent 的可控性。")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_column = select_optional_column("时间列", columns, default_roles.time_column, "time_column")
        with col2:
            region_column = select_optional_column("区域列", columns, default_roles.region_column, "region_column")
        with col3:
            congestion_column = select_optional_column("拥堵相关指标", columns, default_roles.congestion_column, "congestion_column")

        col4, col5 = st.columns(2)
        with col4:
            passenger_column = select_optional_column("客流相关指标", columns, default_roles.passenger_column, "passenger_column")
        with col5:
            accident_column = select_optional_column("事故相关指标", columns, default_roles.accident_column, "accident_column")

    return {
        "time_column": time_column,
        "region_column": region_column,
        "congestion_column": congestion_column,
        "passenger_column": passenger_column,
        "accident_column": accident_column,
    }


def render_metric_cards(kpis: dict[str, Any]) -> None:
    """Render top-level KPI cards for the dashboard."""
    cards = st.columns(4)
    cards[0].metric("记录数", f"{kpis.get('row_count', 0):,}")
    cards[1].metric("字段数", f"{kpis.get('column_count', 0):,}")
    cards[2].metric("区域数", "-" if kpis.get("region_count") is None else f"{kpis['region_count']:,}")
    cards[3].metric("异常点", f"{kpis.get('anomaly_count', 0):,}")

    detail_cards = st.columns(3)
    if kpis.get("avg_congestion") is not None:
        detail_cards[0].metric("拥堵均值", f"{kpis['avg_congestion']:.2f}")
    if kpis.get("total_passenger_flow") is not None:
        detail_cards[1].metric("客流总量", f"{kpis['total_passenger_flow']:,.0f}")
    if kpis.get("total_accidents") is not None:
        detail_cards[2].metric("事故总量", f"{kpis['total_accidents']:,.0f}")


def render_agent_payload(agent_title: str, input_summary: dict[str, Any], output_summary: dict[str, Any]) -> None:
    """Render structured input and output JSON for one agent."""
    with st.expander(f"{agent_title} 结构化输入 / 输出", expanded=False):
        input_col, output_col = st.columns(2)
        with input_col:
            st.markdown("**Input**")
            st.json(input_summary)
        with output_col:
            st.markdown("**Output**")
            st.json(output_summary)


def render_notes(notes: list[str]) -> None:
    """Render agent notes as a compact bullet list."""
    if not notes:
        return
    for note in notes:
        st.write(f"- {note}")


def render_value_judgement_cards(df: pd.DataFrame) -> None:
    """Render product suggestion value judgement as cards."""
    if df.empty:
        return
    st.dataframe(df, use_container_width=True)


def render_profiler_section(agent_input: DataProfilerInput, output) -> None:
    """Render data_profiler_agent output for portfolio demonstration."""
    st.subheader("数据概览区")
    render_agent_payload("data_profiler_agent", agent_input.summary(), output.summary())

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("**字段预览**")
        st.dataframe(output.column_profile, use_container_width=True)
    with right:
        st.markdown("**可分析指标**")
        st.dataframe(pd.DataFrame(output.analyzable_metrics), use_container_width=True)
        render_notes(output.notes)
        st.markdown("**清洗结果**")
        st.dataframe(output.cleaned_df.head(12), use_container_width=True)


def render_analysis_section(agent_input: AnalysisInput, output) -> None:
    """Render analysis_agent output and quantitative tables."""
    st.subheader("核心指标区")
    render_agent_payload("analysis_agent", agent_input.summary(), output.summary())
    render_metric_cards(output.kpis)
    if output.kpis.get("time_range"):
        st.caption(f"时间范围：{output.kpis['time_range']}")

    trend_tab, ranking_tab, anomaly_tab, corr_tab = st.tabs(["趋势结果", "排名结果", "异常点结果", "相关性结果"])
    with trend_tab:
        st.dataframe(output.trend_table if not output.trend_table.empty else pd.DataFrame({"提示": ["趋势结果为空"]}), use_container_width=True)
    with ranking_tab:
        st.dataframe(output.ranking_table if not output.ranking_table.empty else pd.DataFrame({"提示": ["排名结果为空"]}), use_container_width=True)
    with anomaly_tab:
        st.dataframe(output.anomalies.head(30) if not output.anomalies.empty else pd.DataFrame({"提示": ["未检测到显著异常点"]}), use_container_width=True)
    with corr_tab:
        if output.correlation_matrix.empty:
            st.dataframe(pd.DataFrame({"提示": ["相关性结果为空"]}), use_container_width=True)
        else:
            st.dataframe(output.correlation_matrix, use_container_width=True)
            st.dataframe(pd.DataFrame(output.correlation_pairs), use_container_width=True)


def render_chart_section(profiler_output, analysis_output, selected_metrics: list[str], primary_metric: str | None) -> None:
    """Render Plotly charts derived from the structured agent outputs."""
    st.subheader("图表分析区")
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["趋势图", "区域对比图", "异常点图", "分布图"])
    with chart_tab1:
        st.plotly_chart(create_time_trend_chart(profiler_output.cleaned_df, profiler_output.roles, selected_metrics), use_container_width=True)
    with chart_tab2:
        st.plotly_chart(create_region_comparison_chart(profiler_output.cleaned_df, profiler_output.roles, primary_metric), use_container_width=True)
    with chart_tab3:
        st.plotly_chart(create_anomaly_chart(profiler_output.cleaned_df, analysis_output.anomalies, profiler_output.roles, primary_metric), use_container_width=True)
    with chart_tab4:
        st.plotly_chart(create_numeric_distribution_chart(profiler_output.cleaned_df, primary_metric), use_container_width=True)


def render_insight_section(agent_input: InsightInput, output) -> None:
    """Render insight_agent output as business insight cards and structured JSON."""
    st.subheader("AI 洞察报告区")
    render_agent_payload("insight_agent", agent_input.summary(), output.summary())

    st.markdown('<div class="section-label">执行摘要</div>', unsafe_allow_html=True)
    st.info(output.executive_summary)

    insight_cols = st.columns(min(3, max(1, len(output.insights))))
    for index, insight in enumerate(output.insights):
        with insight_cols[index % len(insight_cols)]:
            st.markdown(
                f"""
                <div class="small-card">
                    <div class="small-card-title">{insight.title}</div>
                    <div class="small-card-body">{insight.evidence}</div>
                    <div class="small-card-meta">优先级：{insight.priority}</div>
                    <div class="small-card-body">{insight.implication}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**行动建议**")
        for recommendation in output.recommendations:
            st.write(f"- {recommendation}")
    with col2:
        st.markdown("**分析限制**")
        for limitation in output.limitations:
            st.write(f"- {limitation}")


def render_product_suggestion_section(agent_input: ProductSuggestionInput, output) -> None:
    """Render PM-style product suggestions and value judgement."""
    st.subheader("产品建议与价值判断区")
    render_agent_payload("product_suggestion_agent", agent_input.summary(), output.summary())

    st.markdown("**产品建议卡片**")
    for suggestion in output.suggestions:
        render_suggestion_card(
            category=suggestion.category,
            title=suggestion.title,
            recommendation=suggestion.recommendation,
            value_judgement=suggestion.value_judgement,
            impact_metrics=suggestion.impact_metrics,
            priority=suggestion.priority,
            effort=suggestion.effort,
            evidence=suggestion.evidence,
        )
    st.markdown("**价值判断总览**")
    render_value_judgement_cards(pd.DataFrame(output.value_judgements))
    st.markdown("**简历项目描述区**")
    st.info(output.resume_project_description)


def render_report_section(agent_input: ReportInput, output) -> None:
    """Render report_agent input, output, Markdown report, and download control."""
    st.subheader("报告输出")
    render_agent_payload("report_agent", agent_input.summary(), output.summary())
    if output.error:
        st.warning(f"LLM 调用失败，已回退到本地报告：{output.error}")
    st.caption(f"报告来源：{output.source}")
    st.markdown(output.markdown)
    st.download_button(
        "下载 Markdown 报告",
        data=output.markdown,
        file_name="mobility_insight_report.md",
        mime="text/markdown",
    )


def choose_analysis_metrics(numeric_columns: list[str]) -> tuple[list[str], str | None]:
    """Render metric selectors for analysis_agent and return selected metrics."""
    if not numeric_columns:
        st.info("未识别到数值列，analysis_agent 将只能输出有限结果。")
        return [], None
    selected_metrics = st.multiselect("analysis_agent 分析指标", options=numeric_columns, default=numeric_columns[:3])
    metric_options = ["未选择"] + numeric_columns
    default_primary = selected_metrics[0] if selected_metrics else numeric_columns[0]
    selected_primary = st.selectbox("区域排名与异常图主指标", metric_options, index=metric_options.index(default_primary))
    primary_metric = None if selected_primary == "未选择" else selected_primary
    return selected_metrics, primary_metric


def main() -> None:
    """Run the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="城市出行洞察 Agent", page_icon="A", layout="wide")
    apply_global_styles()
    render_hero()

    st.markdown('<div class="section-title">数据上传区</div>', unsafe_allow_html=True)
    upload_col, sample_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    with sample_col:
        use_sample = st.checkbox("使用示例数据", value=uploaded_file is None)

    raw_df, source_name = read_csv_source(uploaded_file, use_sample)
    if raw_df is None:
        st.info("请上传 CSV 文件，或勾选使用示例数据。")
        return

    st.caption(f"当前数据源：{source_name}")
    default_roles = infer_field_roles(raw_df)
    role_overrides = build_role_overrides(default_roles, list(raw_df.columns))

    profiler_agent = DataProfilerAgent()
    profiler_input = DataProfilerInput(source_name=source_name or "uploaded_csv", raw_df=raw_df, role_overrides=role_overrides)
    profiler_output = profiler_agent.run(profiler_input)

    selected_metrics, primary_metric = choose_analysis_metrics(profiler_output.roles.numeric_columns)

    analysis_agent = AnalysisAgent()
    analysis_input = AnalysisInput(
        cleaned_df=profiler_output.cleaned_df,
        roles=profiler_output.roles,
        selected_metrics=selected_metrics,
        primary_metric=primary_metric,
    )
    analysis_output = analysis_agent.run(analysis_input)

    insight_agent = InsightAgent()
    insight_input = InsightInput(profiler_output=profiler_output, analysis_output=analysis_output)
    insight_output = insight_agent.run(insight_input)

    product_suggestion_agent = ProductSuggestionAgent()
    product_suggestion_input = ProductSuggestionInput(
        profiler_output=profiler_output,
        analysis_output=analysis_output,
        insight_output=insight_output,
    )
    product_suggestion_output = product_suggestion_agent.run(product_suggestion_input)

    st.markdown('<div class="section-title">数据概览区</div>', unsafe_allow_html=True)
    render_profiler_section(profiler_input, profiler_output)

    st.markdown('<div class="section-title">核心指标区</div>', unsafe_allow_html=True)
    render_analysis_section(analysis_input, analysis_output)
    render_chart_section(profiler_output, analysis_output, selected_metrics, primary_metric)

    render_insight_section(insight_input, insight_output)
    render_product_suggestion_section(product_suggestion_input, product_suggestion_output)

    report_agent = ReportAgent()
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = st.toggle("运行 report_agent 时调用 LLM", value=False, help="需要在 .env 中配置 OPENAI_API_KEY")
    run_report = st.button("运行 report_agent / 刷新报告", type="primary")
    if use_llm and not has_key:
        st.info("未检测到 OPENAI_API_KEY，本次将使用本地规则报告。")

    report_input = ReportInput(
        profiler_output=profiler_output,
        analysis_output=analysis_output,
        insight_output=insight_output,
        product_suggestion_output=product_suggestion_output,
        use_llm=use_llm and run_report and has_key,
    )
    report_output = report_agent.run(report_input)
    render_report_section(report_input, report_output)


if __name__ == "__main__":
    main()
