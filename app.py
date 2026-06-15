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
from style import (
    apply_global_styles,
    render_hero,
    render_info_cards,
    render_prompt_cards,
    render_section_banner,
    render_suggestion_card,
    render_workflow_cards,
)


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = BASE_DIR / "sample_data.csv"


def read_csv_source(uploaded_file, use_sample: bool) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV source from upload or sample data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    if use_sample and SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH), SAMPLE_DATA_PATH.name
    return None, None


def select_optional_column(label: str, columns: list[str], default_value: str | None, key: str) -> str | None:
    """Render a selectbox that can return a column or None."""
    options = ["未选择"] + columns
    default_index = options.index(default_value) if default_value in options else 0
    selected = st.selectbox(label, options=options, index=default_index, key=key)
    return None if selected == "未选择" else selected


def build_role_overrides(default_roles: FieldRoles, columns: list[str]) -> dict[str, str | None]:
    """Render manual overrides for the inferred field roles."""
    with st.expander("字段角色设置", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            time_column = select_optional_column("时间列", columns, default_roles.time_column, "time_column")
        with col2:
            region_column = select_optional_column("区域列", columns, default_roles.region_column, "region_column")
        with col3:
            congestion_column = select_optional_column("拥堵指标", columns, default_roles.congestion_column, "congestion_column")

        col4, col5 = st.columns(2)
        with col4:
            passenger_column = select_optional_column("客流指标", columns, default_roles.passenger_column, "passenger_column")
        with col5:
            accident_column = select_optional_column("事故指标", columns, default_roles.accident_column, "accident_column")

    return {
        "time_column": time_column,
        "region_column": region_column,
        "congestion_column": congestion_column,
        "passenger_column": passenger_column,
        "accident_column": accident_column,
    }


def render_positioning_section() -> None:
    """Render the product positioning cards."""
    render_section_banner("这个 Agent 解决什么问题？", "面向交通治理、出行运营和数据产品建设场景的 AI 数据分析 Agent。")
    render_info_cards(
        [
            {"title": "目标用户", "body": "交通管理部门、出行平台运营团队、智慧交通产品经理、数据分析师。"},
            {"title": "核心痛点", "body": "数据分析依赖人工报表，交通指标理解门槛高，洞察产出效率低，建议缺乏统一标准。"},
            {"title": "产品价值", "body": "自动识别交通数据结构，自动生成业务洞察和产品建议，缩短分析到决策链路。"},
        ]
    )


def render_data_upload_section(raw_df: pd.DataFrame, source_name: str | None, roles: FieldRoles) -> None:
    """Render data upload summary."""
    render_section_banner("数据上传", "保留现有上传组件。")
    st.caption(f"当前数据源：{source_name}")
    cols = st.columns(3)
    cols[0].metric("数据规模", f"{len(raw_df):,} 行 / {len(raw_df.columns):,} 列")
    cols[1].metric("时间列", roles.time_column or "未识别")
    cols[2].metric("区域列", roles.region_column or "未识别")

    if roles.time_column and roles.time_column in raw_df.columns:
        time_series = pd.to_datetime(raw_df[roles.time_column], errors="coerce")
        valid_times = time_series.dropna()
        if not valid_times.empty:
            st.caption(f"日期范围：{valid_times.min().date()} - {valid_times.max().date()}")
        else:
            st.caption("日期范围：未识别")
    else:
        st.caption("日期范围：未识别")


def render_data_understanding_section(output) -> None:
    """Render tabs for field preview, analyzable metrics, and data quality."""
    render_section_banner("数据理解", "字段预览、可分析指标和数据质量。")
    tab_fields, tab_metrics, tab_quality = st.tabs(["字段预览", "可分析指标", "数据质量"])

    with tab_fields:
        field_profile = output.column_profile.copy()
        if "field_name" not in field_profile.columns:
            field_profile.insert(0, "field_name", field_profile.index.astype(str))
        display_columns = [c for c in ["field_name", "column_type", "analysis_role", "is_analyzable"] if c in field_profile.columns]
        st.dataframe(field_profile[display_columns] if display_columns else field_profile, use_container_width=True, hide_index=True)

    with tab_metrics:
        metric_df = pd.DataFrame(output.analyzable_metrics)
        if metric_df.empty:
            st.info("当前数据未识别到可分析指标。")
        else:
            metric_display_cols = [c for c in ["field_name", "label", "role", "aggregation", "mean"] if c in metric_df.columns]
            st.dataframe(metric_df[metric_display_cols] if metric_display_cols else metric_df, use_container_width=True, hide_index=True)
            st.caption("系统优先识别流量、拥堵指数、事故数、平均速度、订单量等指标字段。")

    with tab_quality:
        quality = output.quality_summary
        c1, c2, c3 = st.columns(3)
        c1.metric("总行数", f"{len(output.cleaned_df):,}")
        c2.metric("总字段数", f"{len(output.cleaned_df.columns):,}")
        c3.metric("缺失值", f"{quality.get('missing_total', 0):,}")
        c4, c5, c6 = st.columns(3)
        c4.metric("重复值", f"{quality.get('duplicate_rows', 0):,}")
        c5.metric("日期范围", quality.get("time_range", "未识别"))
        c6.metric("清洗状态", quality.get("status", "已完成"))


def render_kpi_cards(kpis: dict[str, Any]) -> None:
    """Render top KPI cards."""
    cols = st.columns(4)
    cols[0].metric("总流量", f"{kpis.get('total_passenger_flow', 0):,.0f}")
    cols[1].metric("平均拥堵指数", f"{kpis.get('avg_congestion', 0):.2f}" if kpis.get("avg_congestion") is not None else "-")
    cols[2].metric("总事故数", f"{kpis.get('total_accidents', 0):,.0f}")
    cols[3].metric("平均速度", f"{kpis.get('avg_speed', 0):.2f}" if kpis.get("avg_speed") is not None else "-")


def render_analysis_section(agent_input: AnalysisInput, output) -> None:
    """Render analysis agent section."""
    render_section_banner("核心指标看板", "趋势、区域对比、异常波动和相关性集中展示。")
    render_kpi_cards(output.kpis)
    if output.kpis.get("time_range"):
        st.caption(f"时间范围：{output.kpis['time_range']}")

    trend_tab, region_tab, anomaly_tab, corr_tab = st.tabs(["趋势分析", "区域对比", "异常波动", "相关性"])
    with trend_tab:
        st.dataframe(output.trend_table if not output.trend_table.empty else pd.DataFrame({"提示": ["趋势结果为空"]}), use_container_width=True)
    with region_tab:
        st.dataframe(output.ranking_table if not output.ranking_table.empty else pd.DataFrame({"提示": ["排名结果为空"]}), use_container_width=True)
    with anomaly_tab:
        st.dataframe(output.anomalies.head(30) if not output.anomalies.empty else pd.DataFrame({"提示": ["未检测到显著异常"]}), use_container_width=True)
    with corr_tab:
        if output.correlation_matrix.empty:
            st.dataframe(pd.DataFrame({"提示": ["相关性结果为空"]}), use_container_width=True)
        else:
            st.dataframe(output.correlation_matrix, use_container_width=True)


def render_chart_section(profiler_output, analysis_output, selected_metrics: list[str], primary_metric: str | None) -> None:
    """Render charts in one compact section."""
    render_section_banner("图表分析", "趋势、区域对比、高峰时段和异常波动集中展示。")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_time_trend_chart(profiler_output.cleaned_df, profiler_output.roles, selected_metrics), use_container_width=True)
    with c2:
        st.plotly_chart(create_region_comparison_chart(profiler_output.cleaned_df, profiler_output.roles, primary_metric), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(create_anomaly_chart(profiler_output.cleaned_df, analysis_output.anomalies, profiler_output.roles, primary_metric), use_container_width=True)
    with c4:
        st.plotly_chart(create_numeric_distribution_chart(profiler_output.cleaned_df, primary_metric), use_container_width=True)


def render_workflow_section() -> None:
    """Render the agent workflow cards."""
    render_section_banner("Agent Workflow", "数据识别 Agent -> 指标分析 Agent -> 业务洞察 Agent -> 产品建议 Agent -> 报告生成 Agent")
    render_workflow_cards(
        [
            {"title": "数据识别 Agent", "input": "CSV、字段名、样本数据", "process": "识别时间列、区域列和数值列，并完成清洗与质量检查。", "output": "字段角色、可分析指标、数据质量摘要。"},
            {"title": "指标分析 Agent", "input": "清洗后的数据、分析指标", "process": "计算趋势、排名、高峰时段、异常波动和工作日/周末差异。", "output": "趋势表、排名表、异常点和相关性结果。"},
            {"title": "业务洞察 Agent", "input": "分析结果、KPI、异常点和相关性", "process": "把数值结果翻译成现象、判断、影响和下一步建议。", "output": "洞察卡片、执行摘要、行动建议。"},
            {"title": "产品建议 Agent", "input": "业务洞察、分析结果、数据质量", "process": "生成交通治理、出行平台运营和数据产品优化建议。", "output": "建议卡片、价值判断、影响指标。"},
            {"title": "报告生成 Agent", "input": "结构化分析结果", "process": "汇总为 Markdown 报告，可本地生成或 LLM 生成。", "output": "可下载 Markdown 报告。"},
        ]
    )


def render_prompt_section() -> None:
    """Render prompt engineering cards."""
    render_section_banner("Prompt Engineering", "用结构化提示词把业务洞察、产品建议和报告输出稳定下来。")
    render_prompt_cards(
        [
            {"title": "业务洞察 Prompt", "body": "目标：生成业务洞察。输出结构：发现、判断、影响、建议。"},
            {"title": "产品建议 Prompt", "body": "目标：生成产品建议。输出结构：建议标题、面向角色、数据证据、产品价值、优先级。"},
            {"title": "报告生成 Prompt", "body": "目标：生成 Markdown 报告。重点展示结构化摘要、结论和建议片段。"},
        ]
    )


def render_insight_section(output) -> None:
    """Render business insights in a structured card format."""
    render_section_banner("业务洞察", "发现、判断、影响、建议。")
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-card-title">执行摘要</div>
            <div class="small-card-body">{output.executive_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for insight in output.insights:
        st.markdown(
            f"""
            <div class="small-card" style="margin-top:12px;">
                <div class="small-card-title">{insight.title}</div>
                <div class="small-card-body"><strong>发现：</strong>{insight.evidence}</div>
                <div class="small-card-body"><strong>判断：</strong>{insight.implication}</div>
                <div class="small-card-meta">优先级：{insight.priority}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_product_suggestion_section(output) -> None:
    """Render six product suggestion cards."""
    render_section_banner("产品建议", "6 张建议卡片，直接面向治理、运营和数据产品场景。")
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


def render_report_section(output: Any) -> None:
    """Render report export only."""
    render_section_banner("报告导出", "仅保留 Markdown 导出按钮。")
    if not getattr(output, "markdown", "").strip():
        st.info("请先完成分析")
        return
    st.download_button(
        "下载 Markdown 报告",
        data=output.markdown,
        file_name="mobility_insight_report.md",
        mime="text/markdown",
    )


def choose_analysis_metrics(numeric_columns: list[str]) -> tuple[list[str], str | None]:
    """Render metric selectors."""
    if not numeric_columns:
        st.info("未识别到数值列，analysis_agent 将只能输出有限结果。")
        return [], None
    selected_metrics = st.multiselect("analysis_agent 分析指标", options=numeric_columns, default=numeric_columns[:3])
    metric_options = ["未选择"] + numeric_columns
    default_primary = selected_metrics[0] if selected_metrics else numeric_columns[0]
    selected_primary = st.selectbox("区域排名与异常图主指标", metric_options, index=metric_options.index(default_primary))
    return selected_metrics, None if selected_primary == "未选择" else selected_primary


def main() -> None:
    """Run the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="城市出行洞察 Agent", page_icon="A", layout="wide")
    apply_global_styles()
    render_hero()

    render_positioning_section()

    render_section_banner("数据上传", "上传 CSV 或直接使用示例数据。")
    upload_col, sample_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    with sample_col:
        use_sample = st.checkbox("使用示例数据", value=uploaded_file is None)

    raw_df, source_name = read_csv_source(uploaded_file, use_sample)
    if raw_df is None:
        st.info("请上传 CSV 文件，或勾选使用示例数据。")
        return

    inferred_roles = infer_field_roles(raw_df)
    render_data_upload_section(raw_df, source_name, inferred_roles)
    default_roles = inferred_roles
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

    render_data_understanding_section(profiler_output)
    render_kpi_cards(analysis_output.kpis)
    render_chart_section(profiler_output, analysis_output, selected_metrics, primary_metric)
    render_workflow_section()
    render_prompt_section()
    render_insight_section(insight_output)
    render_product_suggestion_section(product_suggestion_output)

    report_agent = ReportAgent()
    report_input = ReportInput(
        profiler_output=profiler_output,
        analysis_output=analysis_output,
        insight_output=insight_output,
        product_suggestion_output=product_suggestion_output,
        use_llm=False,
    )
    report_output = report_agent.run(report_input)
    render_report_section(report_output)


if __name__ == "__main__":
    main()
