"""Streamlit entrypoint for the Mobility Insight Agent demo."""

from __future__ import annotations

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
from src.visualization import create_anomaly_chart, create_numeric_distribution_chart, create_region_comparison_chart, create_time_trend_chart
from style import apply_global_styles, render_info_cards, render_prompt_cards, render_section_banner, render_suggestion_card


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
        c1, c2, c3 = st.columns(3)
        with c1:
            time_column = select_optional_column("时间列", columns, default_roles.time_column, "time_column")
        with c2:
            region_column = select_optional_column("区域列", columns, default_roles.region_column, "region_column")
        with c3:
            congestion_column = select_optional_column("拥堵指标", columns, default_roles.congestion_column, "congestion_column")
        c4, c5 = st.columns(2)
        with c4:
            passenger_column = select_optional_column("客流指标", columns, default_roles.passenger_column, "passenger_column")
        with c5:
            accident_column = select_optional_column("事故指标", columns, default_roles.accident_column, "accident_column")
    return {
        "time_column": time_column,
        "region_column": region_column,
        "congestion_column": congestion_column,
        "passenger_column": passenger_column,
        "accident_column": accident_column,
    }


def render_hero() -> None:
    """Render product-style hero."""
    st.markdown(
        """
        <style>
            .hero-shell{
                background: linear-gradient(135deg,#0b1f3a 0%,#12305a 48%,#1d4e89 100%);
                color:#fff;border-radius:18px;padding:28px 28px 22px;box-shadow:0 14px 34px rgba(13,31,57,.18);margin-bottom:18px;
            }
            .hero-title{font-size:2.05rem;font-weight:800;line-height:1.1;margin:0 0 8px 0;}
            .hero-subtitle{font-size:1rem;opacity:.92;line-height:1.65;max-width:980px;margin-bottom:14px;}
            .hero-tags{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;}
            .hero-tag{padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);font-weight:700;font-size:.84rem;}
            .hero-kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;}
            .hero-kpi{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);border-radius:14px;padding:12px 14px;}
            .hero-kpi-label{font-size:.78rem;opacity:.8;margin-bottom:5px;}
            .hero-kpi-value{font-size:1.02rem;font-weight:800;}
            @media (max-width: 900px){.hero-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
            @media (max-width: 640px){.hero-kpi-grid{grid-template-columns:1fr;}}
            .demo-card{background:#fff;border:1px solid #dbe3ee;border-radius:16px;box-shadow:0 10px 28px rgba(16,32,51,.08);padding:16px;}
            .demo-card-title{font-size:.98rem;font-weight:800;color:#102033;margin-bottom:8px;}
            .demo-card-sub{font-size:.9rem;color:#5f7187;line-height:1.55;}
            .grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;}
            .grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
            @media (max-width: 900px){.grid-3,.grid-2{grid-template-columns:1fr;}}
            .workflow-wrap{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;align-items:stretch;}
            @media (max-width: 1080px){.workflow-wrap{grid-template-columns:repeat(2,minmax(0,1fr));}}
            @media (max-width: 640px){.workflow-wrap{grid-template-columns:1fr;}}
            .workflow-card{background:#fff;border:1px solid #dbe3ee;border-radius:14px;box-shadow:0 10px 28px rgba(16,32,51,.08);padding:14px;}
            .workflow-name{font-weight:800;color:#102033;margin-bottom:6px;}
            .workflow-zh{color:#1f5eff;font-weight:700;font-size:.86rem;margin-bottom:8px;}
            .workflow-part{font-size:.88rem;color:#102033;line-height:1.55;margin-bottom:4px;}
            .workflow-arrow{display:flex;align-items:center;justify-content:center;color:#1f5eff;font-size:1.25rem;font-weight:900;}
            @media (max-width: 1080px){.workflow-arrow{display:none;}}
            .prompt-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;}
            @media (max-width: 900px){.prompt-grid{grid-template-columns:1fr;}}
            .prompt-card{background:#fff;border:1px solid #dbe3ee;border-radius:16px;box-shadow:0 10px 28px rgba(16,32,51,.08);padding:14px;}
            .prompt-card-title{font-weight:800;color:#102033;margin-bottom:10px;}
            .prompt-split{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
            @media (max-width: 700px){.prompt-split{grid-template-columns:1fr;}}
            .prompt-box{background:#f8fafc;border:1px solid #dbe3ee;border-radius:12px;padding:12px;}
            .prompt-box-title{font-weight:800;font-size:.84rem;color:#1f5eff;margin-bottom:6px;}
            .prompt-box pre{white-space:pre-wrap;margin:0;font-family:inherit;font-size:.88rem;line-height:1.55;color:#102033;}
            .insight-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
            @media (max-width: 900px){.insight-grid{grid-template-columns:1fr;}}
            .insight-card{background:#fff;border:1px solid #dbe3ee;border-radius:16px;box-shadow:0 10px 28px rgba(16,32,51,.08);padding:14px;}
            .insight-title{font-weight:800;margin-bottom:8px;color:#102033;}
            .insight-line{font-size:.9rem;line-height:1.6;color:#102033;margin-bottom:4px;}
            .badge{display:inline-flex;align-items:center;padding:4px 10px;border-radius:999px;background:#eef3fb;color:#40536b;font-weight:800;font-size:.76rem;}
            .badge-high{background:#eaf1ff;color:#1f5eff;}
            .badge-medium{background:#eef3fb;color:#40536b;}
            .badge-low{background:#f2f5f8;color:#6d7c90;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">AI Mobility Insight Agent</div>
            <div class="hero-subtitle">城市出行数据智能分析平台。上传交通运营数据后，系统自动完成字段识别、指标分析、异常检测、业务洞察生成、产品建议输出与 Markdown 报告生成，帮助交通治理和出行运营团队缩短从数据分析到决策执行的路径。</div>
            <div class="hero-tags">
                <span class="hero-tag">Data Profiling</span>
                <span class="hero-tag">Agent Workflow</span>
                <span class="hero-tag">Prompt Engineering</span>
                <span class="hero-tag">Business Insight</span>
                <span class="hero-tag">Product Suggestion</span>
            </div>
            <div class="hero-kpi-grid">
                <div class="hero-kpi"><div class="hero-kpi-label">Agent 节点</div><div class="hero-kpi-value">5 个</div></div>
                <div class="hero-kpi"><div class="hero-kpi-label">可分析指标</div><div class="hero-kpi-value">4+ 类</div></div>
                <div class="hero-kpi"><div class="hero-kpi-label">业务洞察</div><div class="hero-kpi-value">自动生成</div></div>
                <div class="hero-kpi"><div class="hero-kpi-label">报告输出</div><div class="hero-kpi-value">Markdown</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_positioning_section() -> None:
    """Render product positioning cards."""
    render_section_banner("这个 Agent 解决什么问题？", "面向交通治理、出行运营和数据产品建设场景。")
    st.markdown(
        """
        <div class="grid-3">
            <div class="demo-card"><div class="demo-card-title">目标用户</div><div class="demo-card-sub">交通管理部门、出行平台运营团队、智慧交通产品经理、数据分析师</div></div>
            <div class="demo-card"><div class="demo-card-title">核心痛点</div><div class="demo-card-sub">交通数据分散、字段复杂、人工分析耗时、洞察口径不统一、报告产出慢</div></div>
            <div class="demo-card"><div class="demo-card-title">产品价值</div><div class="demo-card-sub">自动识别数据结构、生成业务洞察、输出产品建议、缩短分析到决策链路</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_upload_section(raw_df: pd.DataFrame, source_name: str | None, roles: FieldRoles) -> None:
    """Render data upload summary cards."""
    render_section_banner("数据上传", "保留现有上传组件。")
    st.caption(f"当前数据源：{source_name}")
    st.markdown(
        f"""
        <div class="grid-3">
            <div class="demo-card"><div class="demo-card-title">数据规模</div><div class="demo-card-sub">{len(raw_df):,} 行 / {len(raw_df.columns):,} 列</div></div>
            <div class="demo-card"><div class="demo-card-title">时间字段</div><div class="demo-card-sub">{roles.time_column or '未识别'}</div></div>
            <div class="demo-card"><div class="demo-card-title">区域字段</div><div class="demo-card-sub">{roles.region_column or '未识别'}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        st.dataframe(output.column_profile, use_container_width=True, hide_index=True)

    with tab_metrics:
        metric_df = pd.DataFrame(output.analyzable_metrics)
        if metric_df.empty:
            st.info("当前数据未识别到可分析指标。")
        else:
            cols = [c for c in ["field_name", "metric_name", "metric_type", "analysis_use"] if c in metric_df.columns]
            st.dataframe(metric_df[cols] if cols else metric_df, use_container_width=True, hide_index=True)
            metric_cards = []
            for _, row in metric_df.head(5).iterrows():
                metric_cards.append(
                    f"<div class='demo-card'><div class='demo-card-title'>{row.get('metric_name', row.get('field_name', '指标'))}</div><div class='demo-card-sub'>{row.get('metric_type', 'numeric')} · {row.get('analysis_use', '通用指标分析')}</div></div>"
                )
            st.markdown(f"<div class='grid-3'>{''.join(metric_cards)}</div>", unsafe_allow_html=True)

    with tab_quality:
        quality = output.quality_summary
        st.markdown(
            f"""
            <div class="grid-3">
                <div class="demo-card"><div class="demo-card-title">总行数</div><div class="demo-card-sub">{len(output.cleaned_df):,}</div></div>
                <div class="demo-card"><div class="demo-card-title">总字段数</div><div class="demo-card-sub">{len(output.cleaned_df.columns):,}</div></div>
                <div class="demo-card"><div class="demo-card-title">缺失值</div><div class="demo-card-sub">{quality.get('missing_total', 0):,}</div></div>
                <div class="demo-card"><div class="demo-card-title">重复值</div><div class="demo-card-sub">{quality.get('duplicate_rows_removed', quality.get('duplicate_rows', 0)):,}</div></div>
                <div class="demo-card"><div class="demo-card-title">日期范围</div><div class="demo-card-sub">{quality.get('time_range', '未识别')}</div></div>
                <div class="demo-card"><div class="demo-card-title">清洗状态</div><div class="demo-card-sub">已完成</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_cards(kpis: dict[str, Any]) -> None:
    """Render custom KPI cards."""
    cards = [
        ("🚗", "总流量", f"{kpis.get('total_passenger_flow', 0):,.0f}" if kpis.get("total_passenger_flow") is not None else "样例数据", "累计客流规模"),
        ("🚦", "平均拥堵指数", f"{kpis.get('avg_congestion', 0):.2f}" if kpis.get("avg_congestion") is not None else "样例数据", "反映道路运行压力"),
        ("⚠️", "总事故数", f"{kpis.get('total_accidents', 0):,.0f}" if kpis.get("total_accidents") is not None else "样例数据", "用于安全风险监测"),
        ("⚡", "平均速度", f"{kpis.get('avg_speed', 0):.2f}" if kpis.get("avg_speed") is not None else "样例数据", "用于通行效率判断"),
    ]
    st.markdown(
        "<div class='grid-3'>"
        + "".join(
            [
                f"<div class='demo-card'><div class='demo-card-title'>{icon} {title}</div><div style='font-size:1.5rem;font-weight:800;color:#102033;margin:4px 0;'>{value}</div><div class='demo-card-sub'>{desc}</div></div>"
                for icon, title, value, desc in cards
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_analysis_section(output) -> None:
    """Render analysis section."""
    render_section_banner("核心指标看板", "总流量、平均拥堵指数、总事故数、平均速度。")
    render_kpi_cards(output.kpis)


def render_chart_section(profiler_output, analysis_output, selected_metrics: list[str], primary_metric: str | None) -> None:
    """Render chart cards."""
    render_section_banner("图表分析", "趋势、区域对比、异常点检测和指标分布。")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='demo-card'><div class='demo-card-title'>时间趋势</div><div class='demo-card-sub'>用于观察拥堵指数随时间变化的趋势与异常波动。</div></div>", unsafe_allow_html=True)
        st.plotly_chart(create_time_trend_chart(profiler_output.cleaned_df, profiler_output.roles, selected_metrics), use_container_width=True)
    with right:
        st.markdown("<div class='demo-card'><div class='demo-card-title'>区域对比</div><div class='demo-card-sub'>用于比较不同区域的客流、拥堵和事故表现。</div></div>", unsafe_allow_html=True)
        st.plotly_chart(create_region_comparison_chart(profiler_output.cleaned_df, profiler_output.roles, primary_metric), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='demo-card'><div class='demo-card-title'>异常点</div><div class='demo-card-sub'>用于识别流量和拥堵的异常波动。</div></div>", unsafe_allow_html=True)
        st.plotly_chart(create_anomaly_chart(profiler_output.cleaned_df, analysis_output.anomalies, profiler_output.roles, primary_metric), use_container_width=True)
    with right:
        st.markdown("<div class='demo-card'><div class='demo-card-title'>指标分布</div><div class='demo-card-sub'>用于查看指标分布形态与离散程度。</div></div>", unsafe_allow_html=True)
        st.plotly_chart(create_numeric_distribution_chart(profiler_output.cleaned_df, primary_metric), use_container_width=True)


def render_workflow_section() -> None:
    """Render the workflow as connected cards."""
    render_section_banner("Agent Workflow", "CSV上传 → Data Profiler → Analysis Agent → Insight Agent → Product Agent → Report Agent")
    st.markdown(
        """
        <div class="workflow-wrap">
            <div class="workflow-card"><div class="workflow-name">CSV Upload</div><div class="workflow-zh">数据输入</div><div class="workflow-part"><b>输入：</b>CSV、字段名、样本数据</div><div class="workflow-part"><b>处理：</b>上传到系统</div><div class="workflow-part"><b>输出：</b>待识别数据</div></div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-card"><div class="workflow-name">Data Profiler</div><div class="workflow-zh">数据识别 Agent</div><div class="workflow-part"><b>输入：</b>CSV、字段名、样本数据</div><div class="workflow-part"><b>处理：</b>识别日期、区域、数值字段，完成质量检查</div><div class="workflow-part"><b>输出：</b>字段角色、可分析指标、数据质量摘要</div></div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-card"><div class="workflow-name">Analysis Agent</div><div class="workflow-zh">指标分析 Agent</div><div class="workflow-part"><b>输入：</b>清洗后的结构化数据</div><div class="workflow-part"><b>处理：</b>趋势、排名、高峰时段、异常波动分析</div><div class="workflow-part"><b>输出：</b>核心指标、图表、异常点</div></div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-card"><div class="workflow-name">Insight Agent</div><div class="workflow-zh">业务洞察 Agent</div><div class="workflow-part"><b>输入：</b>分析结果、KPI、异常点</div><div class="workflow-part"><b>处理：</b>把数值结果转成发现、判断、影响和建议</div><div class="workflow-part"><b>输出：</b>业务洞察卡片</div></div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-card"><div class="workflow-name">Product Agent</div><div class="workflow-zh">产品建议 Agent</div><div class="workflow-part"><b>输入：</b>业务洞察、数据证据、目标用户</div><div class="workflow-part"><b>处理：</b>生成交通治理、出行运营和数据产品优化建议</div><div class="workflow-part"><b>输出：</b>产品建议卡片、价值判断、优先级</div></div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-card"><div class="workflow-name">Report Agent</div><div class="workflow-zh">报告生成 Agent</div><div class="workflow-part"><b>输入：</b>全部分析结果</div><div class="workflow-part"><b>处理：</b>整理成结构化 Markdown 报告</div><div class="workflow-part"><b>输出：</b>可下载报告</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_section() -> None:
    """Render prompt cards with template and example."""
    render_section_banner("Prompt Engineering", "Prompt 模板 + 输出示例。")
    st.markdown(
        """
        <div class="prompt-grid">
            <div class="prompt-card">
                <div class="prompt-card-title">业务洞察 Prompt</div>
                <div class="prompt-split">
                    <div class="prompt-box"><div class="prompt-box-title">Prompt 模板</div><pre>Role:
你是一名智慧交通数据分析师。

Task:
请基于指标结果生成业务洞察。

Output:
发现
判断
影响
建议</pre></div>
                    <div class="prompt-box"><div class="prompt-box-title">输出示例</div><pre>发现：某区域高峰期拥堵指数明显高于均值。
判断：该区域通勤需求集中，存在潮汐交通压力。
影响：可能降低用户出行体验，并增加道路运行风险。
建议：优化高峰期信号配时，并增加重点区域运行监控。</pre></div>
                </div>
            </div>
            <div class="prompt-card">
                <div class="prompt-card-title">产品建议 Prompt</div>
                <div class="prompt-split">
                    <div class="prompt-box"><div class="prompt-box-title">Prompt 模板</div><pre>Role:
你是一名智慧交通产品经理。

Task:
请根据业务洞察生成产品建议。

Output:
建议标题
面向角色
数据证据
产品价值
优先级
实施复杂度</pre></div>
                    <div class="prompt-box"><div class="prompt-box-title">输出示例</div><pre>建议标题：高峰拥堵预警模块
面向角色：交通管理部门
产品价值：提前发现高拥堵风险，辅助调度和治理决策。</pre></div>
                </div>
            </div>
            <div class="prompt-card">
                <div class="prompt-card-title">报告生成 Prompt</div>
                <div class="prompt-split">
                    <div class="prompt-box"><div class="prompt-box-title">Prompt 模板</div><pre>Role:
你是一名数据产品经理。

Task:
请将分析结果整理为 Markdown 报告。

Output:
数据概览
关键发现
业务洞察
产品建议
下一步行动</pre></div>
                    <div class="prompt-box"><div class="prompt-box-title">输出示例</div><pre>生成结构化报告，用于业务复盘和项目汇报。</pre></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_section(output) -> None:
    """Render business insight cards."""
    render_section_banner("业务洞察", "发现、判断、影响、建议。")
    cards = []
    for insight in output.insights[:4]:
        cards.append(
            f"<div class='insight-card'><div class='insight-title'>{insight.title}</div><div class='insight-line'><b>发现：</b>{insight.evidence}</div><div class='insight-line'><b>判断：</b>{insight.implication}</div><div class='insight-line'><b>建议：</b>{output.recommendations[0] if output.recommendations else '继续关注并优化'} </div><div style='margin-top:8px;'><span class='badge'>Priority {insight.priority}</span></div></div>"
        )
    st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_product_suggestion_section(output) -> None:
    """Render six product suggestion cards."""
    render_section_banner("产品建议", "6 张建议卡片。")
    cols = st.columns(2)
    for index, suggestion in enumerate(output.suggestions):
        with cols[index % 2]:
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


def render_report_section(output: Any) -> None:
    """Render report export section."""
    render_section_banner("报告导出", "仅保留 Markdown 导出按钮。")
    if not getattr(output, "markdown", "").strip():
        st.info("请先完成分析后再导出报告")
        return
    st.download_button("下载 Markdown 报告", data=output.markdown, file_name="mobility_insight_report.md", mime="text/markdown")


def main() -> None:
    """Run the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="AI Mobility Insight Agent", page_icon="A", layout="wide")
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
    role_overrides = build_role_overrides(inferred_roles, list(raw_df.columns))

    profiler_output = DataProfilerAgent().run(DataProfilerInput(source_name=source_name or "uploaded_csv", raw_df=raw_df, role_overrides=role_overrides))
    selected_metrics, primary_metric = choose_analysis_metrics(profiler_output.roles.numeric_columns)

    analysis_output = AnalysisAgent().run(
        AnalysisInput(
            cleaned_df=profiler_output.cleaned_df,
            roles=profiler_output.roles,
            selected_metrics=selected_metrics,
            primary_metric=primary_metric,
        )
    )
    insight_output = InsightAgent().run(InsightInput(profiler_output=profiler_output, analysis_output=analysis_output))
    product_suggestion_output = ProductSuggestionAgent().run(
        ProductSuggestionInput(
            profiler_output=profiler_output,
            analysis_output=analysis_output,
            insight_output=insight_output,
        )
    )

    render_data_understanding_section(profiler_output)
    render_analysis_section(analysis_output)
    render_chart_section(profiler_output, analysis_output, selected_metrics, primary_metric)
    render_workflow_section()
    render_prompt_section()
    render_insight_section(insight_output)
    render_product_suggestion_section(product_suggestion_output)

    report_output = ReportAgent().run(
        ReportInput(
            profiler_output=profiler_output,
            analysis_output=analysis_output,
            insight_output=insight_output,
            product_suggestion_output=product_suggestion_output,
            use_llm=False,
        )
    )
    render_report_section(report_output)


if __name__ == "__main__":
    main()
