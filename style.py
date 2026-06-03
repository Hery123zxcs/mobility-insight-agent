"""Shared visual styling helpers for the Streamlit demo."""

from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """Inject restrained B-end SaaS styling for the demo experience."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fb;
                --panel: #ffffff;
                --panel-soft: #f8fafc;
                --border: #dbe3ee;
                --text: #102033;
                --muted: #5f7187;
                --primary: #1f5eff;
                --primary-soft: rgba(31, 94, 255, 0.08);
                --shadow: 0 10px 28px rgba(16, 32, 51, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(31, 94, 255, 0.06), transparent 28%),
                    linear-gradient(180deg, #f7f9fc 0%, #f3f6fb 100%);
                color: var(--text);
            }

            html, body, [class*="css"] {
                font-family: "Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            }

            section[data-testid="stSidebar"] {
                background: #f8fbff;
                border-right: 1px solid var(--border);
            }

            .block-container {
                padding-top: 1.1rem;
                padding-bottom: 2.5rem;
            }

            .hero {
                background: linear-gradient(135deg, #0f1f3a 0%, #17325f 48%, #214c8a 100%);
                color: white;
                border-radius: 18px;
                padding: 28px 28px 24px 28px;
                box-shadow: var(--shadow);
                margin-bottom: 20px;
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: 0;
                line-height: 1.15;
                margin-bottom: 10px;
            }

            .hero-subtitle {
                color: rgba(255, 255, 255, 0.84);
                font-size: 0.98rem;
                line-height: 1.7;
                margin-bottom: 16px;
                max-width: 920px;
            }

            .tag-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }

            .tag {
                display: inline-flex;
                align-items: center;
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.18);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text);
                margin: 26px 0 12px 0;
                padding-left: 6px;
                border-left: 4px solid var(--primary);
            }

            .section-label {
                font-size: 0.92rem;
                font-weight: 700;
                color: var(--muted);
                margin-bottom: 8px;
            }

            div[data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px 16px 12px 16px;
                box-shadow: var(--shadow);
            }

            div[data-testid="stMetric"] label {
                color: var(--muted);
                font-size: 0.82rem;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--text);
                font-size: 1.55rem;
                font-weight: 700;
            }

            .stButton > button {
                background: var(--primary);
                color: white;
                border: 1px solid var(--primary);
                border-radius: 12px;
                padding: 0.55rem 1rem;
                font-weight: 600;
                box-shadow: 0 8px 16px rgba(31, 94, 255, 0.18);
            }

            .stButton > button:hover {
                background: #174fd3;
                border-color: #174fd3;
            }

            .stDownloadButton > button {
                background: var(--panel);
                color: var(--text);
                border: 1px solid var(--border);
                border-radius: 12px;
                font-weight: 600;
            }

            .stDataFrame {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
            }

            hr {
                border: none;
                border-top: 1px solid var(--border);
                margin: 1rem 0 1.2rem 0;
            }

            .small-card,
            .suggestion-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px 16px 14px 16px;
                box-shadow: var(--shadow);
                height: 100%;
            }

            .small-card-title,
            .suggestion-title {
                font-weight: 700;
                color: var(--text);
                margin-bottom: 8px;
                font-size: 0.98rem;
            }

            .small-card-body,
            .suggestion-body {
                color: var(--text);
                font-size: 0.92rem;
                line-height: 1.65;
                margin-bottom: 8px;
            }

            .small-card-meta,
            .suggestion-meta {
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.5;
                margin-top: 4px;
            }

            .suggestion-head {
                display: flex;
                justify-content: space-between;
                gap: 8px;
                margin-bottom: 10px;
                align-items: center;
            }

            .suggestion-tag,
            .suggestion-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.78rem;
                font-weight: 700;
            }

            .suggestion-tag {
                background: var(--primary-soft);
                color: var(--primary);
            }

            .suggestion-pill {
                background: #eef3fb;
                color: #40536b;
            }

            .hero-grid {
                display: flex;
                justify-content: space-between;
                gap: 18px;
                align-items: flex-end;
                flex-wrap: wrap;
            }

            .hero-metric {
                min-width: 160px;
                flex: 1;
            }

            .hero-metric .metric-label {
                color: rgba(255, 255, 255, 0.72);
                font-size: 0.82rem;
                margin-bottom: 4px;
            }

            .hero-metric .metric-value {
                color: #ffffff;
                font-size: 1.02rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the top hero area for the portfolio demo."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">城市出行洞察 Agent</div>
            <div class="hero-subtitle">
                面向城市交通治理、出行平台运营和数据产品建设的 AI 数据产品 Demo。
                上传 CSV 后，系统会自动识别字段、完成分析、生成业务洞察、输出产品建议与价值判断，并产出可直接放入作品集的 Markdown 报告。
            </div>
            <div class="tag-row">
                <span class="tag">字段自动识别</span>
                <span class="tag">Agent 工作流</span>
                <span class="tag">产品建议与价值判断</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_value_card(title: str, body: str, meta: str | None = None) -> None:
    """Render a reusable card used by the dashboard."""
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-card-title">{title}</div>
            <div class="small-card-body">{body}</div>
            {f'<div class="small-card-meta">{meta}</div>' if meta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestion_card(category: str, title: str, recommendation: str, value_judgement: str, impact_metrics: list[str], priority: str, effort: str, evidence: str) -> None:
    """Render a product suggestion card."""
    st.markdown(
        f"""
        <div class="suggestion-card">
            <div class="suggestion-head">
                <span class="suggestion-tag">{category}</span>
                <span class="suggestion-pill">{priority} · {effort}</span>
            </div>
            <div class="suggestion-title">{title}</div>
            <div class="suggestion-body">{recommendation}</div>
            <div class="suggestion-meta"><strong>价值判断：</strong>{value_judgement}</div>
            <div class="suggestion-meta"><strong>影响指标：</strong>{'，'.join(impact_metrics)}</div>
            <div class="suggestion-meta"><strong>证据：</strong>{evidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
