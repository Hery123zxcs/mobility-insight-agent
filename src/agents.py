"""Structured agent workflow for the Mobility Insight Agent app."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Fallback when python-dotenv is not installed in the current test environment."""
        return False

from src.data_processing import (
    FieldRoles,
    aggregate_metric,
    apply_role_overrides,
    build_analyzable_metrics,
    build_column_profile,
    clean_data,
    compute_kpis,
    detect_anomalies,
    infer_field_roles,
    metric_aggregation,
)


def safe_value(value: Any) -> Any:
    """Convert pandas and numpy scalar values into JSON-friendly Python values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, pd.DataFrame, pd.Series)) else False:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def dataframe_preview(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    """Return a JSON-friendly preview of a dataframe."""
    preview = df.head(limit).copy()
    for column in preview.select_dtypes(include=["datetime64[ns]"]).columns:
        preview[column] = preview[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return preview.where(pd.notna(preview), None).to_dict(orient="records")


def dataframe_shape(df: pd.DataFrame) -> dict[str, int]:
    """Return the row and column count of a dataframe."""
    return {"rows": int(len(df)), "columns": int(len(df.columns))}


def roles_to_dict(roles: FieldRoles) -> dict[str, Any]:
    """Serialize inferred field roles for display and LLM prompts."""
    return {
        "time_column": roles.time_column,
        "region_column": roles.region_column,
        "numeric_columns": roles.numeric_columns,
        "congestion_column": roles.congestion_column,
        "passenger_column": roles.passenger_column,
        "accident_column": roles.accident_column,
        "column_types": roles.column_types,
    }


@dataclass
class DataProfilerInput:
    """Structured input for data_profiler_agent."""

    source_name: str
    raw_df: pd.DataFrame
    role_overrides: dict[str, str | None] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the data profiler input."""
        return {
            "source_name": self.source_name,
            "raw_shape": dataframe_shape(self.raw_df),
            "columns": list(self.raw_df.columns),
            "role_overrides": self.role_overrides,
        }


@dataclass
class DataProfilerOutput:
    """Structured output from data_profiler_agent."""

    agent_name: str
    roles: FieldRoles
    cleaned_df: pd.DataFrame
    column_profile: pd.DataFrame
    quality_summary: dict[str, Any]
    analyzable_metrics: list[dict[str, Any]]
    notes: list[str]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the data profiler output."""
        return {
            "agent_name": self.agent_name,
            "roles": roles_to_dict(self.roles),
            "cleaned_shape": dataframe_shape(self.cleaned_df),
            "quality_summary": self.quality_summary,
            "analyzable_metrics": self.analyzable_metrics,
            "notes": self.notes,
        }


class DataProfilerAgent:
    """Agent that identifies fields, evaluates data quality, and selects analyzable metrics."""

    name = "data_profiler_agent"

    def run(self, agent_input: DataProfilerInput) -> DataProfilerOutput:
        """Run field inference, data cleaning, and metric profiling."""
        inferred_roles = infer_field_roles(agent_input.raw_df)
        roles = apply_role_overrides(inferred_roles, agent_input.role_overrides)
        cleaned_df, quality_summary = clean_data(agent_input.raw_df, roles)
        column_profile = build_column_profile(cleaned_df, roles)
        analyzable_metrics = build_analyzable_metrics(cleaned_df, roles)
        notes = self._build_notes(roles, quality_summary, analyzable_metrics)
        return DataProfilerOutput(
            agent_name=self.name,
            roles=roles,
            cleaned_df=cleaned_df,
            column_profile=column_profile,
            quality_summary=quality_summary,
            analyzable_metrics=analyzable_metrics,
            notes=notes,
        )

    def _build_notes(
        self,
        roles: FieldRoles,
        quality_summary: dict[str, Any],
        analyzable_metrics: list[dict[str, Any]],
    ) -> list[str]:
        """Build concise profiler notes for portfolio display."""
        notes: list[str] = []
        notes.append(f"识别到 {len(analyzable_metrics)} 个可分析数值指标。")
        if not roles.time_column:
            notes.append("未识别到时间列，趋势分析将受限。")
        if not roles.region_column:
            notes.append("未识别到区域列，区域排名与对比将受限。")
        if quality_summary.get("missing_total", 0) > 0:
            notes.append(f"清洗后仍有 {quality_summary['missing_total']} 个缺失值，建议复核数据采集质量。")
        if quality_summary.get("removed_rows", 0) > 0:
            notes.append(f"清洗过程移除了 {quality_summary['removed_rows']} 条空行或重复记录。")
        return notes


@dataclass
class AnalysisInput:
    """Structured input for analysis_agent."""

    cleaned_df: pd.DataFrame
    roles: FieldRoles
    selected_metrics: list[str]
    primary_metric: str | None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the analysis input."""
        return {
            "cleaned_shape": dataframe_shape(self.cleaned_df),
            "roles": roles_to_dict(self.roles),
            "selected_metrics": self.selected_metrics,
            "primary_metric": self.primary_metric,
        }


@dataclass
class AnalysisOutput:
    """Structured output from analysis_agent."""

    agent_name: str
    kpis: dict[str, Any]
    trend_table: pd.DataFrame
    ranking_table: pd.DataFrame
    anomalies: pd.DataFrame
    correlation_matrix: pd.DataFrame
    correlation_pairs: list[dict[str, Any]]
    metric_summary: list[dict[str, Any]]
    notes: list[str]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the analysis output."""
        return {
            "agent_name": self.agent_name,
            "kpis": {key: safe_value(value) for key, value in self.kpis.items()},
            "trend_preview": dataframe_preview(self.trend_table, 8),
            "ranking_preview": dataframe_preview(self.ranking_table, 8),
            "anomaly_count": int(len(self.anomalies)),
            "correlation_pairs": self.correlation_pairs,
            "metric_summary": self.metric_summary,
            "notes": self.notes,
        }


class AnalysisAgent:
    """Agent that computes trends, rankings, anomalies, and correlations."""

    name = "analysis_agent"

    def run(self, agent_input: AnalysisInput) -> AnalysisOutput:
        """Run quantitative analysis on cleaned mobility data."""
        df = agent_input.cleaned_df
        roles = agent_input.roles
        metrics = [column for column in agent_input.selected_metrics if column in df.columns]
        primary_metric = agent_input.primary_metric if agent_input.primary_metric in df.columns else (metrics[0] if metrics else None)

        anomalies = detect_anomalies(df, roles.numeric_columns)
        kpis = compute_kpis(df, roles, anomalies)
        trend_table = self._build_trend_table(df, roles, metrics)
        ranking_table = self._build_ranking_table(df, roles, primary_metric)
        correlation_matrix = self._build_correlation_matrix(df, roles.numeric_columns)
        correlation_pairs = self._build_correlation_pairs(correlation_matrix)
        metric_summary = self._build_metric_summary(df, roles.numeric_columns)
        notes = self._build_notes(trend_table, ranking_table, anomalies, correlation_pairs)

        return AnalysisOutput(
            agent_name=self.name,
            kpis=kpis,
            trend_table=trend_table,
            ranking_table=ranking_table,
            anomalies=anomalies,
            correlation_matrix=correlation_matrix,
            correlation_pairs=correlation_pairs,
            metric_summary=metric_summary,
            notes=notes,
        )

    def _build_trend_table(self, df: pd.DataFrame, roles: FieldRoles, metrics: list[str]) -> pd.DataFrame:
        """Aggregate selected metrics by daily time buckets."""
        if not roles.time_column or roles.time_column not in df.columns or not metrics:
            return pd.DataFrame()
        trend_df = df.dropna(subset=[roles.time_column]).copy()
        if trend_df.empty:
            return pd.DataFrame()
        trend_df["_time_bucket"] = trend_df[roles.time_column].dt.floor("D")
        return trend_df.groupby("_time_bucket")[metrics].mean().reset_index()

    def _build_ranking_table(self, df: pd.DataFrame, roles: FieldRoles, metric: str | None) -> pd.DataFrame:
        """Rank regions by the primary metric."""
        if not roles.region_column or roles.region_column not in df.columns or not metric or metric not in df.columns:
            return pd.DataFrame()
        agg = metric_aggregation(metric, roles)
        ranking = aggregate_metric(df, roles.region_column, metric, agg).head(20).copy()
        ranking["rank"] = range(1, len(ranking) + 1)
        ranking["aggregation"] = "总量" if agg == "sum" else "均值"
        return ranking

    def _build_correlation_matrix(self, df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
        """Compute a Pearson correlation matrix for numeric metrics."""
        columns = [column for column in numeric_columns if column in df.columns]
        if len(columns) < 2:
            return pd.DataFrame()
        numeric_df = df[columns].select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        return numeric_df.corr().round(3)

    def _build_correlation_pairs(self, correlation_matrix: pd.DataFrame) -> list[dict[str, Any]]:
        """Extract the strongest metric correlation pairs."""
        if correlation_matrix.empty:
            return []
        pairs: list[dict[str, Any]] = []
        columns = list(correlation_matrix.columns)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                value = correlation_matrix.loc[left, right]
                if pd.isna(value):
                    continue
                pairs.append({"metric_a": left, "metric_b": right, "correlation": float(value)})
        pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        return pairs[:8]

    def _build_metric_summary(self, df: pd.DataFrame, numeric_columns: list[str]) -> list[dict[str, Any]]:
        """Compute descriptive statistics for numeric metrics."""
        rows: list[dict[str, Any]] = []
        for column in numeric_columns:
            if column not in df.columns:
                continue
            values = df[column].dropna()
            rows.append(
                {
                    "metric": column,
                    "count": int(values.count()),
                    "mean": None if values.empty else float(values.mean()),
                    "min": None if values.empty else float(values.min()),
                    "max": None if values.empty else float(values.max()),
                    "std": None if values.empty else float(values.std(ddof=0)),
                }
            )
        return rows

    def _build_notes(
        self,
        trend_table: pd.DataFrame,
        ranking_table: pd.DataFrame,
        anomalies: pd.DataFrame,
        correlation_pairs: list[dict[str, Any]],
    ) -> list[str]:
        """Build concise analysis notes for agent output display."""
        notes: list[str] = []
        notes.append("已完成趋势、排名、异常点和相关性计算。")
        if trend_table.empty:
            notes.append("趋势表为空，原因通常是缺少时间列或未选择数值指标。")
        if ranking_table.empty:
            notes.append("排名表为空，原因通常是缺少区域列或主指标。")
        if not anomalies.empty:
            notes.append(f"发现 {len(anomalies)} 条异常指标记录。")
        if correlation_pairs:
            top_pair = correlation_pairs[0]
            notes.append(
                f"最强相关指标为 {top_pair['metric_a']} 与 {top_pair['metric_b']}，相关系数 {top_pair['correlation']:.2f}。"
            )
        return notes


@dataclass
class InsightItem:
    """One structured business insight created by insight_agent."""

    title: str
    evidence: str
    implication: str
    priority: str


@dataclass
class InsightInput:
    """Structured input for insight_agent."""

    profiler_output: DataProfilerOutput
    analysis_output: AnalysisOutput

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the insight input."""
        return {
            "profiler_agent": self.profiler_output.summary(),
            "analysis_agent": self.analysis_output.summary(),
        }


@dataclass
class InsightOutput:
    """Structured output from insight_agent."""

    agent_name: str
    executive_summary: str
    insights: list[InsightItem]
    recommendations: list[str]
    limitations: list[str]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the insight output."""
        return {
            "agent_name": self.agent_name,
            "executive_summary": self.executive_summary,
            "insights": [asdict(item) for item in self.insights],
            "recommendations": self.recommendations,
            "limitations": self.limitations,
        }


class InsightAgent:
    """Agent that translates quantitative outputs into business insights."""

    name = "insight_agent"

    def run(self, agent_input: InsightInput) -> InsightOutput:
        """Create business insights from profiler and analysis outputs."""
        profiler = agent_input.profiler_output
        analysis = agent_input.analysis_output
        insights = self._build_insights(profiler, analysis)
        recommendations = self._build_recommendations(profiler, analysis)
        limitations = self._build_limitations(profiler, analysis)
        executive_summary = self._build_executive_summary(profiler, analysis, insights)
        return InsightOutput(
            agent_name=self.name,
            executive_summary=executive_summary,
            insights=insights,
            recommendations=recommendations,
            limitations=limitations,
        )

    def _build_insights(self, profiler: DataProfilerOutput, analysis: AnalysisOutput) -> list[InsightItem]:
        """Build structured insight cards from analysis evidence."""
        insights: list[InsightItem] = []
        if not analysis.ranking_table.empty:
            top_row = analysis.ranking_table.iloc[0]
            region_column = profiler.roles.region_column
            metric_columns = [column for column in analysis.ranking_table.columns if column not in {"rank", "aggregation", region_column}]
            metric = metric_columns[0] if metric_columns else "主指标"
            insights.append(
                InsightItem(
                    title="重点区域已识别",
                    evidence=f"{top_row[region_column]} 在 {metric} 排名第 1。",
                    implication="该区域适合作为运力调度、拥堵治理或服务保障的优先关注对象。",
                    priority="high",
                )
            )
        if not analysis.anomalies.empty:
            top_anomaly = analysis.anomalies.iloc[0]
            metric = top_anomaly.get("_anomaly_metric", "指标")
            score = top_anomaly.get("_anomaly_score", 0)
            insights.append(
                InsightItem(
                    title="存在显著异常波动",
                    evidence=f"{metric} 出现异常记录，最高异常分数约 {float(score):.2f}。",
                    implication="需要结合节假日、事故、施工、天气或采集问题做复核。",
                    priority="high",
                )
            )
        if analysis.correlation_pairs:
            pair = analysis.correlation_pairs[0]
            insights.append(
                InsightItem(
                    title="指标之间存在联动关系",
                    evidence=f"{pair['metric_a']} 与 {pair['metric_b']} 的相关系数为 {pair['correlation']:.2f}。",
                    implication="可将这组指标作为联合监控对象，提升预警解释力。",
                    priority="medium",
                )
            )
        if analysis.trend_table.shape[0] >= 2:
            insights.append(
                InsightItem(
                    title="趋势分析已形成时间序列基础",
                    evidence=f"趋势表覆盖 {analysis.trend_table.shape[0]} 个时间点。",
                    implication="后续可以接入周期性、峰谷和节假日对比分析。",
                    priority="medium",
                )
            )
        if not insights:
            insights.append(
                InsightItem(
                    title="当前数据可分析性有限",
                    evidence="缺少足够的时间、区域或数值字段。",
                    implication="建议补充标准化字段后再进行运营洞察生成。",
                    priority="medium",
                )
            )
        return insights

    def _build_recommendations(self, profiler: DataProfilerOutput, analysis: AnalysisOutput) -> list[str]:
        """Build action recommendations from insight evidence."""
        recommendations = [
            "优先关注排名靠前或异常分数较高的区域，建立人工复核与运营响应闭环。",
            "将趋势图用于日常监控，观察高峰日期、连续上升和突然回落等变化。",
            "把强相关指标纳入联合看板，避免只根据单一指标判断交通状态。",
        ]
        if profiler.quality_summary.get("missing_total", 0) > 0:
            recommendations.append("对缺失值较多的字段建立数据质量检查规则，减少后续分析偏差。")
        if analysis.anomalies.empty:
            recommendations.append("当前未发现显著异常，可继续扩大时间窗口以提升异常检测稳定性。")
        return recommendations

    def _build_limitations(self, profiler: DataProfilerOutput, analysis: AnalysisOutput) -> list[str]:
        """Build analysis limitations based on missing roles and empty outputs."""
        limitations: list[str] = []
        if not profiler.roles.time_column:
            limitations.append("缺少时间列，无法可靠判断趋势变化。")
        if not profiler.roles.region_column:
            limitations.append("缺少区域列，无法形成区域排名和空间对比。")
        if not profiler.roles.numeric_columns:
            limitations.append("缺少数值指标，无法进行量化分析。")
        if analysis.correlation_matrix.empty:
            limitations.append("数值指标不足或数据波动不足，相关性分析结果有限。")
        return limitations or ["当前样本可支持基础洞察，但仍建议引入天气、节假日、施工等外部变量。"]

    def _build_executive_summary(
        self,
        profiler: DataProfilerOutput,
        analysis: AnalysisOutput,
        insights: list[InsightItem],
    ) -> str:
        """Build a concise executive summary for dashboard display."""
        row_count = analysis.kpis.get("row_count", len(profiler.cleaned_df))
        anomaly_count = analysis.kpis.get("anomaly_count", 0)
        metric_count = len(profiler.analyzable_metrics)
        return f"本次工作流分析了 {row_count} 条记录、{metric_count} 个数值指标，形成 {len(insights)} 条业务洞察，并识别出 {anomaly_count} 条异常指标记录。"


@dataclass
class ProductSuggestion:
    """One product suggestion with portfolio-ready value judgement."""

    category: str
    title: str
    recommendation: str
    evidence: str
    value_judgement: str
    impact_metrics: list[str]
    priority: str
    effort: str


@dataclass
class ProductSuggestionInput:
    """Structured input for product_suggestion_agent."""

    profiler_output: DataProfilerOutput
    analysis_output: AnalysisOutput
    insight_output: InsightOutput

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the product suggestion input."""
        return {
            "profiler_agent": self.profiler_output.summary(),
            "analysis_agent": self.analysis_output.summary(),
            "insight_agent": self.insight_output.summary(),
        }


@dataclass
class ProductSuggestionOutput:
    """Structured output from product_suggestion_agent."""

    agent_name: str
    suggestions: list[ProductSuggestion]
    value_judgements: list[dict[str, Any]]
    resume_project_description: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the product suggestion output."""
        return {
            "agent_name": self.agent_name,
            "suggestions": [asdict(item) for item in self.suggestions],
            "value_judgements": self.value_judgements,
            "resume_project_description": self.resume_project_description,
        }

    def grouped_suggestions(self) -> dict[str, list[ProductSuggestion]]:
        """Group suggestions by product category for display."""
        grouped: dict[str, list[ProductSuggestion]] = {}
        for suggestion in self.suggestions:
            grouped.setdefault(suggestion.category, []).append(suggestion)
        return grouped


class ProductSuggestionAgent:
    """Agent that converts insights into PM-style product suggestions and value judgement."""

    name = "product_suggestion_agent"

    def run(self, agent_input: ProductSuggestionInput) -> ProductSuggestionOutput:
        """Generate product suggestions across governance, operations, and data product areas."""
        profiler = agent_input.profiler_output
        analysis = agent_input.analysis_output
        insight = agent_input.insight_output
        suggestions = self._build_suggestions(profiler, analysis)
        value_judgements = [self._build_value_judgement(item) for item in suggestions]
        resume_description = self._build_resume_description(profiler, analysis, insight, suggestions)
        return ProductSuggestionOutput(
            agent_name=self.name,
            suggestions=suggestions,
            value_judgements=value_judgements,
            resume_project_description=resume_description,
        )

    def _top_region_evidence(self, profiler: DataProfilerOutput, analysis: AnalysisOutput) -> str:
        """Build evidence text for the top-ranked region when available."""
        if analysis.ranking_table.empty or not profiler.roles.region_column:
            return "当前数据未形成稳定区域排名，建议先完善区域字段。"
        top_row = analysis.ranking_table.iloc[0]
        region = top_row[profiler.roles.region_column]
        metric_columns = [column for column in analysis.ranking_table.columns if column not in {"rank", "aggregation", profiler.roles.region_column}]
        metric = metric_columns[0] if metric_columns else "主指标"
        return f"{region} 在 `{metric}` 上排名第 1，可作为治理和运营优先对象。"

    def _anomaly_evidence(self, analysis: AnalysisOutput) -> str:
        """Build evidence text for anomaly-related suggestions."""
        anomaly_count = len(analysis.anomalies)
        if anomaly_count == 0:
            return "当前样本未发现显著异常，但仍可保留异常监控能力用于持续运营。"
        return f"analysis_agent 检测到 {anomaly_count} 条异常指标记录，可用于触发复核和响应流程。"

    def _build_suggestions(self, profiler: DataProfilerOutput, analysis: AnalysisOutput) -> list[ProductSuggestion]:
        """Build the three requested product suggestion categories."""
        top_region_evidence = self._top_region_evidence(profiler, analysis)
        anomaly_evidence = self._anomaly_evidence(analysis)
        report_evidence = f"report_agent 已能自动生成 Markdown 报告，当前分析覆盖 {analysis.kpis.get('row_count', 0)} 条记录。"
        quality_evidence = f"data_profiler_agent 输出 {len(profiler.analyzable_metrics)} 个可分析指标，缺失值总量为 {profiler.quality_summary.get('missing_total', 0)}。"

        return [
            ProductSuggestion(
                category="交通治理建议",
                title="建立高风险区域分级治理清单",
                recommendation="将区域排名、异常点和拥堵相关指标合并为治理优先级，对排名靠前区域制定分时段治理策略。",
                evidence=top_region_evidence,
                value_judgement="能把原本分散的数据结果转化为可执行的治理清单，帮助交通管理方优先处理高影响区域。",
                impact_metrics=["拥堵时长", "高拥堵区域响应时间", "治理任务闭环率"],
                priority="high",
                effort="medium",
            ),
            ProductSuggestion(
                category="交通治理建议",
                title="建设异常事件复核与派单流程",
                recommendation="当异常点超过阈值时自动生成事件卡片，标注时间、区域、指标和异常分数，并进入人工复核或派单流程。",
                evidence=anomaly_evidence,
                value_judgement="将异常检测从事后看报表推进到近实时响应，提升事件识别和跨部门协作效率。",
                impact_metrics=["异常事件识别效率", "事件响应时长", "人工复核准确率"],
                priority="high",
                effort="medium",
            ),
            ProductSuggestion(
                category="出行平台运营建议",
                title="面向高峰区域做供需调度和用户引导",
                recommendation="基于客流、拥堵和区域排名结果，向高峰区域提前配置运力，并对用户提供绕行、错峰或替代线路提示。",
                evidence=top_region_evidence,
                value_judgement="把洞察结果连接到平台运营动作，可降低高峰体验波动并提升供需匹配效率。",
                impact_metrics=["用户等待时长", "高峰客流承载率", "运营决策效率"],
                priority="high",
                effort="high",
            ),
            ProductSuggestion(
                category="出行平台运营建议",
                title="自动生成运营日报和异常摘要",
                recommendation="将 report_agent 生成的 Markdown 报告作为运营日报基础，支持按区域、指标和异常等级订阅。",
                evidence=report_evidence,
                value_judgement="减少人工整理数据和撰写报告的时间，让运营人员更快进入判断和行动。",
                impact_metrics=["报告生成时间", "运营决策效率", "跨团队同步成本"],
                priority="medium",
                effort="low",
            ),
            ProductSuggestion(
                category="数据产品优化建议",
                title="沉淀字段语义识别与数据质量规则",
                recommendation="把 data_profiler_agent 的字段识别、缺失值统计和可分析指标输出做成数据接入校验能力。",
                evidence=quality_evidence,
                value_judgement="提升数据接入的一致性，降低每次分析前手工判断字段和清洗数据的成本。",
                impact_metrics=["字段识别准确率", "数据质量问题发现时间", "分析准备时间"],
                priority="medium",
                effort="medium",
            ),
            ProductSuggestion(
                category="数据产品优化建议",
                title="引入外部变量增强洞察解释力",
                recommendation="接入天气、节假日、施工、重大活动等外部变量，和现有趋势、异常、相关性结果联合分析。",
                evidence="insight_agent 已输出分析限制，当前样本仍缺少外部解释变量。",
                value_judgement="能减少只看到异常但无法解释原因的问题，提高报告可信度和后续策略命中率。",
                impact_metrics=["异常事件识别效率", "洞察解释率", "策略命中率"],
                priority="medium",
                effort="high",
            ),
        ]

    def _build_value_judgement(self, suggestion: ProductSuggestion) -> dict[str, Any]:
        """Build a concise value judgement row for one suggestion."""
        return {
            "建议": suggestion.title,
            "类别": suggestion.category,
            "价值判断": suggestion.value_judgement,
            "影响指标": suggestion.impact_metrics,
            "优先级": suggestion.priority,
            "实施复杂度": suggestion.effort,
        }

    def _build_resume_description(
        self,
        profiler: DataProfilerOutput,
        analysis: AnalysisOutput,
        insight: InsightOutput,
        suggestions: list[ProductSuggestion],
    ) -> str:
        """Generate a resume-ready project description."""
        metric_count = len(profiler.analyzable_metrics)
        anomaly_count = analysis.kpis.get("anomaly_count", 0)
        suggestion_count = len(suggestions)
        return (
            "城市出行洞察 Agent：独立设计并实现面向城市交通数据的 Streamlit 数据产品，"
            "构建 data_profiler、analysis、insight、product_suggestion、report 多 Agent 工作流，"
            f"支持 CSV 上传、字段自动识别、数据质量诊断、{metric_count} 类可分析指标提取、趋势/排名/异常/相关性分析，"
            f"并将 {anomaly_count} 条异常结果和业务洞察转化为 {suggestion_count} 条产品建议、价值判断和 Markdown 报告，"
            "用于提升交通治理响应、出行平台运营决策和数据分析报告生成效率。"
        )


@dataclass
class ReportInput:
    """Structured input for report_agent."""

    profiler_output: DataProfilerOutput
    analysis_output: AnalysisOutput
    insight_output: InsightOutput
    product_suggestion_output: ProductSuggestionOutput | None = None
    use_llm: bool = False

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the report input."""
        summary = {
            "use_llm": self.use_llm,
            "profiler_agent": self.profiler_output.summary(),
            "analysis_agent": self.analysis_output.summary(),
            "insight_agent": self.insight_output.summary(),
        }
        if self.product_suggestion_output:
            summary["product_suggestion_agent"] = self.product_suggestion_output.summary()
        return summary


@dataclass
class ReportOutput:
    """Structured output from report_agent."""

    agent_name: str
    markdown: str
    source: str
    error: str | None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the report output."""
        return {
            "agent_name": self.agent_name,
            "source": self.source,
            "error": self.error,
            "markdown_length": len(self.markdown),
        }


class ReportAgent:
    """Agent that generates a Markdown report from structured workflow outputs."""

    name = "report_agent"

    def run(self, agent_input: ReportInput) -> ReportOutput:
        """Generate a Markdown report with an LLM when available, otherwise locally."""
        load_dotenv()
        payload = agent_input.summary()
        if agent_input.use_llm and os.getenv("OPENAI_API_KEY"):
            try:
                return ReportOutput(
                    agent_name=self.name,
                    markdown=self._call_llm(payload),
                    source="LLM",
                    error=None,
                )
            except Exception as exc:
                return ReportOutput(
                    agent_name=self.name,
                    markdown=self._build_local_report(agent_input),
                    source="local_fallback",
                    error=str(exc),
                )
        return ReportOutput(
            agent_name=self.name,
            markdown=self._build_local_report(agent_input),
            source="local_rules",
            error=None,
        )

    def _call_llm(self, payload: dict[str, Any]) -> str:
        """Call an OpenAI-compatible chat model for report generation."""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        prompt = (
            "你是城市交通运营分析专家。请基于以下结构化 Agent 工作流输出，生成 Markdown 中文报告。"
            "报告需包含：数据概览、Agent 工作流摘要、趋势与排名、异常与相关性、业务洞察、行动建议、分析限制。"
            "不要编造结构化数据里没有的事实。\n\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你擅长将交通数据分析结果转化为清晰、可执行的运营报告。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def _build_local_report(self, agent_input: ReportInput) -> str:
        """Build a deterministic Markdown report from structured agent outputs."""
        profiler = agent_input.profiler_output
        analysis = agent_input.analysis_output
        insights = agent_input.insight_output
        product_suggestions = agent_input.product_suggestion_output
        roles = profiler.roles

        lines = [
            "# 城市出行洞察报告",
            "",
            "## 1. Agent 工作流摘要",
            f"- data_profiler_agent：识别字段角色、完成数据清洗，得到 {len(profiler.analyzable_metrics)} 个可分析指标。",
            f"- analysis_agent：完成趋势、排名、异常点和相关性分析，发现 {analysis.kpis.get('anomaly_count', 0)} 条异常指标记录。",
            f"- insight_agent：生成 {len(insights.insights)} 条业务洞察和 {len(insights.recommendations)} 条行动建议。",
            f"- product_suggestion_agent：生成 {len(product_suggestions.suggestions) if product_suggestions else 0} 条产品建议与价值判断。",
            "- report_agent：汇总结构化输出并生成 Markdown 报告。",
            "",
            "## 2. 数据概览",
            f"- 数据规模：{analysis.kpis.get('row_count', 0)} 条记录，{analysis.kpis.get('column_count', 0)} 个字段。",
            f"- 时间列：`{roles.time_column or '未识别'}`；区域列：`{roles.region_column or '未识别'}`。",
            f"- 数值指标：{', '.join(roles.numeric_columns) if roles.numeric_columns else '未识别'}。",
        ]
        if analysis.kpis.get("time_range"):
            lines.append(f"- 时间范围：{analysis.kpis['time_range']}。")
        if analysis.kpis.get("region_count") is not None:
            lines.append(f"- 覆盖区域数量：{analysis.kpis['region_count']}。")

        lines.extend(["", "## 3. 关键分析结果"])
        if not analysis.ranking_table.empty:
            top_row = analysis.ranking_table.iloc[0]
            region_column = roles.region_column
            metric_columns = [column for column in analysis.ranking_table.columns if column not in {"rank", "aggregation", region_column}]
            metric = metric_columns[0] if metric_columns else "主指标"
            lines.append(f"- 区域排名第一：{top_row[region_column]}，指标 `{metric}` 表现最高。")
        else:
            lines.append("- 区域排名未生成，可能缺少区域列或主指标。")
        if analysis.correlation_pairs:
            pair = analysis.correlation_pairs[0]
            lines.append(f"- 最强相关关系：`{pair['metric_a']}` 与 `{pair['metric_b']}`，相关系数 {pair['correlation']:.2f}。")
        else:
            lines.append("- 相关性结果不足，可能是数值指标少于两个或样本波动不足。")
        if not analysis.anomalies.empty:
            lines.append(f"- 异常点：共检测到 {len(analysis.anomalies)} 条异常指标记录。")
        else:
            lines.append("- 异常点：未检测到显著异常。")

        lines.extend(["", "## 4. 业务洞察"])
        lines.append(insights.executive_summary)
        for item in insights.insights:
            lines.append(f"- **{item.title}**：{item.evidence} {item.implication}")

        lines.extend(["", "## 5. 行动建议"])
        for recommendation in insights.recommendations:
            lines.append(f"- {recommendation}")

        if product_suggestions:
            lines.extend(["", "## 6. 产品建议与价值判断"])
            for category, suggestions in product_suggestions.grouped_suggestions().items():
                lines.append(f"### {category}")
                for suggestion in suggestions:
                    metrics = "、".join(suggestion.impact_metrics)
                    lines.append(f"- **{suggestion.title}**：{suggestion.recommendation}")
                    lines.append(f"  - 价值判断：{suggestion.value_judgement}")
                    lines.append(f"  - 影响指标：{metrics}")
                    lines.append(f"  - 优先级：{suggestion.priority}；实施复杂度：{suggestion.effort}")

            lines.extend(["", "## 7. 简历项目描述"])
            lines.append(product_suggestions.resume_project_description)

        lines.extend(["", "## 8. 分析限制" if product_suggestions else "## 6. 分析限制"])
        for limitation in insights.limitations:
            lines.append(f"- {limitation}")
        return "\n".join(lines)
