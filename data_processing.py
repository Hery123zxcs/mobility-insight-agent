"""Data cleaning, field inference, and reusable metrics for mobility datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


TIME_KEYWORDS = [
    "time",
    "date",
    "datetime",
    "timestamp",
    "day",
    "month",
    "hour",
    "日期",
    "时间",
    "月份",
    "小时",
    "采集",
]
REGION_KEYWORDS = [
    "region",
    "area",
    "district",
    "zone",
    "city",
    "road",
    "station",
    "line",
    "行政区",
    "区域",
    "城区",
    "街道",
    "路段",
    "站点",
    "线路",
]
CONGESTION_KEYWORDS = ["congestion", "jam", "delay", "traffic", "拥堵", "延误", "车速", "速度"]
PASSENGER_KEYWORDS = ["passenger", "flow", "ridership", "volume", "客流", "人流", "流量", "出行量", "订单"]
ACCIDENT_KEYWORDS = ["accident", "incident", "crash", "event", "事故", "事件", "警情"]


@dataclass
class FieldRoles:
    """Store inferred semantic roles for a mobility dataset."""

    time_column: str | None
    region_column: str | None
    numeric_columns: list[str]
    congestion_column: str | None
    passenger_column: str | None
    accident_column: str | None
    column_types: dict[str, str]


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name before keyword matching."""
    return str(column_name).strip().lower().replace(" ", "_")


def keyword_score(column_name: str, keywords: list[str]) -> int:
    """Count how many role-specific keywords appear in a column name."""
    normalized = normalize_column_name(column_name)
    return sum(1 for keyword in keywords if keyword.lower() in normalized)


def numeric_parse_ratio(series: pd.Series) -> float:
    """Estimate the share of non-null values that can be parsed as numbers."""
    if series.empty:
        return 0.0
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    non_null = series.notna().sum()
    return 0.0 if non_null == 0 else float(parsed.notna().sum() / non_null)


def datetime_parse_ratio(series: pd.Series) -> float:
    """Estimate the share of non-null values that can be parsed as datetimes."""
    if series.empty:
        return 0.0
    parsed = pd.to_datetime(series, errors="coerce")
    non_null = series.notna().sum()
    return 0.0 if non_null == 0 else float(parsed.notna().sum() / non_null)


def classify_column(series: pd.Series) -> str:
    """Classify a column as time, numeric, categorical, or text."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "time"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if datetime_parse_ratio(series) >= 0.75:
        return "time"
    if numeric_parse_ratio(series) >= 0.75:
        return "numeric"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return "categorical" if unique_ratio <= 0.6 else "text"


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer a compact type label for every column in a dataframe."""
    return {column: classify_column(df[column]) for column in df.columns}


def choose_best_column(
    df: pd.DataFrame,
    candidates: list[str],
    keywords: list[str],
    prefer_lower_cardinality: bool = False,
) -> str | None:
    """Choose the best candidate column with keyword and data-shape evidence."""
    if not candidates:
        return None

    scored: list[tuple[float, str]] = []
    for column in candidates:
        score = float(keyword_score(column, keywords) * 10)
        if prefer_lower_cardinality:
            unique_ratio = df[column].nunique(dropna=True) / max(len(df), 1)
            score += max(0.0, 5.0 - unique_ratio * 10.0)
        else:
            score += datetime_parse_ratio(df[column]) if keywords == TIME_KEYWORDS else numeric_parse_ratio(df[column])
        scored.append((score, column))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def choose_metric_role_column(
    df: pd.DataFrame,
    numeric_columns: list[str],
    keywords: list[str],
    excluded_columns: set[str] | None = None,
) -> str | None:
    """Choose a semantic metric column only when its name supports that role."""
    excluded_columns = excluded_columns or set()
    scored: list[tuple[int, str]] = []
    for column in numeric_columns:
        if column in excluded_columns:
            continue
        score = keyword_score(column, keywords)
        if score > 0:
            scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], numeric_parse_ratio(df[item[1]])), reverse=True)
    return scored[0][1]


def infer_field_roles(df: pd.DataFrame) -> FieldRoles:
    """Infer time, region, numeric, and mobility-specific metric roles."""
    column_types = infer_column_types(df)
    time_candidates = [
        column
        for column, column_type in column_types.items()
        if column_type == "time" or keyword_score(column, TIME_KEYWORDS) > 0
    ]
    numeric_columns = [column for column, column_type in column_types.items() if column_type == "numeric"]
    region_candidates = [
        column
        for column, column_type in column_types.items()
        if column_type in {"categorical", "text"}
        and column not in time_candidates
        and df[column].nunique(dropna=True) > 1
    ]

    time_column = choose_best_column(df, time_candidates, TIME_KEYWORDS)
    region_column = choose_best_column(df, region_candidates, REGION_KEYWORDS, prefer_lower_cardinality=True)
    congestion_column = choose_metric_role_column(df, numeric_columns, CONGESTION_KEYWORDS)
    passenger_column = choose_metric_role_column(
        df,
        numeric_columns,
        PASSENGER_KEYWORDS,
        {congestion_column} if congestion_column else set(),
    )
    excluded_metric_columns = {column for column in [congestion_column, passenger_column] if column}
    accident_column = choose_metric_role_column(df, numeric_columns, ACCIDENT_KEYWORDS, excluded_metric_columns)

    return FieldRoles(
        time_column=time_column,
        region_column=region_column,
        numeric_columns=numeric_columns,
        congestion_column=congestion_column,
        passenger_column=passenger_column,
        accident_column=accident_column,
        column_types=column_types,
    )


def apply_role_overrides(roles: FieldRoles, overrides: dict[str, str | None]) -> FieldRoles:
    """Apply user-selected role overrides while preserving inferred metadata."""
    numeric_columns = [column for column in roles.numeric_columns if column]
    return FieldRoles(
        time_column=overrides.get("time_column", roles.time_column),
        region_column=overrides.get("region_column", roles.region_column),
        numeric_columns=numeric_columns,
        congestion_column=overrides.get("congestion_column", roles.congestion_column),
        passenger_column=overrides.get("passenger_column", roles.passenger_column),
        accident_column=overrides.get("accident_column", roles.accident_column),
        column_types=roles.column_types,
    )


def convert_numeric_series(series: pd.Series) -> pd.Series:
    """Convert values to numbers while tolerating commas, spaces, and percent signs."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_data(df: pd.DataFrame, roles: FieldRoles) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean uploaded data and return a structured quality summary."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    before_rows = len(cleaned)
    before_columns = len(cleaned.columns)
    cleaned = cleaned.dropna(how="all").drop_duplicates()

    if roles.time_column in cleaned.columns:
        cleaned[roles.time_column] = pd.to_datetime(cleaned[roles.time_column], errors="coerce")

    for column in roles.numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = convert_numeric_series(cleaned[column])

    missing_by_column = {column: int(value) for column, value in cleaned.isna().sum().to_dict().items()}
    quality_summary = {
        "original_rows": before_rows,
        "cleaned_rows": len(cleaned),
        "removed_rows": before_rows - len(cleaned),
        "column_count": before_columns,
        "missing_by_column": missing_by_column,
        "missing_total": int(cleaned.isna().sum().sum()),
        "duplicate_rows_removed": before_rows - len(cleaned),
    }
    return cleaned, quality_summary


def metric_aggregation(metric_column: str | None, roles: FieldRoles) -> str:
    """Choose a default aggregation method for a metric role."""
    if metric_column in {roles.passenger_column, roles.accident_column}:
        return "sum"
    return "mean"


def aggregate_metric(df: pd.DataFrame, group_column: str, metric_column: str, agg: str) -> pd.DataFrame:
    """Aggregate one metric by one grouping column."""
    if agg == "sum":
        values = df.groupby(group_column, dropna=True)[metric_column].sum().reset_index()
    else:
        values = df.groupby(group_column, dropna=True)[metric_column].mean().reset_index()
    return values.sort_values(metric_column, ascending=False)


def detect_anomalies(df: pd.DataFrame, numeric_columns: list[str], z_threshold: float = 2.5) -> pd.DataFrame:
    """Detect numeric outliers with a simple z-score rule."""
    anomaly_frames: list[pd.DataFrame] = []
    for column in numeric_columns:
        if column not in df.columns:
            continue
        values = df[column].dropna()
        if values.empty or values.std(ddof=0) == 0:
            continue
        z_scores = (df[column] - values.mean()) / values.std(ddof=0)
        matched = df.loc[z_scores.abs() >= z_threshold].copy()
        if matched.empty:
            continue
        matched["_source_index"] = matched.index
        matched["_anomaly_metric"] = column
        matched["_anomaly_score"] = z_scores.loc[matched.index].abs()
        anomaly_frames.append(matched)

    if not anomaly_frames:
        return pd.DataFrame(columns=list(df.columns) + ["_source_index", "_anomaly_metric", "_anomaly_score"])
    return pd.concat(anomaly_frames, ignore_index=True).sort_values("_anomaly_score", ascending=False)


def compute_kpis(df: pd.DataFrame, roles: FieldRoles, anomalies: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level dashboard KPIs from cleaned data."""
    kpis: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "region_count": df[roles.region_column].nunique(dropna=True) if roles.region_column else None,
        "anomaly_count": len(anomalies),
    }

    if roles.time_column and roles.time_column in df.columns:
        valid_times = df[roles.time_column].dropna()
        if not valid_times.empty:
            kpis["time_range"] = f"{valid_times.min().date()} 至 {valid_times.max().date()}"

    metric_map = {
        "avg_congestion": roles.congestion_column,
        "total_passenger_flow": roles.passenger_column,
        "total_accidents": roles.accident_column,
    }
    for label, column in metric_map.items():
        if column and column in df.columns:
            agg = metric_aggregation(column, roles)
            kpis[label] = df[column].sum() if agg == "sum" else df[column].mean()

    return kpis


def build_column_profile(df: pd.DataFrame, roles: FieldRoles) -> pd.DataFrame:
    """Build a field profile table for preview and agent output display."""
    role_by_column: dict[str, list[str]] = {column: [] for column in df.columns}
    role_map = {
        roles.time_column: "时间列",
        roles.region_column: "区域列",
        roles.congestion_column: "拥堵指标",
        roles.passenger_column: "客流指标",
        roles.accident_column: "事故指标",
    }
    for column, role_name in role_map.items():
        if column in role_by_column:
            role_by_column[column].append(role_name)
    for column in roles.numeric_columns:
        if column in role_by_column and "数值列" not in role_by_column[column]:
            role_by_column[column].append("数值列")

    rows = []
    for column in df.columns:
        rows.append(
            {
                "field_name": column,
                "field_type": roles.column_types.get(column, "unknown"),
                "analysis_role": " / ".join(role_by_column[column]) if role_by_column[column] else "-",
                "is_analyzable": bool(column in roles.numeric_columns or column == roles.time_column or column == roles.region_column),
                "recommended_use": "趋势分析 / 高峰识别" if column == roles.time_column else "区域对比 / 排名分析" if column == roles.region_column else "通用指标分析" if column in roles.numeric_columns else "基础字段查看 / 数据理解",
            }
        )
    return pd.DataFrame(rows)


def build_analyzable_metrics(df: pd.DataFrame, roles: FieldRoles) -> list[dict[str, Any]]:
    """Describe numeric metrics that can be used by downstream analysis agents."""
    metrics: list[dict[str, Any]] = []
    semantic_roles = {
        roles.congestion_column: ("拥堵指数", "拥堵分析 / 预警"),
        roles.passenger_column: ("客流量", "流量分析 / 运营监控"),
        roles.accident_column: ("事故数", "安全分析 / 风险监控"),
    }
    for column in roles.numeric_columns:
        if column not in df.columns:
            continue
        metric_name, analysis_use = semantic_roles.get(column, (column, "通用指标分析"))
        metrics.append(
            {
                "field_name": column,
                "metric_name": metric_name,
                "metric_type": roles.column_types.get(column, "numeric"),
                "analysis_use": analysis_use,
                "aggregation": metric_aggregation(column, roles),
                "non_null_count": int(df[column].notna().sum()),
                "missing_count": int(df[column].isna().sum()),
                "mean": None if df[column].dropna().empty else float(df[column].mean()),
                "min": None if df[column].dropna().empty else float(df[column].min()),
                "max": None if df[column].dropna().empty else float(df[column].max()),
            }
        )
    return metrics


def build_time_feature_frame(df: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    """Create reusable time features for peak and weekday/weekend analysis."""
    if not time_column or time_column not in df.columns:
        return pd.DataFrame()
    time_df = df.dropna(subset=[time_column]).copy()
    if time_df.empty:
        return pd.DataFrame()
    time_df["_date"] = time_df[time_column].dt.date
    time_df["_weekday"] = time_df[time_column].dt.day_name()
    time_df["_is_weekend"] = time_df[time_column].dt.dayofweek >= 5
    time_df["_hour"] = time_df[time_column].dt.hour
    time_df["_time_band"] = pd.cut(
        time_df["_hour"],
        bins=[-1, 5, 9, 12, 17, 21, 24],
        labels=["凌晨", "早高峰", "上午", "下午", "晚高峰", "夜间"],
        include_lowest=True,
    )
    return time_df


def summarize_peak_periods(df: pd.DataFrame, roles: FieldRoles, metric_column: str | None) -> list[dict[str, Any]]:
    """Summarize peak time bands for one metric."""
    if not metric_column or metric_column not in df.columns:
        return []
    time_df = build_time_feature_frame(df, roles.time_column)
    if time_df.empty or "_time_band" not in time_df.columns:
        return []
    grouped = (
        time_df.groupby("_time_band", dropna=True)[metric_column]
        .mean()
        .reset_index()
        .sort_values(metric_column, ascending=False)
    )
    if grouped.empty:
        return []
    top_band = grouped.iloc[0]["_time_band"]
    top_value = grouped.iloc[0][metric_column]
    return [
        {
            "time_band": str(row["_time_band"]),
            "value": float(row[metric_column]),
            "is_peak": bool(row["_time_band"] == top_band),
        }
        for _, row in grouped.head(6).iterrows()
    ]


def compare_weekday_weekend(df: pd.DataFrame, roles: FieldRoles, metric_column: str | None) -> dict[str, Any]:
    """Compare weekday and weekend performance for one metric."""
    if not metric_column or metric_column not in df.columns:
        return {}
    time_df = build_time_feature_frame(df, roles.time_column)
    if time_df.empty:
        return {}
    grouped = time_df.groupby("_is_weekend")[metric_column].mean()
    if grouped.empty:
        return {}
    weekday_value = float(grouped.get(False, float("nan")))
    weekend_value = float(grouped.get(True, float("nan")))
    diff = weekend_value - weekday_value
    base = weekday_value if weekday_value not in (0, None) else 0.0
    pct = None if base == 0 else diff / base
    return {
        "weekday_avg": weekday_value,
        "weekend_avg": weekend_value,
        "difference": diff,
        "difference_pct": pct,
    }


def build_metric_change_explanations(df: pd.DataFrame, roles: FieldRoles, metric_columns: list[str]) -> list[dict[str, Any]]:
    """Build natural-language style change explanations for the top metrics."""
    time_df = build_time_feature_frame(df, roles.time_column)
    if time_df.empty:
        return []
    explanations: list[dict[str, Any]] = []
    top_metrics = [column for column in metric_columns if column in time_df.columns][:3]
    for metric in top_metrics:
        trend_by_day = time_df.groupby("_date")[metric].mean().sort_index()
        if trend_by_day.shape[0] < 2:
            continue
        first_value = float(trend_by_day.iloc[0])
        last_value = float(trend_by_day.iloc[-1])
        delta = last_value - first_value
        direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        explanations.append(
            {
                "metric": metric,
                "start_value": first_value,
                "end_value": last_value,
                "change": delta,
                "direction": direction,
                "explanation": f"{metric} 从 {first_value:.2f} 变化到 {last_value:.2f}，整体{direction} {abs(delta):.2f}。",
            }
        )
    return explanations
