"""Rule-based anomaly detection for rainfall monitoring records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import station_series


@dataclass(frozen=True)
class Anomaly:
    """Detected anomaly instance."""

    station_id: str
    anomaly_type: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_days: float
    magnitude: float | None = None
    note: str = ""


def _duration_days(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((end - start).total_seconds() / 86400.0, 0.0)


def detect_continuous_fixed_value(
    series: pd.DataFrame,
    fixed_values: Iterable[float] = (0.5, 1.0, 1.5),
    min_run_length: int = 50,
    merge_gap_hours: float = 1.0,
) -> list[Anomaly]:
    """Detect extended runs of fixed tipping-bucket values."""
    if series.empty:
        return []
    station_id = str(series["station_id"].iloc[0])
    fixed = set(float(v) for v in fixed_values)
    values = series["rainfall"].to_numpy(float)
    times = pd.to_datetime(series["timestamp"]).to_list()
    is_fixed = np.array([v in fixed for v in values])

    raw: list[tuple[int, int]] = []
    i = 0
    while i < len(is_fixed):
        if not is_fixed[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(is_fixed) and is_fixed[j + 1]:
            j += 1
        if j - i + 1 >= min_run_length:
            raw.append((i, j))
        i = j + 1

    if not raw:
        return []

    merged = [raw[0]]
    for start, end in raw[1:]:
        last_start, last_end = merged[-1]
        gap_hours = (times[start] - times[last_end]).total_seconds() / 3600.0
        if gap_hours <= merge_gap_hours:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    anomalies = []
    for start, end in merged:
        anomalies.append(
            Anomaly(
                station_id=station_id,
                anomaly_type="CFVA",
                start_time=times[start],
                end_time=times[end],
                duration_days=_duration_days(times[start], times[end]),
                magnitude=float(np.nanmedian(values[start : end + 1])),
                note="continuous fixed-value block",
            )
        )
    return anomalies


def detect_missing_data(
    series: pd.DataFrame,
    study_start: pd.Timestamp,
    study_end: pd.Timestamp,
    threshold_days: float = 45.0,
) -> list[Anomaly]:
    """Detect long observation gaps within or at the boundaries of a study period."""
    station_id = str(series["station_id"].iloc[0]) if not series.empty else ""
    times = pd.to_datetime(series["timestamp"]).sort_values().to_list()
    anomalies: list[Anomaly] = []

    if not times:
        return [
            Anomaly(
                station_id=station_id,
                anomaly_type="MDA",
                start_time=study_start,
                end_time=study_end,
                duration_days=_duration_days(study_start, study_end),
                note="complete missing period",
            )
        ]

    pairs = [(study_start, times[0], "start-of-record missing")]
    pairs.extend((times[i], times[i + 1], "internal missing") for i in range(len(times) - 1))
    pairs.append((times[-1], study_end, "end-of-record missing"))

    for start, end, note in pairs:
        gap = _duration_days(start, end)
        if gap > threshold_days:
            anomalies.append(
                Anomaly(
                    station_id=station_id,
                    anomaly_type="MDA",
                    start_time=start,
                    end_time=end,
                    duration_days=gap,
                    note=note,
                )
            )
    return anomalies


def detect_isolated_point(series: pd.DataFrame, z_threshold: float = 3.0, window_hours: float = 5.0) -> list[Anomaly]:
    """Detect isolated extreme values without temporal support."""
    if len(series) < 5:
        return []
    station_id = str(series["station_id"].iloc[0])
    times = pd.to_datetime(series["timestamp"]).to_numpy()
    values = series["rainfall"].to_numpy(float)
    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    if sigma <= 0 or not np.isfinite(sigma):
        return []

    anomalies: list[Anomaly] = []
    for idx, value in enumerate(values):
        if abs(value - mu) <= z_threshold * sigma:
            continue
        delta_hours = np.abs((times - times[idx]) / np.timedelta64(1, "h"))
        neighbours = values[(delta_hours > 0) & (delta_hours <= window_hours)]
        if len(neighbours) == 0 or np.nanmax(neighbours) < 0.5 * value:
            timestamp = pd.Timestamp(times[idx])
            anomalies.append(
                Anomaly(
                    station_id=station_id,
                    anomaly_type="IPA",
                    start_time=timestamp,
                    end_time=timestamp,
                    duration_days=0.0,
                    magnitude=float(value),
                    note="isolated extreme point",
                )
            )
    return anomalies


def detect_no_trend_anomaly(series: pd.DataFrame, peak_ratio: float = 2.0) -> list[Anomaly]:
    """Detect brief peaks without an expected rise-recession pattern."""
    if len(series) < 5:
        return []
    station_id = str(series["station_id"].iloc[0])
    values = series["rainfall"].to_numpy(float)
    times = pd.to_datetime(series["timestamp"]).to_list()
    anomalies: list[Anomaly] = []

    for idx in range(2, len(values) - 1):
        prev_mean = np.nanmean(values[idx - 2 : idx])
        next_value = values[idx + 1]
        if prev_mean <= 0:
            continue
        if values[idx] >= peak_ratio * prev_mean and next_value <= 0.5 * values[idx]:
            anomalies.append(
                Anomaly(
                    station_id=station_id,
                    anomaly_type="NTA",
                    start_time=times[idx],
                    end_time=times[idx],
                    duration_days=0.0,
                    magnitude=float(values[idx]),
                    note="peak without trend context",
                )
            )
    return anomalies


def detect_anomalies(
    records: pd.DataFrame,
    station_id: str,
    study_start: str | pd.Timestamp,
    study_end: str | pd.Timestamp,
) -> list[Anomaly]:
    """Run the four anomaly detectors for one station."""
    series = station_series(records, station_id=station_id, nonzero_only=True)
    if series.empty:
        series = pd.DataFrame({"station_id": [str(station_id)], "timestamp": [], "rainfall": []})
    start = pd.Timestamp(study_start)
    end = pd.Timestamp(study_end)
    anomalies: list[Anomaly] = []
    anomalies.extend(detect_continuous_fixed_value(series))
    anomalies.extend(detect_missing_data(series, start, end))
    anomalies.extend(detect_isolated_point(series))
    anomalies.extend(detect_no_trend_anomaly(series))
    return anomalies
