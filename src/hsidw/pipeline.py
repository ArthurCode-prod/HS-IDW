"""High-level HS-IDW workflow functions."""

from __future__ import annotations

import pandas as pd

from .anomaly_detection import detect_anomalies
from .health_score import anomaly_summary, compute_health_score, quality_category
from .preprocessing import prepare_rainfall_records
from .spatial_idw import reconstruct_time_series, select_donor_stations


def run_quality_diagnosis(
    observations: pd.DataFrame,
    study_start: str | pd.Timestamp,
    study_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute anomalies and Health Scores for every station."""
    observations = prepare_rainfall_records(observations, keep_zero=True)
    all_anomalies = []
    score_rows = []

    for station_id in sorted(observations["station_id"].astype(str).unique()):
        anomalies = detect_anomalies(observations, station_id, study_start, study_end)
        score = compute_health_score(anomalies)
        summary = anomaly_summary(anomalies)
        score_rows.append(
            {
                "station_id": station_id,
                "health_score": score,
                "quality_category": quality_category(score),
                **{f"n_{key.lower()}": value for key, value in summary.items()},
            }
        )
        for anomaly in anomalies:
            all_anomalies.append(
                {
                    "station_id": anomaly.station_id,
                    "anomaly_type": anomaly.anomaly_type,
                    "start_time": anomaly.start_time,
                    "end_time": anomaly.end_time,
                    "duration_days": anomaly.duration_days,
                    "magnitude": anomaly.magnitude,
                    "note": anomaly.note,
                }
            )

    return pd.DataFrame(score_rows), pd.DataFrame(all_anomalies)


def reconstruct_station(
    observations: pd.DataFrame,
    station_meta: pd.DataFrame,
    health_scores: pd.DataFrame,
    target_station: str,
    min_health_score: float = 80.0,
    radius_km: float = 5.0,
    power: float = 2.0,
) -> pd.DataFrame:
    """Select donors and reconstruct the target station time series."""
    observations = prepare_rainfall_records(observations, keep_zero=True)
    donors = select_donor_stations(
        station_meta=station_meta,
        target_station=target_station,
        health_scores=health_scores,
        min_health_score=min_health_score,
        radius_km=radius_km,
    )
    reconstructed = reconstruct_time_series(observations, target_station, donors, power=power)
    return reconstructed
