"""Preprocessing utilities for rainfall monitoring records."""

from __future__ import annotations

import pandas as pd


REQUIRED_OBSERVATION_COLUMNS = ("station_id", "timestamp", "rainfall")


def validate_observation_table(records: pd.DataFrame) -> None:
    """Validate the minimum observation-table schema."""
    missing = [col for col in REQUIRED_OBSERVATION_COLUMNS if col not in records.columns]
    if missing:
        raise ValueError(f"Missing required observation columns: {missing}")


def prepare_rainfall_records(records: pd.DataFrame, keep_zero: bool = True) -> pd.DataFrame:
    """Return cleaned rainfall observations sorted by station and timestamp.

    Duplicate station-time pairs are resolved by retaining the larger rainfall value.
    Non-numeric rainfall values are converted to missing values.
    """
    validate_observation_table(records)
    out = records.loc[:, REQUIRED_OBSERVATION_COLUMNS].copy()
    out["station_id"] = out["station_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["rainfall"] = pd.to_numeric(out["rainfall"], errors="coerce")
    out = out.dropna(subset=["station_id", "timestamp"])
    if not keep_zero:
        out = out[out["rainfall"] > 0]
    out = (
        out.sort_values(["station_id", "timestamp", "rainfall"])
        .drop_duplicates(["station_id", "timestamp"], keep="last")
        .sort_values(["station_id", "timestamp"])
        .reset_index(drop=True)
    )
    return out


def station_series(records: pd.DataFrame, station_id: str, nonzero_only: bool = True) -> pd.DataFrame:
    """Extract a single station rainfall series."""
    records = prepare_rainfall_records(records, keep_zero=not nonzero_only)
    out = records[records["station_id"].astype(str) == str(station_id)].copy()
    if nonzero_only:
        out = out[out["rainfall"] > 0]
    return out.reset_index(drop=True)
