"""Synthetic demonstration of the HS-IDW workflow.

This example does not use the confidential operational dataset from the paper.
It creates synthetic rainfall records and station metadata only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsidw.metrics import mae, nse, pbias, rmse
from hsidw.pipeline import reconstruct_station, run_quality_diagnosis


def build_synthetic_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=240, freq="h")
    stations = pd.DataFrame(
        {
            "station_id": ["S001", "S002", "S003", "S004", "S005"],
            "lon": [115.50, 115.53, 115.56, 115.61, 115.67],
            "lat": [31.40, 31.42, 31.43, 31.45, 31.47],
        }
    )

    rows = []
    base = np.maximum(0.0, rng.gamma(shape=0.8, scale=2.2, size=len(timestamps)) - 1.0)
    for idx, station in stations.iterrows():
        noise = rng.normal(0.0, 0.4, len(timestamps))
        rainfall = np.maximum(base * (1.0 + 0.05 * idx) + noise, 0.0)
        for timestamp, value in zip(timestamps, rainfall):
            rows.append({"station_id": station["station_id"], "timestamp": timestamp, "rainfall": round(value, 2)})

    observations = pd.DataFrame(rows)

    # Inject representative anomalies into one low-quality station.
    fixed_mask = (observations["station_id"] == "S005") & (
        observations["timestamp"].between(timestamps[40], timestamps[120])
    )
    observations.loc[fixed_mask, "rainfall"] = 0.5
    missing_mask = (observations["station_id"] == "S005") & (
        observations["timestamp"].between(timestamps[160], timestamps[220])
    )
    observations = observations.loc[~missing_mask].reset_index(drop=True)
    spike_mask = (observations["station_id"] == "S004") & (observations["timestamp"] == timestamps[80])
    observations.loc[spike_mask, "rainfall"] = 35.0
    return observations, stations


def main() -> None:
    observations, stations = build_synthetic_dataset()
    scores, anomalies = run_quality_diagnosis(
        observations=observations,
        study_start="2024-01-01",
        study_end="2024-01-10 23:00:00",
    )
    print("Health scores")
    print(scores)
    print()
    print("Detected anomalies")
    print(anomalies.head(10))
    print()

    target = "S005"
    reconstructed = reconstruct_station(observations, stations, scores, target_station=target)
    reference = observations[observations["station_id"] == "S001"][["timestamp", "rainfall"]].rename(
        columns={"rainfall": "reference"}
    )
    merged = reconstructed.merge(reference, on="timestamp", how="inner")
    print("Synthetic reconstruction metrics against S001 as a demonstration reference")
    print(
        {
            "RMSE": rmse(merged["reference"], merged["reconstructed"]),
            "MAE": mae(merged["reference"], merged["reconstructed"]),
            "NSE": nse(merged["reference"], merged["reconstructed"]),
            "PBIAS": pbias(merged["reference"], merged["reconstructed"]),
        }
    )


if __name__ == "__main__":
    main()
