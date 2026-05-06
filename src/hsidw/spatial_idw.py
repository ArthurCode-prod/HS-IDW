"""Spatial donor selection and inverse distance weighting reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two WGS-84 points in kilometres."""
    radius_km = 6371.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(radius_km * c)


def select_donor_stations(
    station_meta: pd.DataFrame,
    target_station: str,
    health_scores: pd.DataFrame,
    min_health_score: float = 80.0,
    radius_km: float = 5.0,
    max_radius_km: float = 40.0,
) -> pd.DataFrame:
    """Select high-quality neighbouring stations for spatial reconstruction."""
    required_meta = {"station_id", "lon", "lat"}
    required_hs = {"station_id", "health_score"}
    if not required_meta.issubset(station_meta.columns):
        raise ValueError(f"station_meta must include {required_meta}")
    if not required_hs.issubset(health_scores.columns):
        raise ValueError(f"health_scores must include {required_hs}")

    meta = station_meta.merge(health_scores, on="station_id", how="left")
    target = meta[meta["station_id"].astype(str) == str(target_station)]
    if target.empty:
        raise ValueError(f"Target station not found: {target_station}")
    target_row = target.iloc[0]

    candidates = meta[
        (meta["station_id"].astype(str) != str(target_station))
        & (pd.to_numeric(meta["health_score"], errors="coerce") >= min_health_score)
    ].copy()
    candidates["distance_km"] = candidates.apply(
        lambda row: haversine_distance_km(target_row["lon"], target_row["lat"], row["lon"], row["lat"]),
        axis=1,
    )

    active_radius = radius_km
    selected = candidates[candidates["distance_km"] <= active_radius].copy()
    while selected.empty and active_radius < max_radius_km:
        active_radius = min(active_radius + radius_km, max_radius_km)
        selected = candidates[candidates["distance_km"] <= active_radius].copy()

    return selected.sort_values("distance_km").reset_index(drop=True)


def idw_reconstruct(
    donor_values: pd.Series | np.ndarray,
    donor_distances_km: pd.Series | np.ndarray,
    power: float = 2.0,
    epsilon: float = 1e-6,
) -> float:
    """Reconstruct one target rainfall value using inverse distance weighting."""
    values = np.asarray(donor_values, dtype=float)
    distances = np.asarray(donor_distances_km, dtype=float)
    valid = np.isfinite(values) & np.isfinite(distances)
    if valid.sum() == 0:
        return float("nan")
    values = values[valid]
    distances = distances[valid]
    weights = 1.0 / np.power(distances + epsilon, power)
    return float(np.sum(weights * values) / np.sum(weights))


def reconstruct_time_series(
    observations: pd.DataFrame,
    target_station: str,
    donors: pd.DataFrame,
    power: float = 2.0,
) -> pd.DataFrame:
    """Reconstruct target station values at all timestamps available from donors."""
    donor_ids = donors["station_id"].astype(str).tolist()
    distance_map = donors.set_index("station_id")["distance_km"].to_dict()
    obs = observations[observations["station_id"].astype(str).isin(donor_ids)].copy()
    rows = []
    for timestamp, group in obs.groupby("timestamp"):
        values = group.set_index("station_id")["rainfall"]
        distances = [distance_map[sid] for sid in values.index]
        rows.append(
            {
                "station_id": str(target_station),
                "timestamp": timestamp,
                "reconstructed": idw_reconstruct(values.to_numpy(), distances, power=power),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
