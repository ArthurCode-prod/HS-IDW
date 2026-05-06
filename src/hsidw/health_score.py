"""Health Score calculation and quality categories."""

from __future__ import annotations

from .anomaly_detection import Anomaly


def anomaly_penalty(anomaly: Anomaly) -> int:
    """Return the Health Score penalty for one anomaly instance."""
    if anomaly.anomaly_type == "CFVA":
        if anomaly.duration_days >= 70:
            return 30
        if anomaly.duration_days >= 30:
            return 20
        return 10
    if anomaly.anomaly_type == "MDA":
        return 100 if "end-of-record" in anomaly.note else 40
    if anomaly.anomaly_type in {"IPA", "NTA"}:
        return 5
    return 0


def compute_health_score(anomalies: list[Anomaly], base_score: int = 100) -> int:
    """Compute station Health Score from detected anomaly instances."""
    return int(base_score - sum(anomaly_penalty(a) for a in anomalies))


def quality_category(score: float) -> str:
    """Map Health Score to an operational quality category."""
    if score >= 80:
        return "high_quality"
    if score >= 60:
        return "acceptable"
    if score >= 0:
        return "poor_quality"
    return "critical"


def anomaly_summary(anomalies: list[Anomaly]) -> dict[str, int]:
    """Count detected anomalies by type."""
    summary = {"CFVA": 0, "MDA": 0, "IPA": 0, "NTA": 0}
    for anomaly in anomalies:
        summary[anomaly.anomaly_type] = summary.get(anomaly.anomaly_type, 0) + 1
    return summary
