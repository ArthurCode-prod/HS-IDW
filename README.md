# HS-IDW

Health-score-driven rainfall monitoring data quality diagnosis and inverse distance weighting reconstruction.

This repository contains the core algorithmic implementation associated with the manuscript:

> A health-score-based framework for quality assessment and reconstruction of rainfall monitoring data in reservoir watersheds

The code is intentionally separated from the manuscript formatting and figure-generation scripts. It provides reusable functions for:

- preprocessing rainfall monitoring records;
- detecting continuous fixed-value, missing-data, isolated-point, and no-trend anomalies;
- calculating station Health Scores;
- selecting high-quality donor stations;
- reconstructing missing or corrupted rainfall records using IDW;
- evaluating reconstruction accuracy.

## Repository structure

```text
HS-IDW_public_code/
  README.md
  requirements.txt
  LICENSE
  src/
    hsidw/
      __init__.py
      preprocessing.py
      anomaly_detection.py
      health_score.py
      spatial_idw.py
      metrics.py
      pipeline.py
  examples/
    run_synthetic_demo.py
```

## Data availability

The operational rainfall monitoring data used in the manuscript are subject to institutional confidentiality agreements and are not included in this repository. The example script uses synthetic data only, for demonstrating the expected input format and reproducing the computational workflow.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run the synthetic demonstration:

```bash
python examples/run_synthetic_demo.py
```

The demonstration creates a small synthetic monitoring network, injects representative anomalies, computes station Health Scores, performs IDW reconstruction, and prints reconstruction metrics.

## Input format

Rainfall observations should be supplied as a table with at least:

| Column | Description |
| --- | --- |
| `station_id` | station identifier |
| `timestamp` | observation timestamp |
| `rainfall` | rainfall depth in mm |

Station metadata should be supplied as a table with at least:

| Column | Description |
| --- | --- |
| `station_id` | station identifier |
| `lon` | longitude in decimal degrees |
| `lat` | latitude in decimal degrees |

## Citation

If you use this code, please cite the associated manuscript after publication.
