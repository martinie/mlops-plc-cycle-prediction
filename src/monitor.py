from pathlib import Path
import json

import pandas as pd


DATA_PATH = Path("data/processed/cycle_features.csv")
DRIFT_REPORT_PATH = Path("reports/drift_report.json")


def monitor() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    mean_cycle_ms = float(df["cycle_ms"].mean())
    abnormal_rate = float(df["is_abnormal_cycle"].mean())

    drift_detected = mean_cycle_ms > 2500 or abnormal_rate > 0.20

    report = {
        "mean_cycle_ms": mean_cycle_ms,
        "abnormal_rate": abnormal_rate,
        "drift_detected": drift_detected,
        "mean_cycle_ms_threshold": 2500,
        "abnormal_rate_threshold": 0.20,
        "action": "retraining_recommended" if drift_detected else "no_retraining_required",
    }

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DRIFT_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    monitor()