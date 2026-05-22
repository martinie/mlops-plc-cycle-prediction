from pathlib import Path

import pandas as pd


EVENTS_RAW_PATH = Path("data/raw/synthetic_events_raw.csv")
CYCLE_TRUTH_PATH = Path("data/raw/synthetic_cycle_truth.csv")
PROCESSED_PATH = Path("data/processed/cycle_features.csv")


TAG_FEATURE_MAP = {
    "rAngleSealingCrimper": "angle_sealing_crimper",
    "rPercSealingAngle": "percent_sealing_angle",
    "Sealing temperature actual": "sealing_temperature_actual",
    "Main motor speed percent": "main_motor_speed_percent",
    "Reject blowoff command": "reject_blowoff_command",
    "Network diagnostic counter": "network_diagnostic_counter",
}


OUTPUT_COLUMNS = [
    "cycle_index",
    "cycle_ms",
    "previous_cycle_ms",
    "rolling_mean_cycle_ms",
    "rolling_std_cycle_ms",
    "hour",
    "day_of_week",
    "product_code",
    "batch_id",
    "shift",
    "fault_mode",
    "angle_sealing_crimper",
    "percent_sealing_angle",
    "sealing_temperature_actual",
    "main_motor_speed_percent",
    "reject_blowoff_command",
    "network_diagnostic_counter",
    "is_long_gap",
    "is_abnormal_cycle",
]


def preprocess(
    events_raw_path: Path = EVENTS_RAW_PATH,
    cycle_truth_path: Path = CYCLE_TRUTH_PATH,
    processed_path: Path = PROCESSED_PATH,
) -> pd.DataFrame:
    if not events_raw_path.exists():
        raise FileNotFoundError(f"Raw event file not found: {events_raw_path}")

    if not cycle_truth_path.exists():
        raise FileNotFoundError(f"Cycle truth file not found: {cycle_truth_path}")

    events_df = pd.read_csv(events_raw_path)
    truth_df = pd.read_csv(cycle_truth_path)

    required_event_columns = {
        "event_id",
        "timestamp",
        "tag_name",
        "tag_value",
        "event_type",
    }

    required_truth_columns = {
        "cycle_index",
        "cycle_start_time",
        "cycle_end_time",
        "cycle_ms",
        "product_code",
        "batch_id",
        "shift",
        "fault_mode",
        "is_long_gap",
        "is_abnormal_cycle",
    }

    missing_event_columns = required_event_columns - set(events_df.columns)
    missing_truth_columns = required_truth_columns - set(truth_df.columns)

    if missing_event_columns:
        raise ValueError(f"Raw event file missing columns: {sorted(missing_event_columns)}")

    if missing_truth_columns:
        raise ValueError(f"Cycle truth file missing columns: {sorted(missing_truth_columns)}")

    events_df["timestamp"] = pd.to_datetime(
        events_df["timestamp"],
        utc=True,
        format="mixed",
    )

    truth_df["cycle_start_time"] = pd.to_datetime(
        truth_df["cycle_start_time"],
        utc=True,
        format="mixed",
    )

    truth_df["cycle_end_time"] = pd.to_datetime(
        truth_df["cycle_end_time"],
        utc=True,
        format="mixed",
    )

    events_df = events_df.sort_values(["timestamp", "event_id"]).copy()

    is_cycle_anchor = (
        events_df["tag_name"].eq("Monitoring home position (CAM004)")
        & events_df["event_type"].eq("RISING_EDGE")
        & events_df["tag_value"].eq(1)
    )

    events_df["cycle_index"] = is_cycle_anchor.cumsum() - 1
    events_df = events_df[events_df["cycle_index"] >= 0].copy()
    events_df["cycle_index"] = events_df["cycle_index"].astype(int)

    selected_events_df = events_df[events_df["tag_name"].isin(TAG_FEATURE_MAP.keys())].copy()
    selected_events_df["feature_name"] = selected_events_df["tag_name"].map(TAG_FEATURE_MAP)

    cycle_tag_features_df = (
        selected_events_df
        .pivot_table(
            index="cycle_index",
            columns="feature_name",
            values="tag_value",
            aggfunc="last",
        )
        .reset_index()
    )

    processed_df = truth_df.merge(
        cycle_tag_features_df,
        on="cycle_index",
        how="left",
    )

    previous_cycles = processed_df["cycle_ms"].shift(1)

    processed_df["previous_cycle_ms"] = previous_cycles
    processed_df["rolling_mean_cycle_ms"] = (
        previous_cycles
        .rolling(window=10, min_periods=1)
        .mean()
    )
    processed_df["rolling_std_cycle_ms"] = (
        previous_cycles
        .rolling(window=10, min_periods=2)
        .std()
        .fillna(0)
    )

    processed_df["hour"] = processed_df["cycle_start_time"].dt.hour
    processed_df["day_of_week"] = processed_df["cycle_start_time"].dt.dayofweek

    processed_df["network_diagnostic_counter"] = (
        processed_df["network_diagnostic_counter"]
        .fillna(0)
    )

    processed_df = processed_df.dropna(subset=["previous_cycle_ms"]).copy()
    processed_df = processed_df[OUTPUT_COLUMNS]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)

    return processed_df


if __name__ == "__main__":
    output_df = preprocess()
    print(f"Processed rows: {len(output_df)}")
    print(f"Saved to: {PROCESSED_PATH}")