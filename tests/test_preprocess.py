from pathlib import Path

import pandas as pd

from src.preprocess import preprocess


def test_preprocess_creates_required_columns(tmp_path: Path):
    events_raw_path = tmp_path / "synthetic_events_raw.csv"
    cycle_truth_path = tmp_path / "synthetic_cycle_truth.csv"
    processed_path = tmp_path / "cycle_features.csv"

    truth_df = pd.DataFrame(
        {
            "cycle_index": [0, 1, 2],
            "cycle_start_time": [
                "2026-01-01T08:00:00Z",
                "2026-01-01T08:00:02Z",
                "2026-01-01T08:00:04Z",
            ],
            "cycle_end_time": [
                "2026-01-01T08:00:02Z",
                "2026-01-01T08:00:04Z",
                "2026-01-01T08:00:06Z",
            ],
            "cycle_ms": [1500, 1600, 2200],
            "product_code": ["FG-A", "FG-A", "FG-B"],
            "batch_id": ["BATCH-001", "BATCH-001", "BATCH-002"],
            "shift": ["DAY", "DAY", "DAY"],
            "fault_mode": ["NORMAL", "NORMAL", "SLOW_CYCLE"],
            "is_long_gap": [0, 0, 0],
            "is_abnormal_cycle": [0, 0, 1],
        }
    )

    event_rows = []
    event_id = 1

    tag_values_by_cycle = {
        0: {
            "rAngleSealingCrimper": 44.0,
            "rPercSealingAngle": 110.0,
            "Sealing temperature actual": 142.0,
            "Main motor speed percent": 99.0,
            "Reject blowoff command": 0,
            "Network diagnostic counter": 0,
        },
        1: {
            "rAngleSealingCrimper": 45.0,
            "rPercSealingAngle": 111.0,
            "Sealing temperature actual": 143.0,
            "Main motor speed percent": 98.5,
            "Reject blowoff command": 0,
            "Network diagnostic counter": 0,
        },
        2: {
            "rAngleSealingCrimper": 46.0,
            "rPercSealingAngle": 112.0,
            "Sealing temperature actual": 144.0,
            "Main motor speed percent": 97.5,
            "Reject blowoff command": 1,
            "Network diagnostic counter": 2,
        },
    }

    for cycle_index in [0, 1, 2]:
        anchor_time = f"2026-01-01T08:00:0{cycle_index * 2}Z"

        event_rows.append(
            {
                "event_id": event_id,
                "timestamp": anchor_time,
                "tag_name": "Monitoring home position (CAM004)",
                "tag_value": 1,
                "event_type": "RISING_EDGE",
                "product_code": "FG-A",
                "batch_id": "BATCH-001",
                "shift": "DAY",
            }
        )
        event_id += 1

        for tag_name, tag_value in tag_values_by_cycle[cycle_index].items():
            event_rows.append(
                {
                    "event_id": event_id,
                    "timestamp": anchor_time,
                    "tag_name": tag_name,
                    "tag_value": tag_value,
                    "event_type": "VALUE",
                    "product_code": "FG-A",
                    "batch_id": "BATCH-001",
                    "shift": "DAY",
                }
            )
            event_id += 1

    events_df = pd.DataFrame(event_rows)

    truth_df.to_csv(cycle_truth_path, index=False)
    events_df.to_csv(events_raw_path, index=False)

    output_df = preprocess(
        events_raw_path=events_raw_path,
        cycle_truth_path=cycle_truth_path,
        processed_path=processed_path,
    )

    expected_columns = {
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
    }

    assert expected_columns.issubset(set(output_df.columns))
    assert processed_path.exists()
    assert len(output_df) == 2
    assert output_df["previous_cycle_ms"].notna().all()