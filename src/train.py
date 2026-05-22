from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/cycle_features.csv")
MODEL_PATH = Path("models/selected_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
COMPARISON_PATH = Path("reports/model_comparison.csv")

TARGET_COLUMN = "is_abnormal_cycle"

NUMERIC_FEATURE_COLUMNS = [
    "previous_cycle_ms",
    "rolling_mean_cycle_ms",
    "rolling_std_cycle_ms",
    "hour",
    "day_of_week",
    "angle_sealing_crimper",
    "percent_sealing_angle",
    "sealing_temperature_actual",
    "main_motor_speed_percent",
    "network_diagnostic_counter",
]

# removed:
# cycles_ms
# reject_blowoff_command

CATEGORICAL_FEATURE_COLUMNS = [
    "product_code",
]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def safe_roc_auc(y_test, y_score) -> float:
    if len(set(y_test)) < 2:
        return 0.0

    return float(roc_auc_score(y_test, y_score))


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Processed dataset missing columns: {sorted(missing_columns)}")

    df = df.copy()

    for column in NUMERIC_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURE_COLUMNS:
        df[column] = df[column].astype(str).fillna("UNKNOWN")

    df = df.dropna(subset=NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    stratify_target = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_target,
    )

    mlflow.set_experiment("plc-cycle-anomaly-prediction")

    results = []
    best_model = None
    best_model_name = None
    best_f1 = -1.0

    for model_name, model in build_models().items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": safe_roc_auc(y_test, y_score),
            }

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target_column", TARGET_COLUMN)
            mlflow.log_param("numeric_features", ",".join(NUMERIC_FEATURE_COLUMNS))
            mlflow.log_param("categorical_features", ",".join(CATEGORICAL_FEATURE_COLUMNS))

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="plc_cycle_anomaly_classifier",
            )

            result_row = {"model_name": model_name}
            result_row.update(metrics)
            results.append(result_row)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_model_name = model_name

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    comparison_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    final_metrics = {
        "selected_model": best_model_name,
        "selected_metric": "f1",
        "selected_f1": best_f1,
        "rows_used": int(len(df)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2)

    print(f"Selected model: {best_model_name}")
    print(f"Selected F1: {best_f1}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved comparison to: {COMPARISON_PATH}")


if __name__ == "__main__":
    train()