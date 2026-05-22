import pandas as pd
from flask import Flask, jsonify, request

from app.model_loader import load_model


FEATURE_COLUMNS = [
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
    "product_code",
]

# removed:
# cycles_ms
# reject_blowoff_command

app = Flask(__name__)
model = load_model()


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "plc-cycle-anomaly-api",
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing_fields = [field for field in FEATURE_COLUMNS if field not in payload]

    if missing_fields:
        return jsonify(
            {
                "error": "Missing required fields",
                "missing_fields": missing_fields,
            }
        ), 400

    feature_df = pd.DataFrame(
        [
            {field: payload[field] for field in FEATURE_COLUMNS}
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(feature_df)[0])

    response = {
        "prediction": prediction,
        "prediction_label": "abnormal_cycle" if prediction == 1 else "normal_cycle",
        "features_used": FEATURE_COLUMNS,
    }

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(feature_df)[0][1])
        response["abnormal_probability"] = probability

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)