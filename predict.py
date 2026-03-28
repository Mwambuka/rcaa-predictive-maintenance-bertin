"""
Inference Pipeline — Predictive Maintenance for Aircraft Components
Rwanda Civil Aviation Authority — Data Science Assessment

Usage
-----
Score new sensor data from the command line:

    python predict.py --input new_sensor_data.csv --output scored_output.csv

Or import and use programmatically:

    from predict import load_model_bundle, engineer_features_for_inference, score

Example input CSV columns (same schema as training data):
    aircraft_id, component_id, flight_cycles, engine_hours,
    temperature_sensor_1, temperature_sensor_2, vibration_sensor,
    pressure_sensor, fault_code_count, last_maintenance_cycles,
    maintenance_log_flag, sensor_drift_flag, ambient_temperature, humidity
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_PATH  = "models/best_model.pkl"

# Risk tier boundaries (must match analysis.py)
TIER_BOUNDS = {"CRITICAL": 0.60, "HIGH": 0.35, "MEDIUM": 0.15}
TIER_ACTIONS = {
    "CRITICAL": "Ground aircraft immediately — schedule urgent inspection",
    "HIGH":     "Flag for inspection at next maintenance window",
    "MEDIUM":   "Increase monitoring frequency; review at next service",
    "LOW":      "Standard operations — continue routine monitoring",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_bundle(model_path: str = MODEL_PATH) -> dict:
    """Load the persisted model bundle from disk."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Loaded model: {bundle['model_name']}  "
          f"(threshold={bundle['best_threshold']:.2f})")
    return bundle


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact feature engineering from training.

    IMPORTANT: Input rows must be sorted by (aircraft_id, component_id,
    flight_cycles) and span enough history for rolling windows to be
    meaningful (ideally ≥ 5 prior records per component).
    """
    df = (df.sort_values(["aircraft_id", "component_id", "flight_cycles"])
            .reset_index(drop=True))

    group_key   = ["aircraft_id", "component_id"]
    sensor_cols = ["temperature_sensor_1", "temperature_sensor_2",
                   "vibration_sensor", "pressure_sensor", "fault_code_count"]

    for col in sensor_cols:
        grp = df.groupby(group_key)[col]
        df[f"{col}_roll3_mean"] = grp.transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll3_std"]  = grp.transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df[f"{col}_roll5_mean"] = grp.transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{col}_diff1"]      = grp.transform(
            lambda x: x.diff().fillna(0))

    df["cumulative_faults"] = df.groupby(group_key)["fault_code_count"].transform("cumsum")
    df["fault_roll5_sum"]   = df.groupby(group_key)["fault_code_count"].transform(
        lambda x: x.rolling(5, min_periods=1).sum())

    df["hours_per_cycle"]   = df["engine_hours"] / df["flight_cycles"].replace(0, np.nan)
    df["temp_differential"] = df["temperature_sensor_1"] - df["temperature_sensor_2"]

    for col in ["vibration_sensor", "pressure_sensor"]:
        df[f"{col}_zscore"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    df["stress_index"] = (df["vibration_sensor_zscore"].abs() +
                          df["pressure_sensor_zscore"].abs())

    df["maint_urgency"] = df["last_maintenance_cycles"] * (1 + df["fault_code_count"])
    df["vib_x_temp1"]   = df["vibration_sensor"] * df["temperature_sensor_1"]
    df["drift_x_vib"]   = df["sensor_drift_flag"] * df["vibration_sensor"]
    df["drift_x_faults"]= df["sensor_drift_flag"] * df["fault_code_count"]

    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

def _assign_tier(score: float) -> str:
    if score >= TIER_BOUNDS["CRITICAL"]: return "CRITICAL"
    if score >= TIER_BOUNDS["HIGH"]:     return "HIGH"
    if score >= TIER_BOUNDS["MEDIUM"]:   return "MEDIUM"
    return "LOW"


def score(
    df_raw: pd.DataFrame,
    bundle: dict,
    include_history: bool = True,
) -> pd.DataFrame:
    """
    Score a dataframe of sensor readings.

    Parameters
    ----------
    df_raw          : Raw sensor data (same schema as training data).
    bundle          : Model bundle loaded via load_model_bundle().
    include_history : If True, keep all rows (used for rolling window
                      computation). Set False to return only the latest
                      reading per component.

    Returns
    -------
    DataFrame with added columns:
        - failure_risk_score : Predicted failure probability (0–1)
        - risk_tier          : CRITICAL / HIGH / MEDIUM / LOW
        - alert              : 1 if risk_score >= best_threshold, else 0
        - recommended_action : Plain-language action for maintenance crew
    """
    model         = bundle["model"]
    model_name    = bundle["model_name"]
    scaler        = bundle["scaler"]
    feature_cols  = bundle["feature_cols"]
    threshold     = bundle["best_threshold"]
    le_aircraft   = bundle["le_aircraft"]
    le_component  = bundle["le_component"]

    df = df_raw.copy()

    # Encode categoricals — handle unseen labels gracefully
    def safe_encode(le, series):
        known = set(le.classes_)
        return series.map(lambda x: le.transform([x])[0] if x in known else -1)

    df["aircraft_id_enc"]  = safe_encode(le_aircraft,  df["aircraft_id"])
    df["component_id_enc"] = safe_encode(le_component, df["component_id"])

    df = engineer_features_for_inference(df)

    # Select and order features exactly as training
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    # Drop intermediate zscore columns if they leaked into feature_cols
    for col in ["vibration_sensor_zscore", "pressure_sensor_zscore"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X_input = scaler.transform(X) if model_name == "Logistic Regression" else X

    risk_scores = model.predict_proba(X_input)[:, 1]

    df["failure_risk_score"]  = risk_scores
    df["risk_tier"]           = df["failure_risk_score"].apply(_assign_tier)
    df["alert"]               = (risk_scores >= threshold).astype(int)
    df["recommended_action"]  = df["risk_tier"].map(TIER_ACTIONS)

    if not include_history:
        df = (df.sort_values("flight_cycles")
                .groupby(["aircraft_id", "component_id"])
                .last()
                .reset_index())

    output_cols = (["aircraft_id", "component_id", "flight_cycles",
                    "failure_risk_score", "risk_tier", "alert",
                    "recommended_action"] +
                   [c for c in df.columns
                    if c not in {"aircraft_id", "component_id", "flight_cycles",
                                 "failure_risk_score", "risk_tier", "alert",
                                 "recommended_action"}
                    and c in df_raw.columns])
    return df[[c for c in output_cols if c in df.columns]]


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score aircraft sensor data for failure risk."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input CSV (same schema as training data)")
    parser.add_argument("--output", default="scored_output.csv",
                        help="Path to write scored output (default: scored_output.csv)")
    parser.add_argument("--model",  default=MODEL_PATH,
                        help=f"Path to model bundle (default: {MODEL_PATH})")
    parser.add_argument("--latest-only", action="store_true",
                        help="Output only the latest record per component")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model bundle not found: {args.model}")

    print(f"Loading data from: {args.input}")
    df_raw = pd.read_csv(args.input)
    df_raw.columns = df_raw.columns.str.strip().str.replace('"', "")

    bundle = load_model_bundle(args.model)
    result = score(df_raw, bundle, include_history=not args.latest_only)

    result.to_csv(args.output, index=False)
    print(f"\nScored {len(result):,} records → {args.output}")
    print(f"\nRisk tier summary:\n{result['risk_tier'].value_counts()}")

    alerts = result[result["alert"] == 1]
    if len(alerts):
        print(f"\n⚠  {len(alerts)} component(s) above alert threshold "
              f"(>= {bundle['best_threshold']:.2f}):")
        print(alerts[["aircraft_id", "component_id", "flight_cycles",
                       "failure_risk_score", "risk_tier"]].to_string(index=False))
    else:
        print("\n✓  No components above alert threshold.")


if __name__ == "__main__":
    main()
