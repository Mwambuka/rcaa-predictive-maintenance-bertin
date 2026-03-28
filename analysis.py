"""
Predictive Maintenance for Aircraft Components
Rwanda Civil Aviation Authority — Data Science Assessment

Objective : Predict whether a component will fail within the next 10 flight cycles.
Author    : Data Science Candidate
Date      : March 2026
"""

from __future__ import annotations

import json
import os
import pickle
import warnings

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "aircraft_maintenance_dataset.csv"
FIGURES_DIR  = "figures"
MODELS_DIR   = "models"
RANDOM_STATE = 42

PALETTE      = ["#1f4e79", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad"]
TIER_COLORS  = {"CRITICAL": "#c0392b", "HIGH": "#e67e22",
                "MEDIUM": "#f1c40f",   "LOW":  "#27ae60"}
TIER_BOUNDS  = {"CRITICAL": 0.60, "HIGH": 0.35, "MEDIUM": 0.15}

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
sns.set_palette(PALETTE)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load CSV, strip trailing empty columns and quoted headers."""
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    df.columns = df.columns.str.strip().str.replace('"', "")
    return df


def assess_data_quality(df: pd.DataFrame) -> None:
    """Print data quality report and save summary statistics."""
    print(f"Shape            : {df.shape}")
    print(f"Missing values   :\n{df.isnull().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")

    counts = df["failure_within_10_cycles"].value_counts()
    ratio  = counts[0] / counts[1]
    print(f"\nTarget distribution:\n{counts}")
    print(f"Class imbalance ratio : {ratio:.1f}:1")
    print(f"Unique aircraft       : {df['aircraft_id'].nunique()}")
    print(f"Unique components     : {df['component_id'].nunique()}")
    print(f"AC-component pairs    : {df.groupby(['aircraft_id', 'component_id']).ngroups}")
    print(f"\nNumerical summary:\n{df.describe().round(2)}")

    # Flag data quality issues worth noting
    outliers = {
        "humidity_negative":     (df["humidity"] < 0).sum(),
        "humidity_over_100":     (df["humidity"] > 100).sum(),
        "ambient_temp_lt_minus5":(df["ambient_temperature"] < -5).sum(),
    }
    print(f"\nData anomalies:\n{outliers}")

    df.describe().round(3).to_csv("data_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    """Generate all EDA figures."""
    _plot_target_distribution(df)
    _plot_sensor_distributions(df)
    _plot_correlation_matrix(df)
    _plot_failure_by_component(df)
    _plot_sensor_boxplots(df)
    _plot_sensor_trends(df)
    print("EDA figures saved.")


def _plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["failure_within_10_cycles"].value_counts()

    axes[0].bar(["No Failure\n(0)", "Failure\n(1)"], counts.values,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, f"{v:,}\n({v/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Target Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, counts.max() * 1.25)

    axes[1].pie(counts.values, labels=["No Failure", "Failure"],
                colors=[PALETTE[0], PALETTE[1]], autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportion", fontweight="bold")

    plt.suptitle("Target Variable: Failure Within 10 Flight Cycles",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_target_distribution.png", bbox_inches="tight")
    plt.close()


NUMERIC_FEATURES = [
    "engine_hours", "temperature_sensor_1", "temperature_sensor_2",
    "vibration_sensor", "pressure_sensor", "fault_code_count",
    "last_maintenance_cycles", "ambient_temperature", "humidity",
]
FEATURE_LABELS = {
    "engine_hours":            "Engine Hours",
    "temperature_sensor_1":   "Temperature Sensor 1 (°C)",
    "temperature_sensor_2":   "Temperature Sensor 2 (°C)",
    "vibration_sensor":       "Vibration (g)",
    "pressure_sensor":        "Pressure (psi)",
    "fault_code_count":       "Fault Code Count",
    "last_maintenance_cycles":"Cycles Since Last Maintenance",
    "ambient_temperature":    "Ambient Temperature (°C)",
    "humidity":               "Humidity (%)",
}


def _plot_sensor_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feat in zip(axes.flatten(), NUMERIC_FEATURES):
        for label, color in [(0, PALETTE[0]), (1, PALETTE[1])]:
            subset = df[df["failure_within_10_cycles"] == label][feat].dropna()
            ax.hist(subset, bins=40, alpha=0.65, color=color,
                    label=("No Failure" if label == 0 else "Failure"),
                    density=True, edgecolor="none")
        ax.set_title(FEATURE_LABELS[feat], fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Sensor Distributions by Failure Status",
                 fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_feature_distributions.png", bbox_inches="tight")
    plt.close()


def _plot_correlation_matrix(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[NUMERIC_FEATURES + ["failure_within_10_cycles"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_correlation_matrix.png", bbox_inches="tight")
    plt.close()


def _plot_failure_by_component(df: pd.DataFrame) -> None:
    stats = (df.groupby("component_id")["failure_within_10_cycles"]
               .agg(["mean", "sum", "count"])
               .reset_index()
               .rename(columns={"component_id": "Component",
                                 "mean": "Rate", "sum": "Failures", "count": "Total"})
               .sort_values("Rate", ascending=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(stats["Component"], stats["Rate"] * 100,
                  color=PALETTE[:len(stats)], edgecolor="white", linewidth=1.5)
    for bar, row in zip(bars, stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{row.Failures}/{row.Total}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title("Failure Rate by Component Type", fontweight="bold", fontsize=13)
    ax.set_ylabel("Failure Rate (%)")
    ax.set_xlabel("Component")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_failure_by_component.png", bbox_inches="tight")
    plt.close()


def _plot_sensor_boxplots(df: pd.DataFrame) -> None:
    key_feats = ["vibration_sensor", "temperature_sensor_1", "temperature_sensor_2",
                 "pressure_sensor", "fault_code_count", "last_maintenance_cycles"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flatten(), key_feats):
        bp = ax.boxplot(
            [df[df["failure_within_10_cycles"] == 0][feat],
             df[df["failure_within_10_cycles"] == 1][feat]],
            labels=["No Failure", "Failure"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(PALETTE[0])
        bp["boxes"][1].set_facecolor(PALETTE[1])
        ax.set_title(FEATURE_LABELS[feat], fontweight="bold")
    plt.suptitle("Key Sensor Readings by Failure Status", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_boxplots_by_failure.png", bbox_inches="tight")
    plt.close()


def _plot_sensor_trends(df: pd.DataFrame) -> None:
    sample_ac = df["aircraft_id"].unique()[:4]
    sensor_pairs = [
        ("vibration_sensor",    "Vibration over Flight Cycles"),
        ("temperature_sensor_1","Temperature Sensor 1 over Cycles"),
        ("pressure_sensor",     "Pressure over Flight Cycles"),
        ("fault_code_count",    "Fault Codes over Flight Cycles"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (sensor, title) in zip(axes.flatten(), sensor_pairs):
        for ac, color in zip(sample_ac, PALETTE):
            sub  = df[df["aircraft_id"] == ac].sort_values("flight_cycles")
            fail = sub[sub["failure_within_10_cycles"] == 1]
            ax.plot(sub["flight_cycles"], sub[sensor], alpha=0.5, lw=1,
                    color=color, label=ac)
            if len(fail):
                ax.scatter(fail["flight_cycles"], fail[sensor],
                           color="red", s=40, zorder=5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Flight Cycles")
        ax.legend(fontsize=7)
    plt.suptitle("Sensor Trends — Red dots = Failure Events",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_sensor_trends.png", bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal and interaction features per aircraft-component pair.

    All rolling operations are computed per (aircraft_id, component_id) group,
    sorted by flight_cycles, to prevent any look-ahead leakage across aircraft.
    """
    df_fe = (df.sort_values(["aircraft_id", "component_id", "flight_cycles"])
               .reset_index(drop=True))

    group_key   = ["aircraft_id", "component_id"]
    sensor_cols = ["temperature_sensor_1", "temperature_sensor_2",
                   "vibration_sensor", "pressure_sensor", "fault_code_count"]

    # 3.1 Rolling statistics — capture trend and volatility
    for col in sensor_cols:
        grp = df_fe.groupby(group_key)[col]
        df_fe[f"{col}_roll3_mean"] = grp.transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df_fe[f"{col}_roll3_std"]  = grp.transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df_fe[f"{col}_roll5_mean"] = grp.transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df_fe[f"{col}_diff1"]      = grp.transform(
            lambda x: x.diff().fillna(0))

    # 3.2 Fault accumulation — critical temporal signal
    df_fe["cumulative_faults"] = (
        df_fe.groupby(group_key)["fault_code_count"].transform("cumsum"))
    df_fe["fault_roll5_sum"] = df_fe.groupby(group_key)["fault_code_count"].transform(
        lambda x: x.rolling(5, min_periods=1).sum())

    # 3.3 Efficiency & stress proxies
    df_fe["hours_per_cycle"]   = df_fe["engine_hours"] / df_fe["flight_cycles"].replace(0, np.nan)
    df_fe["temp_differential"] = df_fe["temperature_sensor_1"] - df_fe["temperature_sensor_2"]

    for col in ["vibration_sensor", "pressure_sensor"]:
        df_fe[f"{col}_zscore"] = (df_fe[col] - df_fe[col].mean()) / df_fe[col].std()
    df_fe["stress_index"] = (df_fe["vibration_sensor_zscore"].abs() +
                             df_fe["pressure_sensor_zscore"].abs())

    # 3.4 Maintenance urgency
    df_fe["maint_urgency"] = df_fe["last_maintenance_cycles"] * (1 + df_fe["fault_code_count"])

    # 3.5 Interaction features
    df_fe["vib_x_temp1"]    = df_fe["vibration_sensor"] * df_fe["temperature_sensor_1"]
    df_fe["drift_x_vib"]    = df_fe["sensor_drift_flag"] * df_fe["vibration_sensor"]
    df_fe["drift_x_faults"] = df_fe["sensor_drift_flag"] * df_fe["fault_code_count"]

    # 3.6 Encode categoricals
    le_ac   = LabelEncoder()
    le_comp = LabelEncoder()
    df_fe["aircraft_id_enc"]  = le_ac.fit_transform(df_fe["aircraft_id"])
    df_fe["component_id_enc"] = le_comp.fit_transform(df_fe["component_id"])

    new_feats = [c for c in df_fe.columns if c not in df.columns]
    print(f"Features after engineering: {df_fe.shape[1]} "
          f"({len(new_feats)} new features added)")
    return df_fe, le_ac, le_comp


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_and_split(df_fe: pd.DataFrame) -> tuple:
    """Drop null targets, define feature set, split, and scale."""
    df_fe = df_fe.dropna(subset=["failure_within_10_cycles"]).reset_index(drop=True)

    exclude = {"aircraft_id", "component_id", "failure_within_10_cycles",
               "vibration_sensor_zscore", "pressure_sensor_zscore"}
    feature_cols = [c for c in df_fe.columns if c not in exclude]

    X = df_fe[feature_cols].fillna(df_fe[feature_cols].median())
    y = df_fe["failure_within_10_cycles"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Train : {X_train.shape[0]:,} samples | {y_train.sum()} failures")
    print(f"Test  : {X_test.shape[0]:,} samples  | {y_test.sum()} failures")
    print(f"Features: {len(feature_cols)}")

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            scaler, feature_cols, df_fe)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_models(
    X_train, X_test, y_train, y_test,
    X_train_scaled, X_test_scaled,
) -> dict:
    """
    Train four models with cost-sensitive learning to handle 42:1 class imbalance.
    Returns a results dict keyed by model name.
    """
    neg, pos         = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Scale pos weight (class imbalance factor): {scale_pos_weight:.1f}")

    model_configs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", max_depth=10,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
        ),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in model_configs.items():
        print(f"\n  Training {name} ...", end="  ")
        use_scaled = (name == "Logistic Regression")
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled  if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        cv_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                    scoring="average_precision", n_jobs=-1)

        results[name] = {
            "model":        model,
            "use_scaled":   use_scaled,
            "y_pred":       y_pred,
            "y_proba":      y_proba,
            "roc_auc":      roc_auc_score(y_test, y_proba),
            "avg_precision":average_precision_score(y_test, y_proba),
            "f1":           f1_score(y_test, y_pred),
            "precision":    precision_score(y_test, y_pred, zero_division=0),
            "recall":       recall_score(y_test, y_pred),
            "cm":           confusion_matrix(y_test, y_pred),
            "cv_mean":      cv_scores.mean(),
            "cv_std":       cv_scores.std(),
        }
        r = results[name]
        print(f"ROC-AUC={r['roc_auc']:.4f} | PR-AUC={r['avg_precision']:.4f} "
              f"| F1={r['f1']:.4f} | Recall={r['recall']:.4f} "
              f"| CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_evaluation_figures(results: dict, y_test) -> None:
    _plot_metrics_table(results)
    _plot_roc_pr_curves(results, y_test)
    _plot_confusion_matrices(results, y_test)
    _plot_metrics_comparison(results)


def _plot_metrics_table(results: dict) -> None:
    metrics_df = pd.DataFrame({
        "Model":             list(results.keys()),
        "ROC-AUC":           [r["roc_auc"]       for r in results.values()],
        "PR-AUC":            [r["avg_precision"]  for r in results.values()],
        "F1":                [r["f1"]             for r in results.values()],
        "Precision":         [r["precision"]      for r in results.values()],
        "Recall":            [r["recall"]         for r in results.values()],
        "CV PR-AUC (mean)":  [r["cv_mean"]        for r in results.values()],
        "CV Std":            [r["cv_std"]         for r in results.values()],
    }).set_index("Model").round(4)
    print(f"\nMetrics summary:\n{metrics_df}")
    metrics_df.to_csv("model_metrics.csv")


def _plot_roc_pr_curves(results: dict, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for (name, res), color in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0].plot(fpr, tpr, lw=2, color=color,
                     label=f"{name} (AUC={res['roc_auc']:.3f})")
        prec_c, rec_c, _ = precision_recall_curve(y_test, res["y_proba"])
        axes[1].plot(rec_c, prec_c, lw=2, color=color,
                     label=f"{name} (AP={res['avg_precision']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curves")
    axes[0].legend(fontsize=8)

    base_ap = y_test.mean()
    axes[1].axhline(y=base_ap, color="k", ls="--", lw=1,
                    label=f"Random (AP={base_ap:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall Curves")
    axes[1].legend(fontsize=8)

    plt.suptitle("Model Performance Curves", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/07_roc_pr_curves.png", bbox_inches="tight")
    plt.close()


def _plot_confusion_matrices(results: dict, y_test) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Failure", "Failure"],
                    yticklabels=["No Failure", "Failure"],
                    linewidths=0.5)
        ax.set_title(
            f"{name}\nF1={res['f1']:.3f} | Prec={res['precision']:.3f} "
            f"| Rec={res['recall']:.3f}",
            fontweight="bold",
        )
        ax.set(ylabel="Actual", xlabel="Predicted")
    plt.suptitle("Confusion Matrices — Test Set", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/08_confusion_matrices.png", bbox_inches="tight")
    plt.close()


def _plot_metrics_comparison(results: dict) -> None:
    metric_names = ["ROC-AUC", "PR-AUC", "F1 Score", "Precision", "Recall"]
    x     = np.arange(len(metric_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["roc_auc"], res["avg_precision"],
                res["f1"], res["precision"], res["recall"]]
        ax.bar(x + i * width, vals, width, label=name,
               color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_names, fontweight="bold")
    ax.set(ylabel="Score", title="Model Comparison Across Metrics",
           ylim=(0, 1.12))
    ax.axhline(y=1.0, color="gray", ls="--", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/09_metrics_comparison.png", bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def explain_model(
    results: dict,
    best_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    feature_cols: list,
) -> None:
    """
    Generate three explainability artefacts:
      1. Tree-based feature importance bar chart
      2. SHAP beeswarm + bar plots (via XGBoost native pred_contribs)
      3. SHAP dependence plot for the top feature
      4. Interpretable decision-tree rule extraction
    """
    best_model = results[best_name]["model"]

    # ── 7.1 Feature importance ────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp_df = (pd.DataFrame({
                      "Feature": feature_cols,
                      "Importance": best_model.feature_importances_,
                  })
                  .sort_values("Importance", ascending=False)
                  .head(20))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                color=PALETTE[0], alpha=0.85)
        for bar in ax.patches:
            ax.text(bar.get_width() + 5e-4,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}", va="center", fontsize=7)
        ax.set(xlabel="Feature Importance (Gain)",
               title=f"Top 20 Feature Importances — {best_name}")
        ax.title.set_fontweight("bold")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/10_feature_importance.png", bbox_inches="tight")
        plt.close()

        imp_df.to_csv("feature_importance.csv", index=False)
        print(f"Top 10 features:\n{imp_df.head(10).to_string(index=False)}")

    # ── 7.2 SHAP values via XGBoost native pred_contribs ─────────────────────
    print("\nComputing SHAP values via XGBoost pred_contribs...")
    try:
        booster = best_model.get_booster()
        # Use a representative sample of the test set (200 rows)
        sample     = X_test.iloc[:200]
        dmat       = xgb.DMatrix(sample.values, feature_names=list(sample.columns))
        contribs   = booster.predict(dmat, pred_contribs=True)
        shap_vals  = contribs[:, :-1]           # drop bias column
        shap_df    = pd.DataFrame(shap_vals, columns=feature_cols)

        # Mean |SHAP| ranking
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        top15    = mean_abs.head(15)

        # Beeswarm-style summary plot (manual implementation)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: beeswarm
        n_feats = len(top15)
        cmap    = plt.cm.RdBu_r
        for rank, feat in enumerate(top15.index[::-1]):
            feat_vals   = sample[feat].values
            sv          = shap_df[feat].values
            norm        = Normalize(vmin=np.percentile(feat_vals, 5),
                                    vmax=np.percentile(feat_vals, 95))
            colors_arr  = cmap(norm(feat_vals))
            # jitter y for visibility
            jitter  = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(sv))
            axes[0].scatter(sv, np.full_like(sv, rank) + jitter,
                            c=colors_arr, s=12, alpha=0.7, linewidths=0)
        axes[0].set_yticks(range(n_feats))
        axes[0].set_yticklabels(top15.index[::-1], fontsize=8)
        axes[0].axvline(0, color="gray", lw=0.8, ls="--")
        axes[0].set(xlabel="SHAP Value (impact on model output)",
                    title=f"SHAP Beeswarm — {best_name}")
        axes[0].title.set_fontweight("bold")
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[0], label="Feature value (low→high)", shrink=0.6)

        # Right: mean |SHAP| bar
        axes[1].barh(top15.index[::-1], top15.values[::-1],
                     color=PALETTE[0], alpha=0.85)
        axes[1].set(xlabel="Mean |SHAP Value|",
                    title="Mean Absolute SHAP per Feature")
        axes[1].title.set_fontweight("bold")

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/11_shap_summary.png", bbox_inches="tight")
        plt.close()
        print("SHAP beeswarm saved.")

        # ── 7.3 SHAP dependence plot for top feature ────────────────────────
        top_feat  = top15.index[0]
        second_ft = top15.index[1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, feat_y, feat_col, title in [
            (axes[0], top_feat,   second_ft, f"SHAP Dependence: {top_feat}"),
            (axes[1], second_ft,  top_feat,  f"SHAP Dependence: {second_ft}"),
        ]:
            x_vals   = sample[feat_y].values
            sv_vals  = shap_df[feat_y].values
            col_vals = sample[feat_col].values
            norm     = Normalize(vmin=np.percentile(col_vals, 5),
                                 vmax=np.percentile(col_vals, 95))
            sc = ax.scatter(x_vals, sv_vals, c=plt.cm.RdYlGn_r(norm(col_vals)),
                            s=18, alpha=0.75, linewidths=0)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set(xlabel=feat_y, ylabel="SHAP Value", title=title)
            ax.title.set_fontweight("bold")
            sm = ScalarMappable(cmap=plt.cm.RdYlGn_r)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=feat_col, shrink=0.7)

        plt.suptitle("SHAP Dependence Plots — Top Two Features",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/16_shap_dependence.png", bbox_inches="tight")
        plt.close()
        print("SHAP dependence plot saved.")

    except Exception as exc:
        print(f"SHAP skipped: {exc}")

    # ── 7.4 Interpretable decision-tree rule extraction ───────────────────────
    _extract_decision_rules(X_train, y_train, feature_cols)


def _extract_decision_rules(X_train, y_train, feature_cols: list) -> None:
    """
    Train a shallow Decision Tree and export human-readable rules.
    Shallow trees are interpretable and can be shared directly with
    maintenance engineers as a rule-based reference.
    """
    neg, pos  = np.bincount(y_train)
    dt        = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced",
        min_samples_leaf=10, random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_train)

    rules = export_text(dt, feature_names=feature_cols, max_depth=4)
    with open("decision_tree_rules.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("INTERPRETABLE DECISION TREE RULES (max_depth=4)\n")
        f.write("Use these as a rule-of-thumb reference for maintenance crews\n")
        f.write("=" * 70 + "\n\n")
        f.write(rules)
        f.write(f"\nTrain F1  : {f1_score(y_train, y_pred_dt):.4f}")
        f.write(f"\nTrain Rec : {recall_score(y_train, y_pred_dt):.4f}")

    print(f"Decision tree rules saved → decision_tree_rules.txt")
    print(f"Decision tree (depth-4) train F1={f1_score(y_train, y_pred_dt):.4f}, "
          f"Recall={recall_score(y_train, y_pred_dt):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — LEARNING CURVES & CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(
    results: dict, best_name: str,
    X_train, y_train,
) -> None:
    """
    Plot PR-AUC learning curves for the best model.
    Shows whether the model is data-limited (underfitting) or overfitting.
    """
    best_model = results[best_name]["model"]
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Computing learning curves (this takes ~30 s)...")
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=cv, scoring="average_precision",
        train_sizes=np.linspace(0.15, 1.0, 8),
        n_jobs=-1,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color=PALETTE[0])
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                    alpha=0.15, color=PALETTE[1])
    ax.plot(train_sizes, tr_mean, "o-", color=PALETTE[0], lw=2,
            label="Training PR-AUC")
    ax.plot(train_sizes, va_mean, "s-", color=PALETTE[1], lw=2,
            label="Validation PR-AUC (CV)")
    ax.axhline(va_mean[-1], color="gray", ls="--", lw=1,
               label=f"Final CV PR-AUC = {va_mean[-1]:.4f}")
    ax.set(xlabel="Training samples", ylabel="PR-AUC",
           title=f"Learning Curves — {best_name}",
           ylim=(0, 1.05))
    ax.title.set_fontweight("bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/14_learning_curves.png", bbox_inches="tight")
    plt.close()
    print("Learning curves saved.")


def plot_calibration_curve(results: dict, y_test) -> None:
    """
    Reliability diagram — shows whether predicted probabilities are calibrated.
    A well-calibrated model's curve lies close to the diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for (name, res), color in zip(results.items(), PALETTE):
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_test, res["y_proba"], n_bins=10, strategy="quantile"
        )
        axes[0].plot(prob_pred, prob_true, "s-", color=color, lw=2,
                     markersize=5, label=name)
        # Score histogram
        axes[1].hist(res["y_proba"], bins=30, alpha=0.5, color=color,
                     label=name, density=True)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[0].set(xlabel="Mean Predicted Probability",
                ylabel="Fraction of Positives",
                title="Calibration Curves (Reliability Diagram)")
    axes[0].title.set_fontweight("bold")
    axes[0].legend(fontsize=8)

    axes[1].set(xlabel="Predicted Failure Probability",
                ylabel="Density",
                title="Predicted Probability Distributions")
    axes[1].title.set_fontweight("bold")
    axes[1].legend(fontsize=8)

    plt.suptitle("Model Calibration Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/15_calibration_curve.png", bbox_inches="tight")
    plt.close()
    print("Calibration curve saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — THRESHOLD OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_threshold(best_res: dict, y_test) -> float:
    """Sweep thresholds, find max-F1 point, visualise operational zones."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []
    for t in thresholds:
        yp = (best_res["y_proba"] >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall":    recall_score(y_test, yp),
            "f1":        f1_score(y_test, yp, zero_division=0),
        })
    df_t = pd.DataFrame(rows)
    best_t = df_t.loc[df_t["f1"].idxmax(), "threshold"]

    # Also compute threshold for recall >= 0.95 (safety-critical zone)
    safe_t = df_t[df_t["recall"] >= 0.95]["threshold"].max()
    print(f"Optimal threshold (max F1)       : {best_t:.2f}")
    print(f"Safety threshold (Recall >= 0.95): {safe_t:.2f}")
    print(df_t.loc[df_t["f1"].idxmax()])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_t["threshold"], df_t["precision"], lw=2,
            color=PALETTE[0], label="Precision")
    ax.plot(df_t["threshold"], df_t["recall"],    lw=2,
            color=PALETTE[1], label="Recall")
    ax.plot(df_t["threshold"], df_t["f1"],        lw=2,
            color=PALETTE[2], label="F1 Score")
    ax.axvline(best_t, color="gray", ls="--", lw=1.5,
               label=f"Best F1 threshold ({best_t:.2f})")
    ax.axvline(safe_t, color=PALETTE[1], ls=":", lw=1.5,
               label=f"Safety threshold — Recall≥95% ({safe_t:.2f})")

    ax.axvspan(0.05, 0.30, alpha=0.07, color="red",
               label="High Recall Zone (safety-critical)")
    ax.axvspan(0.55, 0.95, alpha=0.07, color="blue",
               label="High Precision Zone (cost-reduction)")

    ax.set(xlabel="Decision Threshold", ylabel="Score",
           title="Threshold Optimisation — Operational Trade-off",
           xlim=(0.05, 0.95))
    ax.title.set_fontweight("bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/12_threshold_optimization.png", bbox_inches="tight")
    plt.close()

    return float(best_t)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAINTENANCE DECISION SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_risk_tier(score: float) -> str:
    if score >= TIER_BOUNDS["CRITICAL"]: return "CRITICAL"
    if score >= TIER_BOUNDS["HIGH"]:     return "HIGH"
    if score >= TIER_BOUNDS["MEDIUM"]:   return "MEDIUM"
    return "LOW"


def build_risk_scores(
    best_model, best_name: str,
    df_fe: pd.DataFrame, feature_cols: list,
    scaler: StandardScaler, best_thresh: float,
) -> pd.DataFrame:
    """Score every record and assign risk tier."""
    X_all   = df_fe[feature_cols].fillna(df_fe[feature_cols].median())
    X_input = scaler.transform(X_all) if best_name == "Logistic Regression" else X_all

    risk_scores          = best_model.predict_proba(X_input)[:, 1]
    df_fe                = df_fe.copy()
    df_fe["risk_score"]  = risk_scores
    df_fe["risk_tier"]   = df_fe["risk_score"].apply(assign_risk_tier)
    df_fe["alert"]       = (risk_scores >= best_thresh).astype(int)

    tier_counts = df_fe["risk_tier"].value_counts()
    print(f"\nRisk tier distribution:\n{tier_counts}")

    _plot_risk_scoring(df_fe, risk_scores)

    out_cols = ["aircraft_id", "component_id", "flight_cycles",
                "failure_within_10_cycles", "risk_score", "risk_tier", "alert"]
    df_fe[out_cols].to_csv("risk_scores_output.csv", index=False)

    top_risk = (df_fe[df_fe["failure_within_10_cycles"] == 0]
                .nlargest(10, "risk_score")
                [["aircraft_id", "component_id", "flight_cycles",
                  "risk_score", "risk_tier", "vibration_sensor",
                  "fault_code_count", "last_maintenance_cycles"]])
    print(f"\nTop 10 at-risk components (not yet in failure window):\n"
          f"{top_risk.to_string(index=False)}")

    return df_fe


def _plot_risk_scoring(df_fe: pd.DataFrame, risk_scores) -> None:
    tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tier_vals  = [df_fe["risk_tier"].value_counts().get(t, 0) for t in tier_order]
    colors     = [TIER_COLORS[t] for t in tier_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].bar(tier_order, tier_vals, color=colors,
                       edgecolor="white", linewidth=2)
    for bar, val in zip(bars, tier_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5,
                     f"{val:,}\n({val/len(df_fe)*100:.1f}%)",
                     ha="center", va="bottom", fontweight="bold")
    axes[0].set(title="Risk Tier Distribution", ylabel="Records")
    axes[0].title.set_fontweight("bold")

    axes[1].hist(risk_scores, bins=60, color=PALETTE[0], alpha=0.7, edgecolor="none")
    for thresh, color, label in [
        (0.15, TIER_COLORS["LOW"],      "LOW/MEDIUM (0.15)"),
        (0.35, TIER_COLORS["MEDIUM"],   "MEDIUM/HIGH (0.35)"),
        (0.60, TIER_COLORS["CRITICAL"], "HIGH/CRITICAL (0.60)"),
    ]:
        axes[1].axvline(thresh, color=color, lw=2, ls="--", label=label)
    axes[1].set(title="Distribution of Failure Risk Scores",
                xlabel="Predicted Failure Probability", ylabel="Count")
    axes[1].title.set_fontweight("bold")
    axes[1].legend(fontsize=8)

    plt.suptitle("Predictive Maintenance Risk Scoring System",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/13_risk_scoring.png", bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SAVE ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    results: dict, best_name: str, best_thresh: float,
    scaler: StandardScaler, feature_cols: list,
    le_ac: LabelEncoder, le_comp: LabelEncoder,
) -> None:
    """Persist model bundle and JSON summary for downstream use."""
    bundle = {
        "model":          results[best_name]["model"],
        "model_name":     best_name,
        "scaler":         scaler,
        "feature_cols":   feature_cols,
        "best_threshold": best_thresh,
        "le_aircraft":    le_ac,
        "le_component":   le_comp,
    }
    with open(f"{MODELS_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model bundle saved → {MODELS_DIR}/best_model.pkl")

    summary = {
        "models": {
            name: {
                "roc_auc":              round(r["roc_auc"],       4),
                "avg_precision":        round(r["avg_precision"],  4),
                "f1":                   round(r["f1"],             4),
                "precision":            round(r["precision"],      4),
                "recall":               round(r["recall"],         4),
                "cv_mean_avg_precision":round(r["cv_mean"],        4),
                "cv_std":               round(r["cv_std"],         4),
            }
            for name, r in results.items()
        },
        "best_model":     best_name,
        "best_threshold": best_thresh,
    }
    with open("final_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Results summary saved → final_results_summary.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    SEPARATOR = "=" * 62

    print(f"\n{SEPARATOR}\n1. DATA LOADING\n{SEPARATOR}")
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"\n{SEPARATOR}\n2. DATA QUALITY ASSESSMENT\n{SEPARATOR}")
    assess_data_quality(df)

    print(f"\n{SEPARATOR}\n3. EXPLORATORY DATA ANALYSIS\n{SEPARATOR}")
    run_eda(df)

    print(f"\n{SEPARATOR}\n4. FEATURE ENGINEERING\n{SEPARATOR}")
    df_fe, le_ac, le_comp = engineer_features(df)

    print(f"\n{SEPARATOR}\n5. PREPROCESSING\n{SEPARATOR}")
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler, feature_cols, df_fe) = preprocess_and_split(df_fe)

    print(f"\n{SEPARATOR}\n6. MODEL TRAINING\n{SEPARATOR}")
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
    )

    print(f"\n{SEPARATOR}\n7. EVALUATION FIGURES\n{SEPARATOR}")
    plot_evaluation_figures(results, y_test)

    best_name = max(results, key=lambda k: results[k]["avg_precision"])
    best_res  = results[best_name]
    print(f"\nBest model by PR-AUC: {best_name} "
          f"(PR-AUC={best_res['avg_precision']:.4f})")

    print(f"\n{SEPARATOR}\n8. EXPLAINABILITY\n{SEPARATOR}")
    explain_model(results, best_name, X_train, X_test, y_train, feature_cols)

    print(f"\n{SEPARATOR}\n9. LEARNING CURVES\n{SEPARATOR}")
    plot_learning_curves(results, best_name, X_train, y_train)

    print(f"\n{SEPARATOR}\n10. CALIBRATION\n{SEPARATOR}")
    plot_calibration_curve(results, y_test)

    print(f"\n{SEPARATOR}\n11. THRESHOLD OPTIMISATION\n{SEPARATOR}")
    best_thresh = optimise_threshold(best_res, y_test)

    print(f"\n{SEPARATOR}\n12. MAINTENANCE DECISION SUPPORT\n{SEPARATOR}")
    build_risk_scores(
        best_res["model"], best_name, df_fe,
        feature_cols, scaler, best_thresh,
    )

    print(f"\n{SEPARATOR}\n13. SAVING ARTEFACTS\n{SEPARATOR}")
    save_artifacts(results, best_name, best_thresh,
                   scaler, feature_cols, le_ac, le_comp)

    print(f"\n{SEPARATOR}\nANALYSIS COMPLETE\n{SEPARATOR}")
    r = best_res
    print(f"  Model        : {best_name}")
    print(f"  ROC-AUC      : {r['roc_auc']:.4f}")
    print(f"  PR-AUC       : {r['avg_precision']:.4f}")
    print(f"  F1 Score     : {r['f1']:.4f}")
    print(f"  Recall       : {r['recall']:.4f}")
    print(f"  Precision    : {r['precision']:.4f}")
    print(f"  Best thresh  : {best_thresh:.2f}")


if __name__ == "__main__":
    main()
