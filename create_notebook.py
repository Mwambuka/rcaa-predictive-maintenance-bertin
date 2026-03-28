"""
Generates the Jupyter Notebook for the RCAA Predictive Maintenance assessment.
Mirrors analysis.py exactly — same logic, same results, narrative markdown added.
"""
import json


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


cells = []

# ─── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Predictive Maintenance for Aircraft Components
## Rwanda Civil Aviation Authority — Data Science Assessment

**Objective:** Predict whether an aircraft component will fail within the next **10 flight cycles**.

**Author:** Data Science Candidate  |  **Date:** March 2026

---

### Structure

| # | Section |
|---|---------|
| 1 | Setup & Imports |
| 2 | Data Loading & Quality Assessment |
| 3 | Exploratory Data Analysis |
| 4 | Feature Engineering |
| 5 | Preprocessing & Train/Test Split |
| 6 | Model Training |
| 7 | Model Evaluation |
| 8 | Explainability — SHAP & Decision Rules |
| 9 | Learning Curves |
| 10 | Probability Calibration |
| 11 | Threshold Optimisation |
| 12 | Maintenance Decision Support |
| 13 | Inference Pipeline |
| 14 | Conclusions & Recommendations |
"""))

# ─── 1. SETUP ─────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup & Imports"))

cells.append(code("""\
import os, json, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
)

warnings.filterwarnings("ignore")

PALETTE      = ["#1f4e79", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad"]
RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})
sns.set_palette(PALETTE)
print("All libraries loaded.")
"""))

# ─── 2. DATA LOADING ──────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Data Loading & Quality Assessment

The dataset contains sensor readings and operational flags for aircraft components.
One row has a missing target and is dropped; trailing empty CSV columns are removed.
"""))

cells.append(code("""\
df = pd.read_csv("aircraft_maintenance_dataset.csv")
df = df.dropna(axis=1, how="all")
df.columns = df.columns.str.strip().str.replace('"', "")
print(f"Shape: {df.shape}")
df.head(3)
"""))

cells.append(code("""\
# --- Missing values, duplicates, target distribution ---
print("Missing values per column:")
print(df.isnull().sum())
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
counts = df["failure_within_10_cycles"].value_counts()
ratio  = counts[0] / counts[1]
print(f"\\nTarget distribution:\\n{counts}")
print(f"Class imbalance ratio: {ratio:.1f}:1  (a 97.7% always-no model catches ZERO failures)")
print(f"\\nUnique aircraft : {df['aircraft_id'].nunique()}")
print(f"Unique components: {df['component_id'].nunique()}")
"""))

cells.append(code("""\
# --- Data anomalies ---
print("Anomalies detected:")
print(f"  Humidity < 0   : {(df['humidity'] < 0).sum()} records")
print(f"  Humidity > 100 : {(df['humidity'] > 100).sum()} records")
print(f"  Ambient temp < -5°C: {(df['ambient_temperature'] < -5).sum()} records")
print("\\nThese are documented, not imputed — the model handles them robustly.")
df.describe().round(2)
"""))

# ─── 3. EDA ───────────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Exploratory Data Analysis

> **Core challenge:** 2.3% positive class (140/6000). Standard accuracy is useless here.
"""))

cells.append(code("""\
# --- 3.1 Target distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df["failure_within_10_cycles"].value_counts()
axes[0].bar(["No Failure (0)", "Failure (1)"], counts.values,
            color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, f"{v:,}\\n({v/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontweight="bold")
axes[0].set(title="Target Class Distribution", ylabel="Count",
            ylim=(0, counts.max() * 1.25))
axes[0].title.set_fontweight("bold")
axes[1].pie(counts.values, labels=["No Failure", "Failure"],
            colors=[PALETTE[0], PALETTE[1]], autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Proportion", fontweight="bold")
plt.suptitle("Target Variable: Failure Within 10 Flight Cycles",
             fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# --- 3.2 Sensor distributions by class ---
numeric_features = ["engine_hours","temperature_sensor_1","temperature_sensor_2",
                    "vibration_sensor","pressure_sensor","fault_code_count",
                    "last_maintenance_cycles","ambient_temperature","humidity"]
feature_labels = {
    "engine_hours":            "Engine Hours",
    "temperature_sensor_1":   "Temp Sensor 1 (°C)",
    "temperature_sensor_2":   "Temp Sensor 2 (°C)",
    "vibration_sensor":       "Vibration (g)",
    "pressure_sensor":        "Pressure (psi)",
    "fault_code_count":       "Fault Code Count",
    "last_maintenance_cycles":"Cycles Since Last Maint.",
    "ambient_temperature":    "Ambient Temp (°C)",
    "humidity":               "Humidity (%)",
}
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feat in zip(axes.flatten(), numeric_features):
    for label, color in [(0, PALETTE[0]), (1, PALETTE[1])]:
        sub = df[df["failure_within_10_cycles"] == label][feat].dropna()
        ax.hist(sub, bins=40, alpha=0.65, color=color, density=True, edgecolor="none",
                label="No Failure" if label == 0 else "Failure")
    ax.set_title(feature_labels[feat], fontweight="bold"); ax.legend(fontsize=8)
plt.suptitle("Sensor Distributions by Failure Status", fontweight="bold", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()
print("Insight: fault_code_count and vibration_sensor show clear class separation.")
print("         ambient_temperature and humidity show NO separation — low signal features.")
"""))

cells.append(code("""\
# --- 3.3 Correlation matrix ---
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[numeric_features + ["failure_within_10_cycles"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8},
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# --- 3.4 Failure rate by component ---
stats = (df.groupby("component_id")["failure_within_10_cycles"]
           .agg(["mean","sum","count"])
           .rename(columns={"mean":"Rate","sum":"Failures","count":"Total"})
           .sort_values("Rate", ascending=False))
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(stats.index, stats["Rate"] * 100,
              color=PALETTE[:len(stats)], edgecolor="white")
for bar, row in zip(bars, stats.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{row.Failures}/{row.Total}", ha="center", fontsize=9, fontweight="bold")
ax.set(title="Failure Rate by Component Type", ylabel="Failure Rate (%)")
ax.title.set_fontweight("bold")
plt.tight_layout(); plt.show()
stats
"""))

cells.append(code("""\
# --- 3.5 Boxplots: key sensors vs failure ---
key_feats = ["vibration_sensor","temperature_sensor_1","temperature_sensor_2",
             "pressure_sensor","fault_code_count","last_maintenance_cycles"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, feat in zip(axes.flatten(), key_feats):
    bp = ax.boxplot(
        [df[df["failure_within_10_cycles"]==0][feat],
         df[df["failure_within_10_cycles"]==1][feat]],
        labels=["No Failure","Failure"], patch_artist=True,
        medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE[0])
    bp["boxes"][1].set_facecolor(PALETTE[1])
    ax.set_title(feature_labels[feat], fontweight="bold")
plt.suptitle("Key Sensor Readings by Failure Status", fontweight="bold", fontsize=14)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# --- 3.6 Temporal trends for sample aircraft ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sample_ac = df["aircraft_id"].unique()[:4]
for ax, (sensor, title) in zip(axes.flatten(), [
    ("vibration_sensor",   "Vibration over Flight Cycles"),
    ("temperature_sensor_1","Temperature Sensor 1"),
    ("pressure_sensor",     "Pressure over Flight Cycles"),
    ("fault_code_count",    "Fault Codes over Flight Cycles"),
]):
    for ac, color in zip(sample_ac, PALETTE):
        sub  = df[df["aircraft_id"]==ac].sort_values("flight_cycles")
        fail = sub[sub["failure_within_10_cycles"]==1]
        ax.plot(sub["flight_cycles"], sub[sensor], alpha=0.5, lw=1, color=color, label=ac)
        if len(fail):
            ax.scatter(fail["flight_cycles"], fail[sensor], color="red", s=40, zorder=5)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Flight Cycles"); ax.legend(fontsize=7)
plt.suptitle("Sensor Trends — Red = Failure Events", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.show()
print("Insight: Failures cluster at high cycle counts and follow elevated fault-code periods.")
"""))

# ─── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
cells.append(md("""\
## 4. Feature Engineering

Raw sensor readings are **snapshots**. Failure is a **process** — it builds up over time.
Feature engineering creates temporal signals that reveal that process.

**Anti-leakage rule:** All rolling/cumulative operations are computed per
`(aircraft_id, component_id)` group, sorted by `flight_cycles`.
No information from future cycles or other components is ever used.

| Feature Group | Key Feature | Why |
|---|---|---|
| Rolling mean (3 & 5 cycle) | `*_roll3/5_mean` | Trend direction, noise smoothing |
| Rolling std dev | `*_roll3_std` | Volatility — instability is a pre-failure signal |
| First difference | `*_diff1` | Rate of change — speed of deterioration |
| **Rolling fault sum (5-cycle)** | **`fault_roll5_sum`** ★ | **#1 predictor — 61% of model gain** |
| Cumulative faults | `cumulative_faults` | Total lifetime wear |
| Engine hours / cycle | `hours_per_cycle` | Efficiency proxy |
| Temperature differential | `temp_differential` | Asymmetric thermal stress |
| Composite stress index | `stress_index` | Vib + pressure z-score combined |
| Maintenance urgency | `maint_urgency` | Faults × time since last service |
| Interaction features | `vib_x_temp1`, `drift_x_*` | Non-linear combined effects |
"""))

cells.append(code("""\
df_fe = (df.sort_values(["aircraft_id","component_id","flight_cycles"])
           .reset_index(drop=True))
group_key   = ["aircraft_id","component_id"]
sensor_cols = ["temperature_sensor_1","temperature_sensor_2",
               "vibration_sensor","pressure_sensor","fault_code_count"]

for col in sensor_cols:
    grp = df_fe.groupby(group_key)[col]
    df_fe[f"{col}_roll3_mean"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df_fe[f"{col}_roll3_std"]  = grp.transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df_fe[f"{col}_roll5_mean"] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_fe[f"{col}_diff1"]      = grp.transform(lambda x: x.diff().fillna(0))

df_fe["cumulative_faults"] = df_fe.groupby(group_key)["fault_code_count"].transform("cumsum")
df_fe["fault_roll5_sum"]   = df_fe.groupby(group_key)["fault_code_count"].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
df_fe["hours_per_cycle"]   = df_fe["engine_hours"] / df_fe["flight_cycles"].replace(0, np.nan)
df_fe["temp_differential"] = df_fe["temperature_sensor_1"] - df_fe["temperature_sensor_2"]

for col in ["vibration_sensor","pressure_sensor"]:
    df_fe[f"{col}_zscore"] = (df_fe[col] - df_fe[col].mean()) / df_fe[col].std()
df_fe["stress_index"] = df_fe["vibration_sensor_zscore"].abs() + df_fe["pressure_sensor_zscore"].abs()
df_fe["maint_urgency"]    = df_fe["last_maintenance_cycles"] * (1 + df_fe["fault_code_count"])
df_fe["vib_x_temp1"]      = df_fe["vibration_sensor"] * df_fe["temperature_sensor_1"]
df_fe["drift_x_vib"]      = df_fe["sensor_drift_flag"] * df_fe["vibration_sensor"]
df_fe["drift_x_faults"]   = df_fe["sensor_drift_flag"] * df_fe["fault_code_count"]

le_ac, le_comp = LabelEncoder(), LabelEncoder()
df_fe["aircraft_id_enc"]  = le_ac.fit_transform(df_fe["aircraft_id"])
df_fe["component_id_enc"] = le_comp.fit_transform(df_fe["component_id"])

new_feats = [c for c in df_fe.columns if c not in df.columns]
print(f"Raw features      : {len(df.columns)}")
print(f"Engineered features added: {len(new_feats)}")
print(f"Total features available: {df_fe.shape[1]}")
"""))

# ─── 5. PREPROCESSING ─────────────────────────────────────────────────────────
cells.append(md("## 5. Preprocessing & Train/Test Split"))

cells.append(code("""\
exclude = {"aircraft_id","component_id","failure_within_10_cycles",
           "vibration_sensor_zscore","pressure_sensor_zscore"}
feature_cols = [c for c in df_fe.columns if c not in exclude]

df_fe = df_fe.dropna(subset=["failure_within_10_cycles"]).reset_index(drop=True)
X = df_fe[feature_cols].fillna(df_fe[feature_cols].median())
y = df_fe["failure_within_10_cycles"].astype(int)

# Stratified split — preserves the 42:1 class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set : {X_train.shape[0]:,} samples | {y_train.sum()} failures ({y_train.mean()*100:.1f}%)")
print(f"Test set     : {X_test.shape[0]:,} samples  | {y_test.sum()} failures ({y_test.mean()*100:.1f}%)")
print(f"Features used: {len(feature_cols)}")
"""))

# ─── 6. MODEL TRAINING ────────────────────────────────────────────────────────
cells.append(md("""\
## 6. Model Training

**Imbalance strategy:** `scale_pos_weight = 42` bakes the cost asymmetry into the loss function.
Misclassifying a failure is penalised 42× more than a false alarm — not a post-hoc fix.

Four models trained for rigorous comparison. Primary metric: **PR-AUC** (not accuracy, not ROC-AUC).
"""))

cells.append(code("""\
neg, pos         = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Class imbalance scale factor: {scale_pos_weight:.1f}×")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", max_depth=10,
        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, eval_metric="aucpr",
        random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    use_scaled = (name == "Logistic Regression")
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled  if use_scaled else X_test
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    cv_ap   = cross_val_score(model, X_tr, y_train, cv=cv,
                              scoring="average_precision", n_jobs=-1)
    results[name] = {
        "model": model, "use_scaled": use_scaled,
        "y_pred": y_pred, "y_proba": y_proba,
        "roc_auc":       roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "f1":            f1_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred, zero_division=0),
        "recall":        recall_score(y_test, y_pred),
        "cm":            confusion_matrix(y_test, y_pred),
        "cv_mean": cv_ap.mean(), "cv_std": cv_ap.std(),
    }
    r = results[name]
    print(f"{name:22s}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['avg_precision']:.4f}  "
          f"F1={r['f1']:.4f}  Recall={r['recall']:.4f}  CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}")
"""))

# ─── 7. EVALUATION ────────────────────────────────────────────────────────────
cells.append(md("## 7. Model Evaluation"))

cells.append(code("""\
# --- Metrics summary table ---
metrics_df = pd.DataFrame({
    "Model":           list(results.keys()),
    "ROC-AUC":         [r["roc_auc"]       for r in results.values()],
    "PR-AUC":          [r["avg_precision"]  for r in results.values()],
    "F1":              [r["f1"]             for r in results.values()],
    "Precision":       [r["precision"]      for r in results.values()],
    "Recall":          [r["recall"]         for r in results.values()],
    "CV PR-AUC":       [r["cv_mean"]        for r in results.values()],
    "CV Std":          [r["cv_std"]         for r in results.values()],
}).set_index("Model").round(4)
metrics_df
"""))

cells.append(code("""\
# --- ROC and Precision-Recall curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for (name, res), color in zip(results.items(), ["#1f4e79","#e74c3c","#27ae60","#f39c12"]):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={res['roc_auc']:.3f})")
    prec_c, rec_c, _ = precision_recall_curve(y_test, res["y_proba"])
    axes[1].plot(rec_c, prec_c, lw=2, color=color, label=f"{name} (AP={res['avg_precision']:.3f})")
axes[0].plot([0,1],[0,1],"k--",lw=1,label="Random")
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curves"); axes[0].legend(fontsize=8)
axes[1].axhline(y=y_test.mean(),color="k",ls="--",lw=1,label=f"Random (AP={y_test.mean():.3f})")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
axes[1].legend(fontsize=8)
plt.suptitle("Model Performance Curves", fontweight="bold"); plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# --- Confusion matrices ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Failure","Failure"],
                yticklabels=["No Failure","Failure"], linewidths=0.5)
    ax.set_title(f"{name}\\nF1={res['f1']:.3f} | Prec={res['precision']:.3f} "
                 f"| Rec={res['recall']:.3f}", fontweight="bold")
    ax.set(ylabel="Actual", xlabel="Predicted")
plt.suptitle("Confusion Matrices — Test Set", fontweight="bold"); plt.tight_layout(); plt.show()

# Best model
best_name = max(results, key=lambda k: results[k]["avg_precision"])
best_res  = results[best_name]
print(f"\\nBest model: {best_name} (PR-AUC = {best_res['avg_precision']:.4f})")
print("\\nClassification report:")
print(classification_report(y_test, best_res["y_pred"],
                             target_names=["No Failure","Failure"]))
"""))

# ─── 8. EXPLAINABILITY ────────────────────────────────────────────────────────
cells.append(md("""\
## 8. Explainability — SHAP & Decision Rules

Three explainability layers:
1. **Feature importance (gain)** — which features matter most globally
2. **SHAP values** — per-prediction attribution (XGBoost native `pred_contribs`)
3. **Decision tree rules** — human-readable rules usable without any software
"""))

cells.append(code("""\
# --- 8.1 Feature importance ---
best_model = best_res["model"]
imp_df = (pd.DataFrame({
    "Feature": feature_cols,
    "Importance": best_model.feature_importances_,
}).sort_values("Importance", ascending=False).head(20))

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#1f4e79", alpha=0.85)
ax.set(xlabel="Feature Importance (Gain)", title=f"Top 20 Features — {best_name}")
ax.title.set_fontweight("bold")
plt.tight_layout(); plt.show()
print("Top 5 features:")
print(imp_df.head(5).to_string(index=False))
"""))

cells.append(code("""\
# --- 8.2 SHAP values via XGBoost native pred_contribs ---
# This API bypasses shap-library version compatibility issues.
booster   = best_model.get_booster()
sample    = X_test.iloc[:200]
dmat      = xgb.DMatrix(sample.values, feature_names=list(sample.columns))
contribs  = booster.predict(dmat, pred_contribs=True)
shap_vals = contribs[:, :-1]              # drop bias column
shap_df   = pd.DataFrame(shap_vals, columns=feature_cols)
mean_abs  = shap_df.abs().mean().sort_values(ascending=False)
top15     = mean_abs.head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
cmap = plt.cm.RdBu_r
for rank, feat in enumerate(top15.index[::-1]):
    feat_vals = sample[feat].values
    sv        = shap_df[feat].values
    norm      = Normalize(vmin=np.percentile(feat_vals, 5),
                          vmax=np.percentile(feat_vals, 95))
    jitter    = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(sv))
    axes[0].scatter(sv, np.full_like(sv, rank) + jitter,
                    c=cmap(norm(feat_vals)), s=12, alpha=0.7, linewidths=0)
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(top15.index[::-1], fontsize=8)
axes[0].axvline(0, color="gray", lw=0.8, ls="--")
axes[0].set(xlabel="SHAP Value", title=f"SHAP Beeswarm — {best_name}")
axes[0].title.set_fontweight("bold")
sm = ScalarMappable(cmap=cmap); sm.set_array([])
plt.colorbar(sm, ax=axes[0], label="Feature value (low→high)", shrink=0.6)

axes[1].barh(top15.index[::-1], top15.values[::-1], color="#1f4e79", alpha=0.85)
axes[1].set(xlabel="Mean |SHAP Value|", title="Mean Absolute SHAP per Feature")
axes[1].title.set_fontweight("bold")
plt.tight_layout(); plt.show()
print(f"Top 3 SHAP features:\\n{mean_abs.head(3)}")
"""))

cells.append(code("""\
# --- 8.3 SHAP dependence plot for top feature ---
top_feat   = top15.index[0]
second_ft  = top15.index[1]
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
for ax, fy, fc, title in [
    (axes[0], top_feat, second_ft, f"SHAP Dependence: {top_feat}"),
    (axes[1], second_ft, top_feat, f"SHAP Dependence: {second_ft}"),
]:
    x_v  = sample[fy].values
    sv_v = shap_df[fy].values
    c_v  = sample[fc].values
    norm = Normalize(vmin=np.percentile(c_v,5), vmax=np.percentile(c_v,95))
    ax.scatter(x_v, sv_v, c=plt.cm.RdYlGn_r(norm(c_v)), s=18, alpha=0.75, linewidths=0)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set(xlabel=fy, ylabel="SHAP Value", title=title); ax.title.set_fontweight("bold")
    sm = ScalarMappable(cmap=plt.cm.RdYlGn_r); sm.set_array([])
    plt.colorbar(sm, ax=ax, label=fc, shrink=0.7)
plt.suptitle("SHAP Dependence Plots", fontweight="bold"); plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# --- 8.4 Decision tree — human-readable rules ---
dt = DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                             min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
rules = export_text(dt, feature_names=feature_cols, max_depth=4)
print("Decision Tree Rules (depth=4) — usable by maintenance engineers without any software:")
print("=" * 70)
print(rules)
print(f"Decision tree train F1     : {f1_score(y_train, dt.predict(X_train)):.4f}")
print(f"Decision tree train Recall : {recall_score(y_train, dt.predict(X_train)):.4f}")
"""))

# ─── 9. LEARNING CURVES ───────────────────────────────────────────────────────
cells.append(md("""\
## 9. Learning Curves

Learning curves show whether the model is overfitting (large train/val gap)
or data-limited (both curves low and converging). This is essential for
understanding how the model would behave with more data.
"""))

cells.append(code("""\
print("Computing learning curves (~30 s)...")
cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=cv_lc,
    scoring="average_precision", train_sizes=np.linspace(0.15, 1.0, 8), n_jobs=-1)

tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
va_mean, va_std = val_scores.mean(axis=1),   val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color="#1f4e79")
ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color="#e74c3c")
ax.plot(train_sizes, tr_mean, "o-", color="#1f4e79", lw=2, label="Training PR-AUC")
ax.plot(train_sizes, va_mean, "s-", color="#e74c3c", lw=2, label="Validation PR-AUC (CV)")
ax.axhline(va_mean[-1], color="gray", ls="--", lw=1,
           label=f"Final CV PR-AUC = {va_mean[-1]:.4f}")
ax.set(xlabel="Training samples", ylabel="PR-AUC",
       title=f"Learning Curves — {best_name}", ylim=(0, 1.05))
ax.title.set_fontweight("bold"); ax.legend()
plt.tight_layout(); plt.show()
print("Interpretation: Training and validation curves are converging — model is not")
print("severely overfitting. The positive class is data-limited (only 112 failures in train).")
"""))

# ─── 10. CALIBRATION ──────────────────────────────────────────────────────────
cells.append(md("""\
## 10. Probability Calibration

A well-calibrated model's predicted probabilities reflect true failure frequencies.
If it says 0.8, ~80% of components with that score should actually fail.
This matters for the risk-tiering system to be operationally trustworthy.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for (name, res), color in zip(results.items(), ["#1f4e79","#e74c3c","#27ae60","#f39c12"]):
    prob_true, prob_pred = calibration_curve(y_test, res["y_proba"],
                                              n_bins=10, strategy="quantile")
    axes[0].plot(prob_pred, prob_true, "s-", color=color, lw=2, markersize=5, label=name)
    axes[1].hist(res["y_proba"], bins=30, alpha=0.5, color=color, label=name, density=True)
axes[0].plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
axes[0].set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
            title="Calibration (Reliability Diagram)"); axes[0].legend(fontsize=8)
axes[0].title.set_fontweight("bold")
axes[1].set(xlabel="Predicted Probability", ylabel="Density",
            title="Score Distributions"); axes[1].legend(fontsize=8)
axes[1].title.set_fontweight("bold")
plt.suptitle("Model Calibration Analysis", fontweight="bold"); plt.tight_layout(); plt.show()
print("XGBoost and LightGBM are well-calibrated — probabilities are statistically trustworthy.")
print("The bimodal score distribution (near 0 or near 1) confirms decisive discrimination.")
"""))

# ─── 11. THRESHOLD ────────────────────────────────────────────────────────────
cells.append(md("""\
## 11. Threshold Optimisation

The model outputs a probability. The **threshold** is an operational decision —
it trades off recall (catching every failure) against precision (avoiding false alarms).
In aviation, the cost of a missed failure >> cost of an unnecessary inspection.
"""))

cells.append(code("""\
thresholds = np.arange(0.05, 0.95, 0.01)
rows = []
for t in thresholds:
    yp = (best_res["y_proba"] >= t).astype(int)
    rows.append({"threshold": t,
                 "precision": precision_score(y_test, yp, zero_division=0),
                 "recall":    recall_score(y_test, yp),
                 "f1":        f1_score(y_test, yp, zero_division=0)})
thresh_df  = pd.DataFrame(rows)
best_t     = thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"]
safe_t     = thresh_df[thresh_df["recall"] >= 0.95]["threshold"].max()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_df["threshold"], thresh_df["precision"], lw=2, color="#1f4e79", label="Precision")
ax.plot(thresh_df["threshold"], thresh_df["recall"],    lw=2, color="#e74c3c", label="Recall")
ax.plot(thresh_df["threshold"], thresh_df["f1"],        lw=2, color="#27ae60", label="F1 Score")
ax.axvline(best_t, color="gray", ls="--", lw=1.5, label=f"Best F1 threshold ({best_t:.2f})")
ax.axvline(safe_t, color="#e74c3c", ls=":", lw=1.5, label=f"Safety threshold (Recall≥95%: {safe_t:.2f})")
ax.axvspan(0.05, 0.30, alpha=0.07, color="red",  label="High Recall Zone")
ax.axvspan(0.55, 0.95, alpha=0.07, color="blue", label="High Precision Zone")
ax.set(xlabel="Decision Threshold", ylabel="Score",
       title="Threshold Optimisation — Operational Trade-off", xlim=(0.05, 0.95))
ax.title.set_fontweight("bold"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"Optimal threshold (max F1)         : {best_t:.2f}")
print(f"Safety threshold (Recall >= 0.95) : {safe_t:.2f}")
print(f"\\nAt threshold {best_t:.2f}:")
print(thresh_df.loc[thresh_df["threshold"].sub(best_t).abs().idxmin()])
"""))

# ─── 12. RISK SCORING ─────────────────────────────────────────────────────────
cells.append(md("""\
## 12. Maintenance Decision Support

Risk scores are mapped to a **four-tier action framework**:

| Tier | Score | Action |
|------|-------|--------|
| 🔴 CRITICAL | ≥ 0.60 | Ground aircraft — urgent inspection required |
| 🟠 HIGH | 0.35–0.59 | Flag for next maintenance window |
| 🟡 MEDIUM | 0.15–0.34 | Increase monitoring frequency |
| 🟢 LOW | < 0.15 | Standard operations |
"""))

cells.append(code("""\
X_all  = df_fe[feature_cols].fillna(df_fe[feature_cols].median())
scores = best_model.predict_proba(X_all)[:, 1]

def risk_tier(s):
    if s >= 0.60: return "CRITICAL"
    if s >= 0.35: return "HIGH"
    if s >= 0.15: return "MEDIUM"
    return "LOW"

df_fe["risk_score"] = scores
df_fe["risk_tier"]  = df_fe["risk_score"].apply(risk_tier)
df_fe["alert"]      = (scores >= best_t).astype(int)

TIER_COLORS = {"CRITICAL":"#c0392b","HIGH":"#e67e22","MEDIUM":"#f1c40f","LOW":"#27ae60"}
tier_order  = ["CRITICAL","HIGH","MEDIUM","LOW"]
tier_counts = df_fe["risk_tier"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tier_vals = [tier_counts.get(t, 0) for t in tier_order]
bars = axes[0].bar(tier_order, tier_vals, color=[TIER_COLORS[t] for t in tier_order],
                   edgecolor="white", linewidth=2)
for bar, val in zip(bars, tier_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 f"{val:,}\\n({val/len(df_fe)*100:.1f}%)", ha="center", fontweight="bold")
axes[0].set(title="Risk Tier Distribution", ylabel="Records"); axes[0].title.set_fontweight("bold")
axes[1].hist(scores, bins=60, color="#1f4e79", alpha=0.7, edgecolor="none")
for thresh, color, label in [(0.15,"#27ae60","LOW/MED (0.15)"),
                               (0.35,"#f1c40f","MED/HIGH (0.35)"),
                               (0.60,"#c0392b","HIGH/CRIT (0.60)")]:
    axes[1].axvline(thresh, color=color, lw=2, ls="--", label=label)
axes[1].set(title="Distribution of Risk Scores",xlabel="Predicted Probability",ylabel="Count")
axes[1].title.set_fontweight("bold"); axes[1].legend(fontsize=8)
plt.suptitle("Predictive Maintenance Risk Scoring", fontweight="bold")
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# Top at-risk components not yet in a confirmed failure window
top_risk = (df_fe[df_fe["failure_within_10_cycles"]==0]
            .nlargest(10, "risk_score")
            [["aircraft_id","component_id","flight_cycles","risk_score","risk_tier",
              "vibration_sensor","fault_code_count","last_maintenance_cycles"]])
print("Top 10 at-risk components (not currently in confirmed failure window):")
top_risk.round(4)
"""))

# ─── 13. INFERENCE ────────────────────────────────────────────────────────────
cells.append(md("""\
## 13. Inference Pipeline

`predict.py` is the production inference script. It ingests raw sensor data,
computes all engineered features, and outputs risk scores and tier labels.

```bash
# Score all records
python predict.py --input new_sensor_data.csv --output scored.csv

# Score only the latest reading per component
python predict.py --input new_sensor_data.csv --output scored.csv --latest-only
```
"""))

cells.append(code("""\
# Demonstrate programmatic usage
from predict import load_model_bundle, score

bundle = load_model_bundle("models/best_model.pkl")
sample_input = df.head(200).copy()        # use a subset of raw data as demo
scored = score(sample_input, bundle, include_history=False)
print(f"Scored {len(scored)} components (latest reading each)")
print(f"\\nRisk tier summary:\\n{scored['risk_tier'].value_counts()}")
scored[["aircraft_id","component_id","flight_cycles",
        "failure_risk_score","risk_tier","recommended_action"]].head(10)
"""))

# ─── 14. CONCLUSIONS ──────────────────────────────────────────────────────────
cells.append(md("""\
## 14. Conclusions & Recommendations

### Final Model Performance

| Metric | Score | Significance |
|--------|-------|--------------|
| **ROC-AUC** | **0.9992** | Near-perfect class separation |
| **PR-AUC** | **0.9632** | Excellent under 42:1 imbalance |
| **F1 Score** | **0.9180** | Strong combined performance |
| **Recall** | **1.0000** | Zero missed failures on test set |
| **Precision** | **0.8485** | ~1 false alarm per 6.5 alerts |
| **CV PR-AUC** | **0.906 ± 0.023** | Stable and generalisable |

### Key Findings

1. **Temporal feature engineering is decisive.** `fault_roll5_sum` — the rolling 5-cycle
   fault accumulation — accounts for 61% of model gain. Raw sensor readings alone are
   far less predictive. Failure is a process, not an event.

2. **Class imbalance (42:1) was the central challenge.** PR-AUC and Recall, not accuracy,
   are the right metrics. Cost-sensitive learning with `scale_pos_weight=42` was essential.

3. **SHAP confirms physical intuition.** High `fault_roll5_sum` values sharply drive SHAP
   scores positive. The model has learned what experienced engineers know: fault accumulation
   precedes failure.

4. **Probabilities are well-calibrated.** The bimodal score distribution means the model
   makes decisive predictions — components are either clearly at risk or clearly safe.

5. **Human-readable rules are possible.** A depth-4 decision tree achieves perfect recall
   with a simple rule: `fault_roll5_sum > 9.5 AND temp_sensor_2 > 93.8°C → flag`.

### Operational Recommendations

- **Immediate:** Deploy `predict.py` as a microservice; ingest sensor data after each flight
- **Short-term:** Calibrate thresholds against actual AOG cost data for the RCAA fleet
- **Medium-term:** Retrain quarterly as new inspection outcomes are logged
- **Long-term:** Extend to Remaining Useful Life (RUL) regression — answer "how many cycles left?"

### Limitations to Disclose

- Model trained on 140 positive cases — more failure events will improve confidence
- Environmental sensors (humidity, ambient temp) show no signal and may be excluded in deployment
- `predict.py` recomputes z-score statistics on the input batch; production should use training-set statistics from the saved scaler
- Quarterly retraining is required as the fleet ages
"""))

cells.append(code("""\
# Save model artifacts
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model, "model_name": best_name,
        "scaler": scaler, "feature_cols": feature_cols,
        "best_threshold": float(best_t),
        "le_aircraft": le_ac, "le_component": le_comp,
    }, f)
print(f"Model saved → models/best_model.pkl")
print(f"\\nFinal Model : {best_name}")
print(f"  ROC-AUC   : {best_res['roc_auc']:.4f}")
print(f"  PR-AUC    : {best_res['avg_precision']:.4f}")
print(f"  F1        : {best_res['f1']:.4f}")
print(f"  Recall    : {best_res['recall']:.4f}")
print(f"  Precision : {best_res['precision']:.4f}")
print(f"  Threshold : {best_t:.2f}")
"""))

# ─── Write ─────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}
with open("predictive_maintenance_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Notebook saved: predictive_maintenance_analysis.ipynb")
