# Predictive Maintenance for Aircraft Components
### Rwanda Civil Aviation Authority — Data Science Technical Assessment

**Objective:** Predict whether an aircraft component will fail within the next **10 flight cycles**, enabling maintenance teams to act before failures occur rather than after.

---

## Results at a Glance

| Metric | Score | What it means |
|--------|-------|---------------|
| **ROC-AUC** | **0.9992** | Near-perfect class separation across all thresholds |
| **PR-AUC** | **0.9632** | Excellent precision-recall trade-off under 42:1 imbalance |
| **F1 Score** | **0.9180** | Strong balanced performance at optimal threshold |
| **Recall** | **1.0000** | Zero missed failures on the held-out test set |
| **Precision** | **0.8485** | ~1 false alarm per 6.5 alerts |
| **CV PR-AUC** | **0.906 ± 0.023** | Stable, generalisable — not an artefact of one split |

**Best model:** XGBoost with `scale_pos_weight=42` (cost-sensitive learning)
**Key finding:** `fault_roll5_sum` (rolling 5-cycle fault accumulation) accounts for **61% of model gain**

---

## Deliverables

| File | Description |
|------|-------------|
| `predictive_maintenance_analysis.ipynb` | Main code deliverable — complete, documented notebook |
| `RCAA_Predictive_Maintenance_Report.pdf` | 9-section PDF report (incl. Executive Summary) |
| `RCAA_Predictive_Maintenance_Slides.pptx` | 12-slide presentation (10–15 min) |

---

## Project Structure

```
.
├── predictive_maintenance_analysis.ipynb   # Main deliverable — notebook
├── RCAA_Predictive_Maintenance_Report.pdf  # Full written report
├── RCAA_Predictive_Maintenance_Slides.pptx # Presentation slides
│
├── analysis.py          # Modular end-to-end analysis script (run this to reproduce everything)
├── predict.py           # Production inference pipeline — score new sensor data
├── create_notebook.py   # Generates the .ipynb from source
├── generate_report.py   # Generates the PDF (ReportLab)
├── generate_slides.py   # Generates the PPTX (python-pptx)
├── requirements.txt     # Python dependencies
│
├── aircraft_maintenance_dataset.csv   # Input dataset (6,000 records)
├── risk_scores_output.csv             # Per-record failure risk scores and tiers
├── decision_tree_rules.txt            # Human-readable rules for maintenance engineers
├── model_metrics.csv                  # Comparison table across all four models
├── feature_importance.csv             # XGBoost feature importances (ranked)
├── data_summary.csv                   # Descriptive statistics
├── final_results_summary.json         # Model metrics and best threshold (JSON)
│
├── figures/                           # 16 charts generated during analysis
│   ├── 01_target_distribution.png     # Class imbalance visualisation
│   ├── 02_feature_distributions.png   # Sensor distributions by class
│   ├── 03_correlation_matrix.png      # Feature correlations
│   ├── 04_failure_by_component.png    # Failure rates by component type
│   ├── 05_boxplots_by_failure.png     # Sensor readings: failure vs. no-failure
│   ├── 06_sensor_trends.png           # Sensor trajectories over flight cycles
│   ├── 07_roc_pr_curves.png           # ROC and PR curves for all models
│   ├── 08_confusion_matrices.png      # Confusion matrices for all models
│   ├── 09_metrics_comparison.png      # Side-by-side metric bar chart
│   ├── 10_feature_importance.png      # XGBoost feature gain importances
│   ├── 11_shap_summary.png            # SHAP beeswarm + mean |SHAP| bar
│   ├── 12_threshold_optimization.png  # Precision/Recall/F1 vs threshold
│   ├── 13_risk_scoring.png            # Risk tier distribution + score histogram
│   ├── 14_learning_curves.png         # Training vs validation PR-AUC curves
│   ├── 15_calibration_curve.png       # Reliability diagram (probability calibration)
│   └── 16_shap_dependence.png         # SHAP dependence plots for top two features
│
└── models/
    └── best_model.pkl                 # Saved model bundle (model + scaler + encoders)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Records | 6,000 (1 null-target row dropped) |
| Aircraft | 20 unique IDs |
| Component types | 4 — Engine1, Engine2, Wing, LandingGear |
| Failure records | 140 (2.33%) |
| Class imbalance | **42:1** |
| Raw input features | 13 sensors + operational flags |
| Engineered features | 43 total |
| Data anomalies | 1 negative humidity; 28 humidity > 100%; 7 extreme ambient temps |

---

## Approach

### 1. Data Quality Assessment
Checked for missing values (1 row — dropped), duplicates (0), outliers (humidity values outside 0–100%, extreme ambient temperatures). All anomalies documented; none imputed.

### 2. Exploratory Data Analysis
- `fault_code_count` and `vibration_sensor` are the strongest raw discriminators
- `ambient_temperature` and `humidity` show no meaningful signal — they can be excluded in deployment
- Failure events cluster in later flight cycles and concentrate in Engine1 and Wing components
- `sensor_drift_flag` is rare but correlates with elevated fault counts

### 3. Feature Engineering
All rolling operations computed **per (aircraft_id, component_id) pair, sorted by flight_cycles** — no look-ahead leakage.

| Feature Group | Key feature | Rationale |
|---|---|---|
| Rolling mean (3 & 5 cycles) | `*_roll3_mean`, `*_roll5_mean` | Trend direction; noise smoothing |
| Rolling std dev | `*_roll3_std` | Volatility — instability pre-failure |
| First difference | `*_diff1` | Rate of change — deterioration speed |
| **Rolling fault sum** | **`fault_roll5_sum`** ★ | **#1 predictor — 61% of model gain** |
| Cumulative faults | `cumulative_faults` | Total lifetime wear |
| Composite stress index | `stress_index` | Vib + pressure z-score combined |
| Maintenance urgency | `maint_urgency` | Faults × time since last service |
| Interactions | `vib_x_temp1`, `drift_x_vib` | Non-linear combined effects |

### 4. Class Imbalance Handling
Three strategies combined:
- **`scale_pos_weight = 42`** — penalises minority-class misclassification 42× in the loss function
- **Stratified K-Fold CV (k=5)** — preserves class ratio across all folds
- **PR-AUC as primary metric** — unaffected by the large number of true negatives

### 5. Model Comparison

| Model | ROC-AUC | PR-AUC | F1 | Recall | CV PR-AUC |
|-------|---------|--------|----|--------|-----------|
| Logistic Regression | 0.9832 | 0.5573 | 0.5106 | 0.8571 | 0.541 ± 0.099 |
| Random Forest | 0.9974 | 0.8695 | 0.8438 | 0.9643 | 0.841 ± 0.113 |
| LightGBM | 0.9987 | 0.9361 | 0.9153 | 0.9643 | 0.901 ± 0.029 |
| **XGBoost** ★ | **0.9992** | **0.9632** | **0.9180** | **1.0000** | **0.906 ± 0.023** |

### 6. Explainability (Three Layers)
1. **XGBoost feature importance** (gain) — ranked feature contributions
2. **SHAP values** (via XGBoost native `pred_contribs`) — per-prediction attribution
3. **Decision tree rules** (`decision_tree_rules.txt`) — human-readable rules for engineers:
   ```
   IF fault_roll5_sum > 9.5 AND temp_sensor_2_5cycle_avg > 93.8°C → FLAG FOR INSPECTION
   ```

### 7. Threshold Optimisation

| Context | Threshold | Trade-off |
|---------|-----------|-----------|
| Safety-critical (long-haul) | 0.20–0.30 | Maximum recall; more inspections |
| Balanced operations (default) | 0.50–0.68 | Optimal F1 |
| Cost-optimised (backup fleet) | 0.70–0.80 | Fewer alerts; marginal miss risk |

### 8. Risk Scoring System

| Tier | Score | Action |
|------|-------|--------|
| 🔴 CRITICAL | ≥ 0.60 | Ground aircraft — urgent inspection |
| 🟠 HIGH | 0.35–0.59 | Flag for next maintenance window |
| 🟡 MEDIUM | 0.15–0.34 | Increase monitoring frequency |
| 🟢 LOW | < 0.15 | Standard operations |

---

## How to Reproduce

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full analysis

```bash
python analysis.py
# Generates all 16 figures, model metrics, SHAP values, decision rules,
# learning curves, calibration curve, risk scores, and saves the model bundle.
```

### Regenerate deliverables

```bash
python create_notebook.py          # Jupyter notebook
python generate_report.py          # PDF report
python generate_slides.py          # PowerPoint slides
```

### Score new sensor data

```bash
# Score all records and output risk tiers
python predict.py --input new_data.csv --output scored.csv

# Score only the latest reading per component
python predict.py --input new_data.csv --output scored.csv --latest-only
```

Input CSV must follow the same schema as `aircraft_maintenance_dataset.csv`.

### Load the model programmatically

```python
import pickle
import pandas as pd
from predict import load_model_bundle, score

bundle = load_model_bundle("models/best_model.pkl")
df_raw = pd.read_csv("new_sensor_data.csv")

result = score(df_raw, bundle, include_history=False)
print(result[["aircraft_id", "component_id", "failure_risk_score", "risk_tier"]])
```

---

## Key Findings

1. **Temporal feature engineering is the decisive factor.** The rolling 5-cycle fault accumulation feature (`fault_roll5_sum`) drives 61% of the XGBoost model's gain. Raw sensor readings are nearly useless without temporal context.

2. **Class imbalance (42:1) is the core challenge.** Standard accuracy (97.7% from always predicting "no failure") is meaningless. PR-AUC and Recall are the right metrics for this problem.

3. **SHAP confirms physical intuition.** High `fault_roll5_sum` values push SHAP scores sharply positive — the model has learned that rapid fault accumulation is the primary failure precursor. This is consistent with component wear mechanics.

4. **Calibration is strong.** XGBoost and LightGBM produce well-calibrated probabilities (close to the reliability diagram diagonal), meaning the risk tiers are statistically trustworthy, not just ranked.

5. **Environmental sensors add no value.** `ambient_temperature` and `humidity` can be excluded in production without performance loss — reducing sensor dependencies for deployment.

---

## Limitations & Assumptions

- Rolling features assume records within each aircraft-component pair are chronologically ordered by `flight_cycles`.
- `last_maintenance_cycles = 0` is treated as a freshly maintained component. In a real system, this should be validated against maintenance records.
- Z-score normalisations in `predict.py` are computed on the input batch. For strict consistency, these should use training-set statistics from the saved scaler.
- The dataset is synthetic in characteristics (humidity > 100%, extreme temperatures). Real-world deployment requires data quality validation at ingestion.
- Model performance is based on 140 positive cases. Collecting more failure events will improve both performance and confidence intervals.
- Quarterly retraining is recommended as fleet age and usage patterns evolve.

---

*Rwanda Civil Aviation Authority — Data Science Assessment | March 2026*
