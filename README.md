# Predictive Maintenance for Aircraft Components
### Rwanda Civil Aviation Authority — Data Science Technical Assessment

**Objective:** Predict whether an aircraft component will fail within the next **10 flight cycles**, enabling maintenance teams to act before failures occur rather than after.

---

## Results at a Glance

| Metric | Score | Interpretation |
|--------|-------|---------------|
| **ROC-AUC** | **0.9993** | Near-perfect class separation |
| **PR-AUC** | **0.9687** | Excellent precision-recall trade-off under imbalance |
| **F1 Score** | **0.9333** | Strong balanced performance |
| **Recall** | **1.0000** | Zero missed failures on the test set |
| **Precision** | **0.8750** | ~1 false alarm per 7 alerts |
| **CV PR-AUC** | **0.915 ± 0.026** | Stable and generalisable across folds |

**Best model:** XGBoost with cost-sensitive learning (`scale_pos_weight = 42`)

---

## Project Structure

```
.
├── predictive_maintenance_analysis.ipynb   # Main deliverable — full end-to-end notebook
├── RCAA_Predictive_Maintenance_Report.pdf  # 8-section written report
├── RCAA_Predictive_Maintenance_Slides.pptx # 12-slide presentation (10–15 min)
│
├── analysis.py                             # Standalone reproducible analysis script
├── create_notebook.py                      # Generates the .ipynb from source
├── generate_report.py                      # Generates the PDF report (ReportLab)
├── generate_slides.py                      # Generates the PPTX slides (python-pptx)
│
├── aircraft_maintenance_dataset.csv        # Input dataset (6,000 records)
├── risk_scores_output.csv                  # Per-record failure risk scores and tiers
├── model_metrics.csv                       # Comparison table across all models
├── feature_importance.csv                  # XGBoost feature importances (ranked)
├── data_summary.csv                        # Descriptive statistics for all features
├── final_results_summary.json              # Model metrics and best threshold (JSON)
│
├── figures/                                # All charts generated during analysis
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_matrix.png
│   ├── 04_failure_by_component.png
│   ├── 05_boxplots_by_failure.png
│   ├── 06_sensor_trends.png
│   ├── 07_roc_pr_curves.png
│   ├── 08_confusion_matrices.png
│   ├── 09_metrics_comparison.png
│   ├── 10_feature_importance.png
│   ├── 12_threshold_optimization.png
│   └── 13_risk_scoring.png
│
└── models/
    └── best_model.pkl                      # Saved XGBoost model + scaler + encoders
```

---

## Dataset

| Property | Value |
|----------|-------|
| Records | 6,000 (after dropping 1 null-target row) |
| Aircraft | 20 unique IDs |
| Component types | 4 — Engine1, Engine2, Wing, LandingGear |
| Failure records | 140 (2.33%) |
| Class imbalance | **42:1** |
| Raw input features | 13 sensors + operational flags |
| Engineered features | 43 total (30 new features added) |

**Raw features:** `flight_cycles`, `engine_hours`, `temperature_sensor_1`, `temperature_sensor_2`, `vibration_sensor`, `pressure_sensor`, `fault_code_count`, `last_maintenance_cycles`, `maintenance_log_flag`, `sensor_drift_flag`, `ambient_temperature`, `humidity`

**Target:** `failure_within_10_cycles` (binary: 0 = no failure, 1 = failure imminent)

---

## Approach

### 1. Exploratory Data Analysis
- Assessed class distribution, missing values, and feature ranges
- Identified `fault_code_count` and `vibration_sensor` as the strongest raw discriminators
- Found `ambient_temperature` and `humidity` have negligible predictive signal
- Confirmed failure events cluster in later flight cycles and in Engine1 / Wing components

### 2. Feature Engineering
The most impactful step. All rolling features are computed **per aircraft-component pair** (sorted by `flight_cycles`) to prevent data leakage.

| Feature Group | Description |
|---------------|-------------|
| Rolling mean (3 & 5 cycles) | Trend direction for each sensor |
| Rolling std dev (3 cycles) | Volatility — instability signal |
| First difference (Δ) | Rate of change — sudden spikes |
| Cumulative fault count | Lifetime accumulated wear |
| **Rolling fault sum (5-cycle)** | **Top predictor — 68% of XGBoost gain** |
| Engine hours / cycle | Efficiency degradation proxy |
| Temperature differential | Asymmetric thermal stress |
| Composite stress index | Vibration + pressure z-score combined |
| Maintenance urgency | Fault count × time since last service |
| Interaction features | Vibration × temperature, drift × faults |

### 3. Handling Class Imbalance
Three strategies were applied:
- **Cost-sensitive learning** — `scale_pos_weight = 42` in XGBoost
- **Stratified K-Fold cross-validation** (k=5) — preserves class ratio in every fold
- **PR-AUC as the primary metric** — more informative than ROC-AUC at extreme imbalance ratios

### 4. Model Training & Comparison

| Model | ROC-AUC | PR-AUC | F1 | Recall | CV PR-AUC |
|-------|---------|--------|----|--------|-----------|
| Logistic Regression | 0.9832 | 0.5573 | 0.5106 | 0.8571 | 0.541 ± 0.099 |
| Random Forest | 0.9974 | 0.8752 | 0.8254 | 0.9286 | 0.843 ± 0.103 |
| LightGBM | 0.9986 | 0.9369 | 0.8621 | 0.8929 | 0.906 ± 0.026 |
| **XGBoost** | **0.9993** | **0.9687** | **0.9333** | **1.0000** | **0.915 ± 0.026** |

XGBoost outperforms alternatives because gradient boosting progressively focuses on hard-to-classify minority samples, and its regularisation (`subsample=0.8`, `colsample_bytree=0.8`) prevents overfitting to the small positive class.

### 5. Threshold Optimisation
The default 0.5 threshold is not optimal for imbalanced safety-critical problems. The optimal threshold (max F1) is **0.58**, achieving Recall = 1.00 and Precision = 0.903.

| Context | Threshold | Trade-off |
|---------|-----------|-----------|
| Safety-critical (long-haul) | 0.20–0.30 | Maximum recall; more inspections |
| Balanced operations | 0.50–0.58 | Optimal F1 |
| Cost-optimised | 0.65–0.75 | Fewer false alarms; accept some miss risk |

### 6. Maintenance Decision Support
Risk scores are mapped to a 4-tier action framework:

| Tier | Score | Recommended Action |
|------|-------|--------------------|
| 🔴 CRITICAL | ≥ 0.60 | Ground aircraft — schedule urgent inspection |
| 🟠 HIGH | 0.35–0.59 | Flag for next maintenance window |
| 🟡 MEDIUM | 0.15–0.34 | Increase monitoring frequency |
| 🟢 LOW | < 0.15 | Standard operations — routine monitoring |

---

## How to Reproduce

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm imbalanced-learn shap reportlab python-pptx
```

Python 3.8+ required.

### Run the full analysis

```bash
# Generates all figures, model metrics, risk scores, and saves the trained model
python analysis.py
```

### Regenerate deliverables

```bash
# Jupyter notebook
python create_notebook.py

# PDF report
python generate_report.py

# PowerPoint slides
python generate_slides.py
```

### Load the saved model for inference

```python
import pickle
import pandas as pd

with open("models/best_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model      = artifact["model"]           # Trained XGBoost classifier
scaler     = artifact["scaler"]          # StandardScaler (for LR only)
features   = artifact["feature_cols"]   # List of 43 feature names
threshold  = artifact["best_threshold"] # 0.58 — optimal decision threshold

# Score new data (after computing the same engineered features)
risk_score = model.predict_proba(X_new)[: , 1]
prediction = (risk_score >= threshold).astype(int)
```

---

## Key Findings

1. **Temporal feature engineering was decisive.** `fault_roll5_sum` (rolling 5-cycle fault accumulation) alone accounts for 68% of the XGBoost model's predictive gain. This validates the physical intuition that failure is preceded by escalating fault activity — not a single catastrophic sensor reading.

2. **Class imbalance (42:1) is the central challenge.** Standard accuracy is meaningless (predicting "no failure" always scores 97.7%). PR-AUC and Recall are the appropriate metrics for this problem.

3. **Environmental sensors add minimal value.** `ambient_temperature` and `humidity` showed no meaningful separation between failure and non-failure classes in this dataset. They can be excluded in deployment without significant performance loss.

4. **XGBoost achieves zero missed failures** on the test set at threshold 0.58, with a manageable false alarm rate (1 in 8 alerts is a false positive).

---

## Limitations & Assumptions

- Rolling features assume records within each aircraft-component pair are chronologically ordered by `flight_cycles`.
- `last_maintenance_cycles = 0` is treated as a freshly maintained component, not as missing data.
- Cumulative features use all prior history; in production, these must be computed from historical records only.
- The dataset appears synthetic — real-world deployment should validate sensor calibration and label quality before relying on model outputs.
- Model should be retrained quarterly as the fleet ages and usage patterns evolve.

---

## Deliverables Summary

| Deliverable | File | Description |
|-------------|------|-------------|
| Code | `predictive_maintenance_analysis.ipynb` | Complete, documented notebook |
| Report | `RCAA_Predictive_Maintenance_Report.pdf` | 8-section written report |
| Presentation | `RCAA_Predictive_Maintenance_Slides.pptx` | 12 slides, 10–15 min |

---

*Rwanda Civil Aviation Authority — Data Science Assessment | March 2026*
