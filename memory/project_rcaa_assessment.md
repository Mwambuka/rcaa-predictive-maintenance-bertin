---
name: RCAA Predictive Maintenance Assessment
description: Take-home data science task for Rwanda Civil Aviation Authority — aircraft component failure prediction
type: project
---

Completed full RCAA predictive maintenance take-home assessment. Dataset: aircraft_maintenance_dataset.csv (6,000 records, 20 aircraft, 4 components, 42:1 class imbalance).

Best model: XGBoost — ROC-AUC=0.9993, PR-AUC=0.9687, Recall=1.00, F1=0.9333.
Key finding: fault_roll5_sum (rolling 5-cycle fault accumulation) accounts for 68% of XGBoost gain.

**Why:** RCAA job application assessment requiring code, PDF report, and presentation slides.
**How to apply:** Deliverables are complete. If asked to update/revise, all generator scripts are in the project root.

Deliverables:
- `predictive_maintenance_analysis.ipynb` — Jupyter notebook (main code deliverable)
- `RCAA_Predictive_Maintenance_Report.pdf` — 8-section PDF report
- `RCAA_Predictive_Maintenance_Slides.pptx` — 12-slide presentation
- `analysis.py` — standalone script (generates all figures, models, CSVs)
- `figures/` — 12 PNG figures
- `models/best_model.pkl` — saved XGBoost model + scaler
- `risk_scores_output.csv` — per-record risk scores and tiers
