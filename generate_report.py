"""
Generates the PDF Report for RCAA Predictive Maintenance Assessment.
Sections: Executive Summary, Problem, EDA, Feature Engineering,
          Model Selection, Explainability, Threshold, Decision Support,
          Challenges & Conclusions.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import json, os

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1f4e79")
MED_BLUE   = colors.HexColor("#2980b9")
LIGHT_BLUE = colors.HexColor("#d6e4f0")
RED        = colors.HexColor("#c0392b")
GREEN      = colors.HexColor("#27ae60")
ORANGE     = colors.HexColor("#e67e22")
GRAY       = colors.HexColor("#95a5a6")
LIGHT_GRAY = colors.HexColor("#f5f6fa")

FIGURES = "figures"
OUTPUT  = "RCAA_Predictive_Maintenance_Report.pdf"

with open("final_results_summary.json") as f:
    res = json.load(f)
best_name = res["best_model"]
best      = res["models"][best_name]

# ── Page template ─────────────────────────────────────────────────────────────
def header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    w, h = A4
    canvas_obj.setFillColor(DARK_BLUE)
    canvas_obj.rect(0, h - 1.5*cm, w, 1.5*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 9)
    canvas_obj.drawString(1.5*cm, h - 1.0*cm,
                          "RCAA — Predictive Maintenance Assessment")
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(w - 1.5*cm, h - 1.0*cm, "Confidential")
    canvas_obj.setStrokeColor(DARK_BLUE)
    canvas_obj.setLineWidth(1)
    canvas_obj.line(1.5*cm, 1.5*cm, w - 1.5*cm, 1.5*cm)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.setFillColor(GRAY)
    canvas_obj.drawString(1.5*cm, 0.8*cm, "Rwanda Civil Aviation Authority")
    canvas_obj.drawRightString(w - 1.5*cm, 0.8*cm, f"Page {doc.page}")
    canvas_obj.restoreState()

# ── Styles ────────────────────────────────────────────────────────────────────
body = ParagraphStyle("body", fontSize=9.5, leading=15, spaceAfter=6,
                      textColor=colors.HexColor("#2c3e50"), alignment=TA_JUSTIFY)
h2   = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                      textColor=DARK_BLUE, spaceBefore=10, spaceAfter=5)
h3   = ParagraphStyle("h3", fontSize=11, fontName="Helvetica-Bold",
                      textColor=MED_BLUE,  spaceBefore=8,  spaceAfter=4)
caption = ParagraphStyle("caption", fontSize=8, textColor=GRAY,
                         alignment=TA_CENTER, spaceAfter=10,
                         fontName="Helvetica-Oblique")
bullet  = ParagraphStyle("bullet", parent=body, leftIndent=14,
                         bulletIndent=4, spaceBefore=2, spaceAfter=2)
exec_body = ParagraphStyle("exec_body", fontSize=10, leading=16,
                            textColor=colors.HexColor("#1a252f"), alignment=TA_JUSTIFY)

def section_bar(title):
    return Table(
        [[Paragraph(f"  {title}",
                    ParagraphStyle("sb", fontSize=13, fontName="Helvetica-Bold",
                                   textColor=colors.white, leading=16))]],
        colWidths=[17*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ]),
    )

def metric_box(label, value, color=DARK_BLUE):
    return Table(
        [[Paragraph(str(value),
                    ParagraphStyle("mv", fontSize=18, fontName="Helvetica-Bold",
                                   textColor=color, alignment=TA_CENTER))],
         [Paragraph(label,
                    ParagraphStyle("ml", fontSize=7.5, textColor=GRAY,
                                   alignment=TA_CENTER))]],
        colWidths=[3.9*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), LIGHT_GRAY),
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
            ("BOX",        (0,0),(-1,-1), 1, colors.HexColor("#dce6f2")),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ]),
    )

def add_figure(path, width=15*cm, caption_text=None):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width*0.6))
        if caption_text:
            items.append(Paragraph(caption_text, caption))
    return items

def data_table(headers, rows, col_widths):
    data = [headers] + rows
    style = TableStyle([
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0),  DARK_BLUE),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT_BLUE, colors.white]),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#aed6f1")),
    ])
    return Table(data, colWidths=col_widths, style=style)

# ── Build document ────────────────────────────────────────────────────────────
doc   = SimpleDocTemplate(OUTPUT, pagesize=A4,
                          leftMargin=1.5*cm, rightMargin=1.5*cm,
                          topMargin=2.2*cm, bottomMargin=2.0*cm)
story = []

# ════════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════════
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph(
    "Predictive Maintenance for Aircraft Components",
    ParagraphStyle("cov1", fontSize=24, fontName="Helvetica-Bold",
                   textColor=DARK_BLUE, alignment=TA_CENTER, leading=30)))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Data Science Technical Assessment",
    ParagraphStyle("cov2", fontSize=15, textColor=MED_BLUE,
                   alignment=TA_CENTER)))
story.append(Spacer(1, 0.3*cm))
story.append(HRFlowable(width="80%", thickness=2, color=DARK_BLUE, hAlign="CENTER"))
story.append(Spacer(1, 0.6*cm))

metrics_row = Table(
    [[metric_box("ROC-AUC",    f"{best['roc_auc']:.4f}",        DARK_BLUE),
      metric_box("PR-AUC",     f"{best['avg_precision']:.4f}",  MED_BLUE),
      metric_box("F1 Score",   f"{best['f1']:.4f}",             GREEN),
      metric_box("Recall",     f"{best['recall']:.4f}",         RED)]],
    colWidths=[4.1*cm]*4,
    style=TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]),
)
story.append(metrics_row)
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(f"Best Model: <b>{best_name}</b>",
    ParagraphStyle("bm", fontSize=11, textColor=DARK_BLUE, alignment=TA_CENTER)))
story.append(Spacer(1, 1.2*cm))

cover_info = Table(
    [["Submitted to:", "Rwanda Civil Aviation Authority (RCAA)"],
     ["Role:",         "Data Scientist"],
     ["Date:",         "March 2026"],
     ["Dataset:",      "aircraft_maintenance_dataset.csv  |  6,000 records"],
     ["Task:",         "Binary classification — failure within next 10 flight cycles"]],
    colWidths=[4*cm, 12*cm],
    style=TableStyle([
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1),9.5),
        ("TEXTCOLOR",     (0,0),(0,-1), DARK_BLUE),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT_BLUE, colors.white]),
        ("TOPPADDING",    (0,0),(-1,-1),6),
        ("BOTTOMPADDING", (0,0),(-1,-1),6),
        ("LEFTPADDING",   (0,0),(-1,-1),8),
    ]),
)
story.append(cover_info)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════
story.append(section_bar("Executive Summary"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "This report presents a complete predictive maintenance solution for aircraft components, "
    "developed as part of the Rwanda Civil Aviation Authority data science assessment. "
    "The goal is to predict which components will fail within the next 10 flight cycles, "
    "giving maintenance teams actionable advance warning before failures become unplanned events.",
    exec_body))
story.append(Spacer(1, 0.25*cm))
story.append(Paragraph("<b>The Problem.</b>  "
    "Unplanned aircraft component failures cause Aircraft-on-Ground (AOG) events — "
    "one of the most costly outcomes in commercial aviation. They also carry direct "
    "safety implications. At the same time, over-maintaining components that are not "
    "at risk wastes engineering resources. Predictive maintenance solves both problems "
    "by targeting inspections precisely where and when they are needed.",
    exec_body))
story.append(Spacer(1, 0.25*cm))
story.append(Paragraph("<b>The Data.</b>  "
    "A dataset of 6,000 records across 20 aircraft and 4 component types was provided. "
    "Only 140 records (2.3%) represent failure events — a 42:1 class imbalance that "
    "makes this a fundamentally different problem from standard classification tasks. "
    "Naive accuracy is meaningless here; a model predicting 'no failure' for every record "
    "would score 97.7% — and catch zero failures.",
    exec_body))
story.append(Spacer(1, 0.25*cm))
story.append(Paragraph("<b>The Approach.</b>  "
    "The analysis follows a rigorous five-step process: data quality assessment, "
    "exploratory data analysis, temporal feature engineering (transforming static sensor "
    "snapshots into time-aware signals), cost-sensitive model training, and threshold "
    "optimisation aligned to aviation safety requirements. Four models were trained and "
    "compared; XGBoost was selected as the best performer.",
    exec_body))
story.append(Spacer(1, 0.25*cm))
story.append(Paragraph("<b>The Result.</b>  "
    "XGBoost achieves <b>PR-AUC = 0.9632</b> and <b>Recall = 1.00</b> on the held-out "
    "test set — meaning zero failure events are missed. The model produces a continuous "
    "risk score (0–1) for every component, mapped to a four-tier action framework "
    "(CRITICAL / HIGH / MEDIUM / LOW) that maintenance planners can act on directly.",
    exec_body))
story.append(Spacer(1, 0.25*cm))
story.append(Paragraph("<b>The Key Finding.</b>  "
    "The single most important predictor is <i>fault_roll5_sum</i> — the rolling "
    "5-cycle sum of fault codes. This engineered feature alone accounts for 61% of "
    "the model's predictive power. It captures the physical reality that component "
    "failure is not caused by one bad sensor reading, but by escalating fault activity "
    "over time. No raw sensor in the dataset had comparable predictive value. "
    "This insight has immediate practical value: maintenance engineers should prioritise "
    "components showing rapid fault code accumulation, regardless of any single-cycle reading.",
    exec_body))
story.append(Spacer(1, 0.25*cm))

exec_table = Table(
    [["Dimension",        "Assessment"],
     ["Data quality",     "Good — 1 null row removed; 36 sensor anomalies flagged but not imputed"],
     ["Feature engineering", "Critical — rolling fault accumulation added 61% of predictive gain"],
     ["Model performance","Excellent — PR-AUC 0.963, Recall 1.00 at optimal threshold"],
     ["Explainability",   "SHAP values confirm fault accumulation dominates; results align with physics"],
     ["Calibration",      "Probabilities are well-separated; bimodal distribution supports risk tiering"],
     ["Deployment readiness", "Inference pipeline (predict.py) ready; quarterly retraining recommended"]],
    colWidths=[5.5*cm, 11.5*cm],
    style=TableStyle([
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0), DARK_BLUE),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT_BLUE, colors.white]),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#aed6f1")),
    ]),
)
story.append(exec_table)
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 1. PROBLEM UNDERSTANDING
# ════════════════════════════════════════════════════════════════
story.append(section_bar("1. Problem Understanding"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Airlines maintain aircraft to two standards: scheduled maintenance (calendar or cycle-based) "
    "and condition-based maintenance (triggered by observed degradation). Predictive maintenance "
    "goes one step further — using sensor and operational data to anticipate which components are "
    "approaching failure <i>before</i> any observable symptom appears.",
    body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Business Objectives", h3))
story.append(data_table(
    ["Objective", "Success Metric", "Priority"],
    [["Catch every failure before it occurs",   "Recall → 1.00",          "Critical — safety"],
     ["Minimise false maintenance alerts",       "Precision → high",       "High — cost"],
     ["Support maintenance scheduling",          "Calibrated risk scores", "High — operations"],
     ["Explainable to engineers",                "SHAP + decision rules",  "Medium — adoption"],
     ["Deployable on standard infrastructure",   "Standard Python stack",  "Medium — IT"]],
    [6*cm, 5*cm, 6*cm],
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Why Class Imbalance Is the Central Challenge", h3))
story.append(Paragraph(
    "With 140 positive cases in 6,000 records (2.3%), standard accuracy is entirely misleading. "
    "A trivial model that predicts 'no failure' for every record achieves 97.7% accuracy — and "
    "catches zero failures. Three complementary strategies address this: "
    "(1) <b>cost-sensitive learning</b> — scale_pos_weight = 42 makes the model penalise "
    "missed failures 42× more than false alarms; "
    "(2) <b>PR-AUC as the primary metric</b> — focuses on the minority class and is unaffected "
    "by the large number of true negatives; "
    "(3) <b>stratified cross-validation</b> — preserves the class ratio in every fold, giving "
    "reliable variance estimates despite the imbalance.",
    body))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 2. DATA EXPLORATION
# ════════════════════════════════════════════════════════════════
story.append(section_bar("2. Data Exploration"))
story.append(Spacer(1, 0.3*cm))

story.append(data_table(
    ["Property", "Value"],
    [["Total records",       "6,000 (after dropping 1 null-target row)"],
     ["Aircraft",            "20 unique IDs"],
     ["Component types",     "4 — Engine1, Engine2, Wing, LandingGear"],
     ["Failure records",     "140  (2.33% — extreme class imbalance)"],
     ["Flight cycle range",  "1 – 412 cycles per component"],
     ["Data anomalies",      "1 negative humidity; 28 humidity > 100%; 7 extreme ambient temps"],
     ["Missing values",      "1 row with null target — dropped (not imputed)"]],
    [5*cm, 12*cm],
))
story.append(Spacer(1, 0.4*cm))

story.extend(add_figure(f"{FIGURES}/01_target_distribution.png", 13*cm,
    "Figure 1 — Target distribution. 97.7% no-failure vs 2.3% failure; "
    "standard accuracy is meaningless here."))

story.extend(add_figure(f"{FIGURES}/02_feature_distributions.png", 16*cm,
    "Figure 2 — Sensor distributions by class. Fault codes and vibration show "
    "the strongest separation; humidity and ambient temperature show none."))

story.extend(add_figure(f"{FIGURES}/05_boxplots_by_failure.png", 15*cm,
    "Figure 3 — Fault code count and vibration are the most discriminating "
    "raw features. Single-cycle readings already hint at the signal."))

story.append(Paragraph("Key EDA Findings", h3))
for finding in [
    "<b>fault_code_count</b> is the strongest single discriminator — failure records show markedly higher fault code frequency.",
    "<b>vibration_sensor</b> is elevated pre-failure, consistent with bearing or structural wear.",
    "<b>temperature_sensor_2</b> shows mild but consistent elevation in failure records — likely a secondary thermal stress indicator.",
    "<b>ambient_temperature and humidity</b> show no meaningful separation. These may not be available in all real-world deployments, and their exclusion would not degrade performance.",
    "<b>Engine1 and Wing</b> components have the highest failure rates. This pattern warrants closer inspection during operations.",
    "Failures <b>cluster in later flight cycles</b> (higher cycle counts), consistent with wear-out failure mechanics.",
    "The <b>sensor_drift_flag</b> is rare but correlates with elevated fault counts — sensor degradation may precede component degradation.",
]:
    story.append(Paragraph(f"• {finding}", bullet))

story.extend(add_figure(f"{FIGURES}/04_failure_by_component.png", 13*cm,
    "Figure 4 — Failure rates by component. Engine1 shows the highest rate; "
    "targeted monitoring protocols could be designed per component type."))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════
story.append(section_bar("3. Feature Engineering"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Raw sensor readings capture a <i>snapshot</i> of the component's current state. "
    "Failure, however, is a <i>process</i> — it emerges from the accumulation and "
    "acceleration of degradation over time. Feature engineering transforms static "
    "observations into temporal signals that the model can use to detect that process.",
    body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "<b>Anti-leakage design:</b> All rolling and cumulative features are computed "
    "per (aircraft_id, component_id) pair, sorted by flight_cycles. This ensures "
    "that each record's features only use its own component's history — never "
    "information from future cycles or other components.",
    body))
story.append(Spacer(1, 0.3*cm))

story.append(data_table(
    ["Feature Group", "Count", "Rationale"],
    [["Rolling mean (3 & 5 cycles)",  "10", "Trend direction; smooths single-cycle noise"],
     ["Rolling std dev (3 cycles)",   "5",  "Volatility — instability is a pre-failure signal"],
     ["First difference (Δ cycle)",   "5",  "Rate of change — detects sudden acceleration"],
     ["Cumulative fault count",       "1",  "Total lifetime fault accumulation on the component"],
     ["Rolling fault sum (5-cycle)",  "1",  "★ Top predictor — intensity of recent fault activity"],
     ["Engine hours / cycle",         "1",  "Efficiency degradation proxy"],
     ["Temperature differential",     "1",  "Asymmetric thermal stress between sensors"],
     ["Composite stress index",       "1",  "Vibration + pressure z-score combined"],
     ["Maintenance urgency",          "1",  "Fault count × time since last service"],
     ["Interaction features",         "3",  "Vib × Temp, Drift × Faults (non-linear effects)"],
     ["Encoded categorical IDs",      "2",  "Aircraft and component type as numeric"]],
    [5*cm, 1.8*cm, 10.2*cm],
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "<b>Result:</b> 13 raw input features expanded to <b>43 engineered features</b>. "
    "The post-analysis finding that <i>fault_roll5_sum</i> alone drives 61% of model "
    "gain validates this engineering strategy — the temporal signal was latent in the "
    "raw data but only became exploitable once expressed as a rolling window.",
    body))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 4. MODEL SELECTION & EVALUATION
# ════════════════════════════════════════════════════════════════
story.append(section_bar("4. Model Selection & Evaluation"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "Four models spanning the complexity spectrum were trained and compared, "
    "each using cost-sensitive learning to address the 42:1 class imbalance.",
    body))
story.append(Spacer(1, 0.2*cm))

story.append(data_table(
    ["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "CV PR-AUC"],
    [["Logistic Regression", "0.9832", "0.5573", "0.5106", "0.3636", "0.8571", "0.541 ± 0.099"],
     ["Random Forest",       "0.9974", "0.8695", "0.8438", "0.7917", "0.9643", "0.841 ± 0.113"],
     ["LightGBM",            "0.9987", "0.9361", "0.9153", "0.8667", "0.9643", "0.901 ± 0.029"],
     [f"{best_name} ★",      "0.9992", "0.9632", "0.9180", "0.8485", "1.0000", "0.906 ± 0.023"]],
    [4.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.0*cm, 1.8*cm, 3.3*cm],
))
story.append(Spacer(1, 0.3*cm))

story.extend(add_figure(f"{FIGURES}/07_roc_pr_curves.png", 15*cm,
    "Figure 5 — ROC and PR curves. PR-AUC is the primary metric under "
    "extreme class imbalance. XGBoost dominates on both."))

story.extend(add_figure(f"{FIGURES}/08_confusion_matrices.png", 15*cm,
    "Figure 6 — Confusion matrices at default threshold. "
    "XGBoost achieves 0 false negatives — no missed failures on the test set."))

story.append(Paragraph("Why XGBoost?", h3))
for reason in [
    "Gradient boosting builds an ensemble of trees that <b>sequentially focuses on misclassified minority samples</b>, directly addressing the imbalance.",
    "<b>scale_pos_weight = 42</b> adjusts the loss function so minority class errors are 42× more costly — not a post-hoc adjustment but baked into training.",
    "<b>Regularisation</b> (subsample=0.8, colsample_bytree=0.8) prevents overfitting to the small positive class.",
    "Logistic Regression's linear boundary cannot capture the non-linear interaction between fault accumulation, vibration, and maintenance history.",
    "XGBoost's CV variance (±0.023) is lower than Random Forest (±0.113), indicating more stable generalisation.",
]:
    story.append(Paragraph(f"• {reason}", bullet))

story.extend(add_figure(f"{FIGURES}/14_learning_curves.png", 13*cm,
    "Figure 7 — Learning curves. The gap between training and validation "
    "PR-AUC narrows as training data increases, confirming the model is not "
    "severely overfitting. Performance is data-limited for the positive class."))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 5. EXPLAINABILITY
# ════════════════════════════════════════════════════════════════
story.append(section_bar("5. Model Explainability"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "In aviation, a model that cannot explain its predictions will not be adopted by "
    "maintenance engineers. Three complementary explainability approaches are used.",
    body))
story.append(Spacer(1, 0.2*cm))

story.extend(add_figure(f"{FIGURES}/10_feature_importance.png", 13*cm,
    "Figure 8 — Feature importances (XGBoost gain). "
    "fault_roll5_sum dominates at 61% — the temporal accumulation of fault codes "
    "is by far the most important signal."))

story.extend(add_figure(f"{FIGURES}/11_shap_summary.png", 15*cm,
    "Figure 9 — SHAP beeswarm (left) and mean |SHAP| bar (right). "
    "Each dot is one test record; colour indicates feature value (red=high, blue=low). "
    "High fault_roll5_sum strongly pushes predictions towards failure."))

story.extend(add_figure(f"{FIGURES}/16_shap_dependence.png", 15*cm,
    "Figure 10 — SHAP dependence plots. "
    "The step-change in SHAP value for fault_roll5_sum > ~10 is the key non-linearity "
    "the model has learned. Below this threshold, risk is near-zero regardless of "
    "other sensors."))

story.append(Paragraph("Decision Tree Rules (Engineer-Readable)", h3))
story.append(Paragraph(
    "A shallow Decision Tree (depth=4) was trained on the same data to produce "
    "human-readable rules that maintenance crews can apply without any software. "
    "The tree achieves perfect recall on the training set:",
    body))
story.append(Spacer(1, 0.15*cm))

rules_text = (
    "IF fault_roll5_sum > 9.5:<br/>"
    "&nbsp;&nbsp;&nbsp;IF vibration_roll3_mean ≤ 1.59:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IF temp_sensor_2_roll5_mean > 93.8°C:<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ PREDICT FAILURE<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ELSE → No failure predicted<br/>"
    "&nbsp;&nbsp;&nbsp;ELSE (vibration_roll3_mean > 1.59) → No failure predicted<br/>"
    "ELSE (fault_roll5_sum ≤ 9.5) → No failure predicted"
)
story.append(Table(
    [[Paragraph(rules_text,
                ParagraphStyle("rules", fontSize=9, leading=14,
                               fontName="Courier", textColor=DARK_BLUE))]],
    colWidths=[17*cm],
    style=TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), LIGHT_BLUE),
        ("BOX",           (0,0),(-1,-1), 1, MED_BLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
    ]),
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Interpretation for engineers: a component that has accumulated more than 10 fault codes "
    "in the last 5 cycles AND is running hot (Temp Sensor 2 > 93.8°C) is in the "
    "highest-risk category and should be flagged immediately.",
    body))

story.extend(add_figure(f"{FIGURES}/15_calibration_curve.png", 15*cm,
    "Figure 11 — Calibration curves. XGBoost and LightGBM are well-calibrated "
    "(close to the diagonal), meaning the predicted probabilities reflect true "
    "failure rates. This is essential for the risk-tiering system to be trustworthy."))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 6. THRESHOLD & DECISION SUPPORT
# ════════════════════════════════════════════════════════════════
story.append(section_bar("6. Threshold Optimisation & Maintenance Decision Support"))
story.append(Spacer(1, 0.3*cm))

story.extend(add_figure(f"{FIGURES}/12_threshold_optimization.png", 13*cm,
    "Figure 12 — Threshold optimisation. "
    "Optimal F1 is achieved at 0.68. For safety-critical routes, "
    "operators should use 0.20–0.30 to maximise recall."))

story.append(data_table(
    ["Context", "Threshold", "Expected Performance"],
    [["Safety-critical (long-haul)",    "0.20–0.30", "Recall ≈ 1.00; more false alarms accepted"],
     ["Balanced operations (default)",  "0.50–0.68", "Optimal F1; balanced precision and recall"],
     ["Cost-optimised (backup fleet)",  "0.70–0.80", "Fewer alerts; marginal miss risk accepted"]],
    [6*cm, 3*cm, 8*cm],
))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Four-Tier Risk Framework", h3))
story.append(data_table(
    ["Tier", "Score Range", "Population", "Action"],
    [["🔴 CRITICAL", "≥ 0.60", "~2.4%",    "Ground aircraft — urgent inspection required"],
     ["🟠 HIGH",     "0.35–0.59", "~0.02%", "Flag for next maintenance window"],
     ["🟡 MEDIUM",   "0.15–0.34", "~0.02%", "Increase monitoring; review at next service"],
     ["🟢 LOW",      "< 0.15",  "~97.6%",   "Standard operations — routine monitoring"]],
    [2.5*cm, 3*cm, 2.5*cm, 9*cm],
))
story.append(Spacer(1, 0.3*cm))

story.extend(add_figure(f"{FIGURES}/13_risk_scoring.png", 15*cm,
    "Figure 13 — Risk tier distribution and score histogram. "
    "The strongly bimodal distribution (LOW or CRITICAL, few intermediate cases) "
    "reflects the model's confidence: when fault accumulation crosses the critical "
    "threshold, the risk is clear and decisive."))

story.append(Paragraph("Integration Pathway", h3))
for i, step in enumerate([
    "<b>Data ingestion:</b> Flight Data Management System (FDMS) exports sensor logs after every flight cycle.",
    "<b>Feature pipeline:</b> predict.py ingests raw sensor data, computes all engineered features per component, and scores each aircraft-component pair.",
    "<b>Risk dashboard:</b> Maintenance control centre receives ranked list of CRITICAL and HIGH components, with drill-down to sensor history.",
    "<b>Action dispatch:</b> CRITICAL components trigger automated work-order creation; HIGH components appear on next-shift maintenance queue.",
    "<b>Feedback loop:</b> Post-inspection outcomes are logged and used for quarterly model retraining.",
], 1):
    story.append(Paragraph(f"{i}. {step}", bullet))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 7. CHALLENGES, ASSUMPTIONS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
story.append(section_bar("7. Challenges, Assumptions & Recommendations"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Key Challenges Encountered", h3))
for c in [
    "<b>42:1 class imbalance:</b> Standard accuracy and ROC-AUC are misleading; PR-AUC and Recall were adopted as primary metrics. Three imbalance-correction strategies were combined.",
    "<b>Temporal structure in flat format:</b> Rolling and cumulative features required explicit per-group sorting by flight_cycles to avoid cross-component leakage.",
    "<b>SHAP compatibility issue:</b> XGBoost 3.2.0 + SHAP 0.49.1 incompatibility required using XGBoost's native pred_contribs API to extract SHAP values directly from the booster.",
    "<b>Small positive class (140 events):</b> Statistical confidence is limited; 5-fold stratified CV with multiple seeds was used to ensure variance estimates are reliable.",
    "<b>Sensor anomalies in environmental features:</b> Humidity values outside 0–100% and extreme ambient temperatures suggest measurement or logging errors that should be addressed in production.",
]:
    story.append(Paragraph(f"• {c}", bullet))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Key Assumptions", h3))
for a in [
    "Records within each aircraft-component pair are in chronological order of flight_cycles.",
    "last_maintenance_cycles = 0 means a freshly maintained component, not a missing value.",
    "The dataset is representative of normal fleet operations; distribution shift (e.g. new aircraft types) would require model retraining.",
    "Environmental sensors (humidity, ambient temperature) are available at prediction time.",
]:
    story.append(Paragraph(f"• {a}", bullet))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Recommendations for Production Deployment", h3))
story.append(data_table(
    ["Priority", "Action", "Benefit"],
    [["Immediate",    "Deploy predict.py as microservice",            "Same-flight risk scores"],
     ["Immediate",    "Validate on held-out aircraft cohort",          "Confirm generalisation"],
     ["Short-term",   "Calibrate thresholds against AOG cost data",    "Align model to real economics"],
     ["Short-term",   "Set up prediction logging & drift monitoring",  "Catch model degradation early"],
     ["Medium-term",  "Retrain quarterly with new inspection records", "Keep model current as fleet ages"],
     ["Long-term",    "Add Remaining Useful Life (RUL) regression",    "Answer 'how many cycles left?'"],
     ["Long-term",    "Explore LSTM for longer-range temporal patterns","Capture multi-cycle dependencies"]],
    [3*cm, 7*cm, 7*cm],
))
story.append(PageBreak())

# ════════════════════════════════════════════════════════════════
# 8. CONCLUSIONS
# ════════════════════════════════════════════════════════════════
story.append(section_bar("8. Conclusions"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "This work demonstrates that near-perfect aircraft component failure prediction is "
    "achievable using standard machine learning tools, provided the analysis is grounded "
    "in the physical reality of how failures develop.",
    body))
story.append(Spacer(1, 0.2*cm))

story.append(data_table(
    ["Metric", "Score", "Significance"],
    [["ROC-AUC",    "0.9992", "Near-perfect discrimination across all thresholds"],
     ["PR-AUC",     "0.9632", "Excellent precision-recall balance under 42:1 imbalance"],
     ["F1 Score",   "0.9180", "Strong combined performance at optimal threshold"],
     ["Recall",     "1.0000", "Zero missed failures on test set — primary safety objective met"],
     ["Precision",  "0.8485", "~1 false alarm per 6.5 alerts — operationally manageable"],
     ["CV PR-AUC",  "0.906 ± 0.023", "Stable, generalisable — not an artefact of one split"]],
    [3*cm, 2.5*cm, 11.5*cm],
))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "The most important finding is that <b>temporal feature engineering — specifically "
    "rolling fault accumulation — is the decisive factor</b>. The top engineered feature "
    "accounts for 61% of the model's gain. This confirms the physical interpretation: "
    "component failure is a process of escalating degradation, not a random event. "
    "Any maintenance system that monitors only point-in-time sensor values, without "
    "tracking the <i>rate</i> of fault accumulation, is operating with incomplete information.",
    body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "This model is ready for a proof-of-concept deployment within RCAA's maintenance "
    "operations. With a structured feedback loop and quarterly retraining, it has the "
    "potential to meaningfully reduce unplanned AOG events, optimise maintenance "
    "scheduling, and contribute to the safety of Rwanda's civil aviation fleet.",
    body))

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Report saved: {OUTPUT}")
