"""
Generates the PDF Report for RCAA Predictive Maintenance Assessment.
Uses ReportLab for professional PDF layout.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
import json
import os

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT = "RCAA_Predictive_Maintenance_Report.pdf"
FIGURES = "figures"
DARK_BLUE  = colors.HexColor("#1f4e79")
MED_BLUE   = colors.HexColor("#2980b9")
LIGHT_BLUE = colors.HexColor("#d6e4f0")
RED        = colors.HexColor("#c0392b")
GREEN      = colors.HexColor("#27ae60")
ORANGE     = colors.HexColor("#e67e22")
GOLD       = colors.HexColor("#f1c40f")
GRAY       = colors.HexColor("#95a5a6")
LIGHT_GRAY = colors.HexColor("#f5f6fa")

# ── Page header/footer ───────────────────────────────────────────────────────
def header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    w, h = A4
    # Header bar
    canvas_obj.setFillColor(DARK_BLUE)
    canvas_obj.rect(0, h - 1.5 * cm, w, 1.5 * cm, fill=1, stroke=0)
    canvas_obj.setFillColor(colors.white)
    canvas_obj.setFont("Helvetica-Bold", 9)
    canvas_obj.drawString(1.5 * cm, h - 1.0 * cm,
                          "RCAA Predictive Maintenance — Data Science Assessment")
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawRightString(w - 1.5 * cm, h - 1.0 * cm, "Confidential")
    # Footer line
    canvas_obj.setStrokeColor(DARK_BLUE)
    canvas_obj.setLineWidth(1)
    canvas_obj.line(1.5 * cm, 1.5 * cm, w - 1.5 * cm, 1.5 * cm)
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.setFillColor(GRAY)
    canvas_obj.drawString(1.5 * cm, 0.8 * cm, "Rwanda Civil Aviation Authority")
    canvas_obj.drawRightString(w - 1.5 * cm, 0.8 * cm, f"Page {doc.page}")
    canvas_obj.restoreState()

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                    fontSize=20, textColor=DARK_BLUE, spaceAfter=8,
                    fontName='Helvetica-Bold')
h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                    fontSize=14, textColor=DARK_BLUE, spaceBefore=14, spaceAfter=6,
                    fontName='Helvetica-Bold',
                    borderPad=4, leftIndent=0)
h3 = ParagraphStyle('H3', parent=styles['Heading3'],
                    fontSize=11, textColor=MED_BLUE, spaceBefore=8, spaceAfter=4,
                    fontName='Helvetica-Bold')
body = ParagraphStyle('Body', parent=styles['Normal'],
                      fontSize=9.5, leading=15, spaceAfter=6,
                      textColor=colors.HexColor("#2c3e50"), alignment=TA_JUSTIFY)
bullet = ParagraphStyle('Bullet', parent=body, leftIndent=14, bulletIndent=4,
                        spaceBefore=2, spaceAfter=2)
caption = ParagraphStyle('Caption', parent=styles['Normal'],
                         fontSize=8, textColor=GRAY, alignment=TA_CENTER,
                         spaceAfter=10, fontName='Helvetica-Oblique')
metric_label = ParagraphStyle('MetricLabel', parent=styles['Normal'],
                               fontSize=8, textColor=GRAY, alignment=TA_CENTER)
metric_val = ParagraphStyle('MetricVal', parent=styles['Normal'],
                             fontSize=20, fontName='Helvetica-Bold',
                             textColor=DARK_BLUE, alignment=TA_CENTER)

def section_bar(title):
    """Returns a visually distinct section header."""
    return Table(
        [[Paragraph(f"  {title}", ParagraphStyle(
            'SB', fontSize=13, fontName='Helvetica-Bold',
            textColor=colors.white, leading=16))]],
        colWidths=[17 * cm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), DARK_BLUE),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [DARK_BLUE]),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ])
    )

def metric_box(label, value, color=DARK_BLUE):
    t = Table(
        [[Paragraph(str(value), ParagraphStyle('MV', fontSize=18, fontName='Helvetica-Bold',
                                               textColor=color, alignment=TA_CENTER))],
         [Paragraph(label, ParagraphStyle('ML', fontSize=7.5, textColor=GRAY,
                                          alignment=TA_CENTER))]],
        colWidths=[3.9 * cm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#dce6f2')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])
    )
    return t

def add_figure(path, width=15 * cm, caption_text=None):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width * 0.6))
        if caption_text:
            items.append(Paragraph(caption_text, caption))
    return items

# ── Load results ─────────────────────────────────────────────────────────────
with open("final_results_summary.json") as f:
    res = json.load(f)
best_model_name = res['best_model']
best = res['models'][best_model_name]

# ── Document ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    topMargin=2.2 * cm, bottomMargin=2.0 * cm
)
story = []

# ════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════
story.append(Spacer(1, 2 * cm))
story.append(Paragraph(
    "Predictive Maintenance for Aircraft Components",
    ParagraphStyle('Cover1', fontSize=24, fontName='Helvetica-Bold',
                   textColor=DARK_BLUE, alignment=TA_CENTER, leading=30)
))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Data Science Technical Assessment",
    ParagraphStyle('Cover2', fontSize=15, textColor=MED_BLUE,
                   alignment=TA_CENTER, fontName='Helvetica')
))
story.append(Spacer(1, 0.3 * cm))
story.append(HRFlowable(width="80%", thickness=2, color=DARK_BLUE, hAlign='CENTER'))
story.append(Spacer(1, 0.6 * cm))

# Key metrics on cover
metrics_row = Table(
    [[metric_box("ROC-AUC", f"{best['roc_auc']:.4f}", DARK_BLUE),
      metric_box("PR-AUC", f"{best['avg_precision']:.4f}", MED_BLUE),
      metric_box("F1 Score", f"{best['f1']:.4f}", GREEN),
      metric_box("Recall", f"{best['recall']:.4f}", RED)]],
    colWidths=[4.1 * cm] * 4,
    style=TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')])
)
story.append(metrics_row)
story.append(Spacer(1, 0.6 * cm))
story.append(Paragraph(f"Best Model: <b>{best_model_name}</b>", ParagraphStyle(
    'BM', fontSize=11, textColor=DARK_BLUE, alignment=TA_CENTER)))
story.append(Spacer(1, 1.5 * cm))

cover_info = Table(
    [['Submitted to:', 'Rwanda Civil Aviation Authority (RCAA)'],
     ['Role:', 'Data Scientist'],
     ['Date:', 'March 2026'],
     ['Dataset:', 'aircraft_maintenance_dataset.csv (6,000 records)'],
     ['Task:', 'Binary classification — failure within next 10 flight cycles']],
    colWidths=[4 * cm, 12 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK_BLUE),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ])
)
story.append(cover_info)
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 1. PROBLEM UNDERSTANDING
# ════════════════════════════════════════════════════
story.append(section_bar("1. Problem Understanding"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "Airlines face two competing risks: performing maintenance too early wastes resources, "
    "while performing it too late risks Aircraft-on-Ground (AOG) events, safety incidents, "
    "and regulatory non-compliance. The goal of predictive maintenance is to identify "
    "<b>which components are likely to fail within the next 10 flight cycles</b>, allowing "
    "maintenance to be scheduled proactively at the lowest cost and highest safety margin.",
    body
))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("Business Objectives", h3))
biz_table = Table(
    [['Objective', 'Success Metric'],
     ['Reduce unplanned AOG events', 'High Recall (catch all failures)'],
     ['Minimise unnecessary inspections', 'High Precision (reduce false alarms)'],
     ['Support maintenance scheduling', 'Calibrated probability scores'],
     ['Explainable to engineers', 'SHAP feature attribution']],
    colWidths=[8.5 * cm, 8.5 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#aed6f1')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ])
)
story.append(biz_table)
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "<b>Why this is not a simple problem:</b> The class imbalance (42:1) means that even a "
    "model predicting 'no failure' for every record achieves 97.7% accuracy — an entirely "
    "useless result. We must use metrics (PR-AUC, F1, Recall) that reflect actual predictive "
    "value, and we must explicitly handle the imbalance in training.",
    body
))
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 2. DATA EXPLORATION
# ════════════════════════════════════════════════════
story.append(section_bar("2. Data Exploration & Key Insights"))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph("Dataset Overview", h3))
data_overview = Table(
    [['Property', 'Value'],
     ['Total records', '6,000'],
     ['Features (raw)', '13 input + 1 target'],
     ['Aircraft', '20 unique IDs'],
     ['Component types', '4 (Engine1, Engine2, Wing, LandingGear)'],
     ['Failure records', '140 (2.33%)'],
     ['Missing values', '1 row (target null) — dropped'],
     ['Duplicate rows', '0'],
     ['Flight cycle range', '1 – 412 cycles'],
     ['Engine hours range', '0.6 – 285 hours']],
    colWidths=[7 * cm, 10 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), MED_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(data_overview)
story.append(Spacer(1, 0.4 * cm))

story.extend(add_figure(
    f"{FIGURES}/01_target_distribution.png", 14 * cm,
    "Figure 1 — Severe class imbalance: 97.7% no-failure vs 2.3% failure records"
))

story.extend(add_figure(
    f"{FIGURES}/02_feature_distributions.png", 16 * cm,
    "Figure 2 — Sensor distributions by class. "
    "Vibration, fault codes, and temperature show clearest separation."
))

story.append(Paragraph("Key EDA Findings", h3))
findings = [
    "<b>Fault code count</b> is the most discriminating raw feature — failure records show significantly higher fault code accumulation.",
    "<b>Vibration sensor</b> readings are elevated in pre-failure states, consistent with physical wear mechanics.",
    "<b>Temperature sensors</b> show mild separation, especially in the 85–105°C range.",
    "<b>Environmental features</b> (ambient temperature, humidity) show <i>no meaningful separation</i> between classes, suggesting they are noise in this dataset.",
    "Failure events are <b>not uniformly distributed</b> across components: Engine1 and Wing components show higher failure rates.",
    "<b>Sensor drift flag</b> is rare but correlates with elevated fault counts when present.",
]
for f in findings:
    story.append(Paragraph(f"• {f}", bullet))

story.extend(add_figure(
    f"{FIGURES}/04_failure_by_component.png", 14 * cm,
    "Figure 3 — Failure rates by component type. Engine1 and Wing components are highest-risk."
))

story.extend(add_figure(
    f"{FIGURES}/05_boxplots_by_failure.png", 16 * cm,
    "Figure 4 — Boxplots confirm vibration, fault codes, and pressure separate classes clearly."
))
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════
story.append(section_bar("3. Feature Engineering"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "Raw sensor readings capture the current state but not the <b>trajectory</b> "
    "of deterioration. Failure is rarely caused by a single bad reading; it emerges "
    "from <i>accumulation over time</i>. Feature engineering transforms the static snapshot "
    "into a richer temporal representation.",
    body
))

fe_table = Table(
    [['Feature Group', 'Features Created', 'Rationale'],
     ['Rolling Mean (3/5 cycle)', 'per sensor × 2 windows', 'Smooths noise; captures trend direction'],
     ['Rolling Std Dev (3 cycle)', 'per sensor', 'Volatility = instability signal'],
     ['First Difference (Δ)', 'per sensor', 'Rate of change — sudden spikes'],
     ['Cumulative Fault Count', '1 feature', 'Total accumulated wear on component'],
     ['Rolling Fault Sum (5-cycle)', '1 feature', 'Intensity of recent fault activity'],
     ['Engine Hours / Cycle', '1 feature', 'Efficiency degradation proxy'],
     ['Temperature Differential', '1 feature', 'Asymmetric thermal stress'],
     ['Composite Stress Index', '1 feature', 'Vib + Pressure z-score combined'],
     ['Maintenance Urgency', '1 feature', 'Fault count × time since maintenance'],
     ['Interaction: Vib × Temp', '1 feature', 'Non-linear thermal-mechanical stress'],
     ['Drift × Fault interactions', '2 features', 'Sensor reliability × fault count'],
     ['Encoded IDs', '2 features', 'Aircraft / component as numeric']],
    colWidths=[5 * cm, 5 * cm, 7 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(fe_table)
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "After feature engineering, the model has access to <b>43 features</b> (up from 13). "
    "Post-analysis, the <i>fault_roll5_sum</i> feature (rolling 5-cycle fault accumulation) "
    "emerged as by far the most important predictor — capturing the physical reality that "
    "failures are preceded by intensifying fault code activity.",
    body
))

story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("Important Assumptions", h3))
story.append(Paragraph(
    "• Rolling features were computed <b>per aircraft-component pair</b> (sorted by flight_cycles) "
    "to prevent data leakage across independent maintenance tracks.<br/>"
    "• The 1 row with a missing target was dropped rather than imputed — imputing a failure/no-failure "
    "label introduces artificial signal.<br/>"
    "• <i>last_maintenance_cycles = 0</i> is treated as valid data (freshly maintained component), "
    "though it could also indicate missing data in a real system.",
    body
))
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 4. MODEL SELECTION & EVALUATION
# ════════════════════════════════════════════════════
story.append(section_bar("4. Model Selection & Evaluation"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("Handling Class Imbalance", h3))
story.append(Paragraph(
    "With a 42:1 class ratio, three strategies were combined: "
    "(1) <b>cost-sensitive learning</b> — <i>class_weight='balanced'</i> and <i>scale_pos_weight</i> "
    "make the algorithm penalise minority-class misclassification proportionally; "
    "(2) <b>stratified K-fold cross-validation</b> (k=5) — each fold preserves the class ratio; "
    "(3) <b>PR-AUC as the primary validation metric</b> — more informative than ROC-AUC when "
    "positive class prevalence is very low.",
    body
))

story.append(Paragraph("Model Comparison", h3))
model_table = Table(
    [['Model', 'ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall', 'CV PR-AUC'],
     ['Logistic Regression', '0.9832', '0.5573', '0.5106', '0.3636', '0.8571', '0.541 ± 0.099'],
     ['Random Forest',       '0.9974', '0.8752', '0.8254', '0.7429', '0.9286', '0.843 ± 0.103'],
     ['LightGBM',            '0.9986', '0.9369', '0.8621', '0.8333', '0.8929', '0.906 ± 0.026'],
     [f'{best_model_name} ★','0.9993', '0.9687', '0.9333', '0.8750', '1.0000', '0.915 ± 0.026']],
    colWidths=[4.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 1.8 * cm, 3.6 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [LIGHT_BLUE, colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#d5f5e3')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(model_table)
story.append(Spacer(1, 0.3 * cm))

story.extend(add_figure(
    f"{FIGURES}/07_roc_pr_curves.png", 15 * cm,
    "Figure 5 — ROC and Precision-Recall curves. PR-AUC is the primary metric given class imbalance."
))

story.extend(add_figure(
    f"{FIGURES}/08_confusion_matrices.png", 15 * cm,
    "Figure 6 — Confusion matrices at default threshold. XGBoost achieves 0 false negatives."
))

story.append(Paragraph("Why XGBoost?", h3))
story.append(Paragraph(
    "XGBoost consistently outperformed alternatives because: "
    "(1) gradient boosting builds specialised trees that progressively focus on hard-to-classify minority samples; "
    "(2) <i>scale_pos_weight</i> directly adjusts the loss function rather than just resampling; "
    "(3) regularisation (subsample=0.8, colsample_bytree=0.8) prevents overfitting to the small positive class. "
    "Logistic Regression's linear decision boundary cannot capture the non-linear interaction "
    "between fault accumulation, vibration, and maintenance history.",
    body
))
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 5. EXPLAINABILITY & THRESHOLD
# ════════════════════════════════════════════════════
story.append(section_bar("5. Model Explainability & Threshold Selection"))
story.append(Spacer(1, 0.3 * cm))

story.extend(add_figure(
    f"{FIGURES}/10_feature_importance.png", 14 * cm,
    "Figure 7 — Top 20 feature importances (XGBoost gain). "
    "fault_roll5_sum dominates — confirming the temporal accumulation hypothesis."
))

story.append(Paragraph("Interpretation of Top Features", h3))
interp_table = Table(
    [['Feature', 'Importance', 'Interpretation'],
     ['fault_roll5_sum', '68.1%', 'Rolling 5-cycle fault total — key pre-failure signal'],
     ['fault_code_count_roll5_mean', '20.6%', 'Average fault intensity over 5 cycles'],
     ['cumulative_faults', '1.1%', 'Total lifetime fault accumulation'],
     ['temperature_sensor_2_roll5_mean', '0.9%', 'Sustained elevated temperature'],
     ['vibration_sensor_roll3_mean', '0.8%', 'Recent vibration trend'],
     ['sensor_drift_flag', '0.4%', 'Sensor reliability indicator'],
     ['flight_cycles', '0.4%', 'Component age proxy']],
    colWidths=[5.5 * cm, 2.5 * cm, 9 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), MED_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(interp_table)
story.append(Spacer(1, 0.3 * cm))

story.extend(add_figure(
    f"{FIGURES}/12_threshold_optimization.png", 14 * cm,
    "Figure 8 — Threshold optimisation. Optimal F1 at threshold=0.58 achieves perfect recall. "
    "Aviation operators in safety-critical contexts should use a lower threshold (~0.25)."
))

story.append(Paragraph("Threshold Guidance for Operations", h3))
thresh_guidance = Table(
    [['Context', 'Recommended Threshold', 'Expected Trade-off'],
     ['Safety-critical routes (long-haul)', '0.20 – 0.30', 'Very high recall; more inspections'],
     ['Balanced operations', '0.50 – 0.58', 'Optimal F1; balanced precision/recall'],
     ['Cost-optimised (short-haul, backup fleet)', '0.65 – 0.75', 'Fewer false alarms; accept some miss risk']],
    colWidths=[5.5 * cm, 4.5 * cm, 7 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(thresh_guidance)
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 6. MAINTENANCE DECISION SUPPORT
# ════════════════════════════════════════════════════
story.append(section_bar("6. Maintenance Decision Support System"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "The model is operationalised as a real-time risk scoring pipeline. Every flight, "
    "sensor readings are ingested, features are computed, and each aircraft-component pair "
    "receives a failure probability score (0–1) mapped to a maintenance tier.",
    body
))

tier_table = Table(
    [['Tier', 'Score Range', 'Count (Dataset)', 'Recommended Action'],
     ['🔴 CRITICAL', '≥ 0.60', '143 (2.4%)', 'Ground aircraft — urgent inspection required'],
     ['🟠 HIGH',     '0.35–0.59', '2 (0.03%)', 'Flag for next maintenance window'],
     ['🟡 MEDIUM',   '0.15–0.34', '2 (0.03%)', 'Increase monitoring; review next service'],
     ['🟢 LOW',      '< 0.15', '5,853 (97.6%)', 'Standard operations — routine monitoring']],
    colWidths=[2.5 * cm, 3 * cm, 3.5 * cm, 8 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fadbd8')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#fde8d8')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fef9e7')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#d5f5e3')),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, GRAY),
    ])
)
story.append(tier_table)
story.append(Spacer(1, 0.3 * cm))

story.extend(add_figure(
    f"{FIGURES}/13_risk_scoring.png", 15 * cm,
    "Figure 9 — Risk tier distribution and score histogram. "
    "The bimodal distribution confirms the model makes clear high/low risk distinctions."
))

story.append(Paragraph("Integration Pathway", h3))
for step in [
    "<b>Data ingestion:</b> Flight data management system (FDMS) exports sensor logs after each flight cycle.",
    "<b>Feature computation:</b> Automated pipeline computes rolling, cumulative, and interaction features per component.",
    "<b>Scoring:</b> Trained XGBoost model outputs a failure probability and risk tier for each aircraft-component.",
    "<b>Dashboard:</b> Maintenance control centre dashboard shows all CRITICAL/HIGH tier components with drill-down to sensor detail.",
    "<b>Feedback loop:</b> Post-inspection outcomes are logged to retrain the model quarterly.",
]:
    story.append(Paragraph(f"→ {step}", bullet))
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 7. CHALLENGES, ASSUMPTIONS & RECOMMENDATIONS
# ════════════════════════════════════════════════════
story.append(section_bar("7. Challenges, Assumptions & Recommendations"))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph("Key Challenges", h3))
challenges = [
    "<b>Extreme class imbalance (42:1):</b> Required careful metric selection (PR-AUC over accuracy) and algorithmic adjustment (scale_pos_weight). Standard off-the-shelf models would effectively ignore the minority class.",
    "<b>Temporal structure:</b> Data looks like time series per aircraft-component, but it is provided as a flat file without explicit timestamps. Rolling features must be computed carefully per-group to avoid look-ahead bias.",
    "<b>Small positive class:</b> 140 failure events limits statistical power. Cross-validation was essential to get reliable performance estimates.",
    "<b>Feature leakage risk:</b> Features like cumulative_faults use future data relative to a hypothetical deployment point; in production, these must be recomputed from historical data only.",
    "<b>Synthetic data characteristics:</b> The humidity range includes -1.5% and values > 100%, and ambient temperature range is very wide. Real-world data would require additional anomaly detection.",
]
for c in challenges:
    story.append(Paragraph(f"• {c}", bullet))

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Assumptions", h3))
for a in [
    "Data rows within each aircraft-component pair are chronologically ordered by flight_cycles.",
    "last_maintenance_cycles = 0 indicates a freshly maintained component (not missing data).",
    "sensor_drift_flag and maintenance_log_flag are reliable binary labels despite low prevalence.",
    "Environmental features (humidity, ambient_temperature) are available at prediction time.",
]:
    story.append(Paragraph(f"• {a}", bullet))

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Recommendations for Production Deployment", h3))
recs = Table(
    [['Priority', 'Action', 'Benefit'],
     ['Immediate', 'Deploy risk scoring dashboard', 'Actionable insight for maintenance planners'],
     ['Short-term', 'Calibrate thresholds via cost analysis', 'Align model to actual cost-benefit ratio'],
     ['Short-term', 'Set up prediction logging', 'Enable model drift monitoring'],
     ['Medium-term', 'Retrain quarterly with new data', 'Maintain accuracy as fleet ages'],
     ['Long-term', 'Add survival analysis (RUL)', 'Estimate exact remaining useful life'],
     ['Long-term', 'Explore LSTM for richer temporal modelling', 'Capture longer-range dependencies']],
    colWidths=[3 * cm, 7 * cm, 7 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(recs)
story.append(PageBreak())

# ════════════════════════════════════════════════════
# 8. CONCLUSIONS
# ════════════════════════════════════════════════════
story.append(section_bar("8. Conclusions"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "This project demonstrates that a well-engineered XGBoost model can achieve "
    "<b>near-perfect failure detection</b> (Recall = 1.00, PR-AUC = 0.97) for aircraft "
    "component failure prediction — even under severe class imbalance conditions.",
    body
))
story.append(Spacer(1, 0.2 * cm))

final_metrics = Table(
    [['Metric', 'Value', 'Interpretation'],
     ['ROC-AUC', '0.9993', 'Near-perfect class separation'],
     ['PR-AUC', '0.9687', 'Excellent precision-recall trade-off'],
     ['F1 Score', '0.9333', 'Strong balanced performance'],
     ['Recall', '1.0000', 'Zero missed failures in test set'],
     ['Precision', '0.8750', '1 false alarm per 7 alerts'],
     ['CV PR-AUC', '0.915 ± 0.026', 'Stable and generalisable']],
    colWidths=[5 * cm, 3 * cm, 9 * cm],
    style=TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#d5f5e3'), colors.white]),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#aed6f1')),
    ])
)
story.append(final_metrics)
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "The most important finding is that <b>temporal feature engineering is the decisive factor</b>. "
    "The rolling 5-cycle fault accumulation feature alone accounts for 68% of the model's "
    "predictive power — validating the physical intuition that component failure is preceded "
    "by escalating fault activity, not a single catastrophic reading.",
    body
))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "This model is ready for a proof-of-concept deployment. With proper integration into "
    "existing maintenance management systems and a structured feedback loop, it has the "
    "potential to meaningfully reduce unplanned AOG events and optimise maintenance costs "
    "for the Rwandan civil aviation fleet.",
    body
))

# Build
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Report saved: {OUTPUT}")
