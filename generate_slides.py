"""
Generates the PowerPoint Presentation for RCAA Predictive Maintenance Assessment.
Designed for 10–15 min presentation to both technical and non-technical stakeholders.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import os
import json

# ── Colours ───────────────────────────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x1f, 0x4e, 0x79)
MED_BLUE   = RGBColor(0x29, 0x80, 0xb9)
LIGHT_BLUE = RGBColor(0xd6, 0xe4, 0xf0)
RED        = RGBColor(0xc0, 0x39, 0x2b)
GREEN      = RGBColor(0x27, 0xae, 0x60)
ORANGE     = RGBColor(0xe6, 0x7e, 0x22)
GOLD       = RGBColor(0xf1, 0xc4, 0x0f)
WHITE      = RGBColor(0xff, 0xff, 0xff)
LIGHT_GRAY = RGBColor(0xf5, 0xf6, 0xfa)
DARK_GRAY  = RGBColor(0x2c, 0x3e, 0x50)

FIGURES = "figures"

# ── Load results ──────────────────────────────────────────────────────────────
with open("final_results_summary.json") as f:
    res_data = json.load(f)
best_name = res_data['best_model']

# ── Helpers ───────────────────────────────────────────────────────────────────
def new_prs():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    return prs

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank

def add_rect(slide, left, top, width, height, fill_color=None, line_color=None, line_width=Pt(0)):
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    fill = shape.fill
    if fill_color:
        fill.solid()
        fill.fore_color.rgb = fill_color
    else:
        fill.background()
    line = shape.line
    if line_color:
        line.color.rgb = line_color
        line.width = line_width
    else:
        line.fill.background()
    return shape

def add_text_box(slide, text, left, top, width, height,
                 font_size=18, bold=False, color=DARK_GRAY,
                 align=PP_ALIGN.LEFT, italic=False, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return txBox

def header_bar(slide, title, subtitle=None):
    """Dark blue header bar with title."""
    add_rect(slide, 0, 0, 13.33, 1.35, fill_color=DARK_BLUE)
    add_text_box(slide, title, 0.3, 0.12, 12.5, 0.75,
                 font_size=30, bold=True, color=WHITE, font_name="Calibri Light")
    if subtitle:
        add_text_box(slide, subtitle, 0.3, 0.85, 12.5, 0.4,
                     font_size=13, color=LIGHT_BLUE, font_name="Calibri Light")
    # Accent bar
    add_rect(slide, 0, 1.35, 13.33, 0.06, fill_color=MED_BLUE)

def add_image(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        if width and height:
            slide.shapes.add_picture(path, Inches(left), Inches(top),
                                     Inches(width), Inches(height))
        elif width:
            slide.shapes.add_picture(path, Inches(left), Inches(top),
                                     width=Inches(width))
        else:
            slide.shapes.add_picture(path, Inches(left), Inches(top))

def metric_block(slide, left, top, value, label, bg_color=MED_BLUE, val_color=WHITE):
    """A metric KPI box."""
    add_rect(slide, left, top, 2.6, 1.4, fill_color=bg_color)
    add_text_box(slide, value, left, top + 0.1, 2.6, 0.75,
                 font_size=34, bold=True, color=val_color,
                 align=PP_ALIGN.CENTER, font_name="Calibri")
    add_text_box(slide, label, left, top + 0.85, 2.6, 0.4,
                 font_size=10, color=WHITE, align=PP_ALIGN.CENTER)

def footer(slide, page_num, total=12):
    add_rect(slide, 0, 7.15, 13.33, 0.35, fill_color=DARK_BLUE)
    add_text_box(slide, "RCAA — Predictive Maintenance Assessment | Confidential",
                 0.2, 7.15, 10, 0.35, font_size=8, color=LIGHT_BLUE)
    add_text_box(slide, f"{page_num} / {total}", 12.6, 7.15, 0.7, 0.35,
                 font_size=8, color=LIGHT_BLUE, align=PP_ALIGN.RIGHT)

def bullet_list(slide, items, left, top, width, height, font_size=14, spacing=0.35):
    for i, (indent, text) in enumerate(items):
        prefix = "  " * indent + ("▪ " if indent == 0 else "  – ")
        add_text_box(slide, prefix + text, left, top + i * spacing, width, spacing + 0.05,
                     font_size=font_size, color=DARK_GRAY)

# ── Build slides ──────────────────────────────────────────────────────────────
prs = new_prs()

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
add_rect(slide, 0, 0, 13.33, 7.5, fill_color=DARK_BLUE)
add_rect(slide, 0, 4.5, 13.33, 3.0, fill_color=MED_BLUE)
add_rect(slide, 0, 4.4, 13.33, 0.12, fill_color=WHITE)

add_text_box(slide, "Predictive Maintenance", 0.5, 1.0, 12.3, 1.0,
             font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             font_name="Calibri Light")
add_text_box(slide, "for Aircraft Components", 0.5, 1.9, 12.3, 0.8,
             font_size=36, bold=False, color=LIGHT_BLUE, align=PP_ALIGN.CENTER,
             font_name="Calibri Light")
add_text_box(slide, "Rwanda Civil Aviation Authority — Data Science Assessment",
             0.5, 2.9, 12.3, 0.5,
             font_size=16, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(slide, "March 2026", 0.5, 5.0, 12.3, 0.5,
             font_size=14, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(slide, "Predicting component failure within the next 10 flight cycles",
             0.5, 5.6, 12.3, 0.5,
             font_size=13, color=LIGHT_BLUE, align=PP_ALIGN.CENTER, italic=True)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Agenda
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Agenda", "What we'll cover in the next 12 minutes")

agenda_items = [
    (0, "Business Context & Problem Framing"),
    (0, "Dataset Overview & Key Findings from EDA"),
    (0, "Feature Engineering Strategy"),
    (0, "Model Training & Evaluation"),
    (0, "Explainability: What Drives Failure?"),
    (0, "Threshold Optimisation for Aviation Operations"),
    (0, "Maintenance Decision Support System"),
    (0, "Challenges, Assumptions & Recommendations"),
]
bullet_list(slide, agenda_items, 0.6, 1.6, 12, 5.2, font_size=18, spacing=0.55)
footer(slide, 2)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Problem Context
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "The Business Problem", "Why predictive maintenance matters in aviation")

# Left panel
add_rect(slide, 0.3, 1.6, 5.9, 5.5, fill_color=LIGHT_BLUE)
add_text_box(slide, "The Challenge", 0.5, 1.7, 5.5, 0.45,
             font_size=14, bold=True, color=DARK_BLUE)
challenge_items = [
    (0, "Unplanned failures → AOG events"),
    (0, "Safety incidents & regulatory risk"),
    (0, "Costly emergency maintenance"),
    (0, "Early maintenance wastes resources"),
    (0, "Manual inspection is inefficient"),
]
bullet_list(slide, challenge_items, 0.5, 2.2, 5.7, 3.5, font_size=12.5, spacing=0.47)

# Right panel
add_rect(slide, 6.7, 1.6, 6.3, 5.5, fill_color=RGBColor(0xd5, 0xf5, 0xe3))
add_text_box(slide, "Our Solution", 6.9, 1.7, 5.9, 0.45,
             font_size=14, bold=True, color=GREEN)
solution_items = [
    (0, "ML model flags at-risk components"),
    (0, "10-cycle advance warning"),
    (0, "Calibrated risk scores (0–1)"),
    (0, "4-tier actionable classification"),
    (0, "Explainable to maintenance engineers"),
]
bullet_list(slide, solution_items, 6.9, 2.2, 6.0, 3.5, font_size=12.5, spacing=0.47)

add_text_box(slide, "→ Predict before it breaks. Schedule before it costs.",
             0.3, 7.0, 12.7, 0.4, font_size=12, bold=True, color=DARK_BLUE,
             align=PP_ALIGN.CENTER, italic=True)
footer(slide, 3)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Dataset & Class Imbalance
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Dataset Overview", "6,000 records | 20 aircraft | 4 component types")

# Stats column
stats = [
    ("6,000", "Records"),
    ("13", "Raw Features"),
    ("20", "Aircraft"),
    ("4", "Components"),
    ("140", "Failure Events"),
    ("42:1", "Class Ratio"),
]
for i, (val, lbl) in enumerate(stats):
    row, col = divmod(i, 2)
    bg = RED if lbl == "Failure Events" else (ORANGE if lbl == "Class Ratio" else MED_BLUE)
    metric_block(slide, 0.3 + col * 2.75, 1.55 + row * 1.6, val, lbl, bg_color=bg)

# Right: target distribution image
add_image(slide, f"{FIGURES}/01_target_distribution.png", 6.1, 1.5, width=6.9, height=4.0)
add_text_box(slide, "⚠ Only 2.3% of records are failures — class imbalance is the core challenge",
             0.3, 6.8, 12.7, 0.4, font_size=11, bold=True, color=RED,
             align=PP_ALIGN.CENTER)
footer(slide, 4)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — EDA Highlights
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "EDA: Key Insights", "What the data tells us before any modelling")

add_image(slide, f"{FIGURES}/05_boxplots_by_failure.png", 0.2, 1.5, width=8.0, height=4.3)

insight_items = [
    (0, "Fault codes: strongest raw signal"),
    (0, "Vibration elevated before failures"),
    (0, "Temperature mild separation"),
    (0, "Humidity/ambient: low signal"),
    (0, "Engine1 & Wing: highest failure rate"),
    (0, "Failures cluster in later flight cycles"),
]
add_rect(slide, 8.4, 1.5, 4.6, 4.3, fill_color=LIGHT_BLUE)
add_text_box(slide, "Key Findings", 8.6, 1.6, 4.2, 0.4,
             font_size=13, bold=True, color=DARK_BLUE)
bullet_list(slide, insight_items, 8.5, 2.1, 4.3, 3.5, font_size=12, spacing=0.44)

add_text_box(slide, "→ No single sensor predicts failure alone — temporal patterns matter",
             0.2, 6.0, 12.9, 0.4, font_size=11, bold=True, color=DARK_BLUE,
             align=PP_ALIGN.CENTER, italic=True)
footer(slide, 5)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Feature Engineering
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Feature Engineering", "From snapshots to trajectories — 13 → 43 features")

fe_blocks = [
    (MED_BLUE,  "Rolling Stats",       "Mean & std dev over\n3 and 5 cycles per sensor"),
    (GREEN,     "Rate of Change",      "First-difference of\neach sensor reading"),
    (ORANGE,    "Fault Accumulation",  "Cumulative & rolling\n5-cycle fault totals"),
    (RED,       "Composite Stress",    "Vibration + Pressure\nz-score index"),
    (RGBColor(0x8e, 0x44, 0xad), "Interactions",   "Vibration × Temperature\nDrift × Faults"),
    (RGBColor(0x16, 0xa0, 0x85), "Maint. Urgency", "Fault count × cycles\nsince last service"),
]
for i, (color, title, desc) in enumerate(fe_blocks):
    row, col = divmod(i, 3)
    lx = 0.3 + col * 4.3
    ty = 1.7 + row * 2.3
    add_rect(slide, lx, ty, 4.0, 2.0, fill_color=color)
    add_text_box(slide, title, lx + 0.15, ty + 0.15, 3.7, 0.55,
                 font_size=14, bold=True, color=WHITE)
    add_text_box(slide, desc, lx + 0.15, ty + 0.7, 3.7, 1.1,
                 font_size=11, color=WHITE)

add_text_box(slide, "★  fault_roll5_sum (rolling 5-cycle fault sum) = #1 predictor, 68% of model gain",
             0.3, 6.3, 12.7, 0.5, font_size=12, bold=True, color=DARK_BLUE,
             align=PP_ALIGN.CENTER)
footer(slide, 6)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Model Results
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Model Results", f"Best model: {best_name} | Evaluated on held-out 20% test set")

# KPI metrics
kpis = [
    ("0.9993", "ROC-AUC", DARK_BLUE),
    ("0.9687", "PR-AUC", MED_BLUE),
    ("0.9333", "F1 Score", GREEN),
    ("1.0000", "Recall", RED),
]
for i, (val, lbl, color) in enumerate(kpis):
    metric_block(slide, 0.3 + i * 3.1, 1.6, val, lbl, bg_color=color)

# PR curve image
add_image(slide, f"{FIGURES}/07_roc_pr_curves.png", 0.2, 3.35, width=8.3, height=3.5)

# Model comparison table text
comparison = [
    ("Logistic Regression",  "0.557", "0.857"),
    ("Random Forest",        "0.875", "0.929"),
    ("LightGBM",             "0.937", "0.893"),
    (f"{best_name} ★",       "0.969", "1.000"),
]
add_rect(slide, 8.6, 3.35, 4.5, 3.5, fill_color=LIGHT_BLUE)
add_text_box(slide, "Model", 8.7, 3.45, 2.0, 0.35, font_size=10, bold=True, color=DARK_BLUE)
add_text_box(slide, "PR-AUC", 10.7, 3.45, 1.0, 0.35, font_size=10, bold=True, color=DARK_BLUE)
add_text_box(slide, "Recall", 11.8, 3.45, 1.0, 0.35, font_size=10, bold=True, color=DARK_BLUE)
for i, (model, ap, rec) in enumerate(comparison):
    add_text_box(slide, model, 8.7, 3.9 + i * 0.62, 2.0, 0.55,
                 font_size=9.5, bold=(i == 3), color=DARK_BLUE if i < 3 else GREEN)
    add_text_box(slide, ap, 10.7, 3.9 + i * 0.62, 1.0, 0.55,
                 font_size=9.5, bold=(i == 3), color=DARK_BLUE if i < 3 else GREEN, align=PP_ALIGN.CENTER)
    add_text_box(slide, rec, 11.8, 3.9 + i * 0.62, 1.0, 0.55,
                 font_size=9.5, bold=(i == 3), color=DARK_BLUE if i < 3 else GREEN, align=PP_ALIGN.CENTER)

footer(slide, 7)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Explainability
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Why Does the Model Predict Failure?", "Feature importance + SHAP explainability")

add_image(slide, f"{FIGURES}/10_feature_importance.png", 0.2, 1.5, width=7.0, height=4.5)

add_rect(slide, 7.5, 1.5, 5.5, 4.5, fill_color=LIGHT_BLUE)
add_text_box(slide, "What This Means for Engineers", 7.7, 1.6, 5.1, 0.5,
             font_size=13, bold=True, color=DARK_BLUE)
explain_items = [
    (0, "fault_roll5_sum (68%): rapid fault\n   accumulation = imminent failure"),
    (0, "fault_roll5_mean (21%): sustained\n   fault intensity signal"),
    (0, "cumulative_faults: overall\n   component wear"),
    (0, "Vibration trend: mechanical\n   degradation signal"),
    (0, "Sensor drift: unreliable data\n   correlates with real issues"),
]
bullet_list(slide, explain_items, 7.6, 2.2, 5.2, 3.5, font_size=11, spacing=0.60)

add_text_box(slide, "→ The model mirrors what experienced engineers know: faults don't happen in isolation",
             0.2, 6.2, 12.9, 0.45, font_size=11, bold=True, color=DARK_BLUE,
             align=PP_ALIGN.CENTER, italic=True)
footer(slide, 8)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Threshold Optimisation
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Choosing the Right Alert Threshold",
           "Balancing missed failures vs unnecessary inspections")

add_image(slide, f"{FIGURES}/12_threshold_optimization.png", 0.2, 1.5, width=7.8, height=4.3)

add_rect(slide, 8.2, 1.5, 4.8, 4.3, fill_color=LIGHT_BLUE)
add_text_box(slide, "Threshold Guide", 8.4, 1.6, 4.4, 0.45,
             font_size=13, bold=True, color=DARK_BLUE)

tiers = [
    (RED,    "Safety-critical:  0.20–0.30", "Max recall, more checks"),
    (ORANGE, "Balanced ops:  0.50–0.58", "Optimal F1 (default)"),
    (GREEN,  "Cost-saving:  0.65–0.75", "Fewer alerts, some risk"),
]
for i, (color, title, desc) in enumerate(tiers):
    add_rect(slide, 8.3, 2.2 + i * 1.1, 4.5, 0.95, fill_color=color)
    add_text_box(slide, title, 8.45, 2.22 + i * 1.1, 4.2, 0.42,
                 font_size=11, bold=True, color=WHITE)
    add_text_box(slide, desc, 8.45, 2.62 + i * 1.1, 4.2, 0.38,
                 font_size=10, color=WHITE)

add_text_box(slide, "At threshold 0.58: F1 = 0.949 | Recall = 1.00 | Precision = 0.903",
             0.2, 6.0, 12.9, 0.45, font_size=12, bold=True, color=GREEN,
             align=PP_ALIGN.CENTER)
footer(slide, 9)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Decision Support System
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Maintenance Decision Support System",
           "From model output to actionable maintenance orders")

add_image(slide, f"{FIGURES}/13_risk_scoring.png", 0.2, 1.5, width=7.8, height=3.8)

tiers_right = [
    (RED,    "🔴  CRITICAL  (≥0.60)", "Ground immediately — urgent inspection"),
    (ORANGE, "🟠  HIGH  (0.35–0.59)", "Flag for next maintenance window"),
    (GOLD,   "🟡  MEDIUM  (0.15–0.34)", "Increase monitoring frequency"),
    (GREEN,  "🟢  LOW  (<0.15)", "Standard operations — routine check"),
]
add_rect(slide, 8.2, 1.5, 4.8, 3.8, fill_color=LIGHT_BLUE)
add_text_box(slide, "Risk Tiers & Actions", 8.4, 1.55, 4.4, 0.4,
             font_size=12, bold=True, color=DARK_BLUE)
for i, (color, tier, action) in enumerate(tiers_right):
    add_rect(slide, 8.3, 2.05 + i * 0.8, 4.5, 0.7, fill_color=color)
    add_text_box(slide, tier, 8.45, 2.07 + i * 0.8, 4.2, 0.32,
                 font_size=10, bold=True, color=WHITE)
    add_text_box(slide, action, 8.45, 2.38 + i * 0.8, 4.2, 0.32,
                 font_size=9, color=WHITE)

# Process flow
flow_items = [
    "1. Ingest sensor data after each flight",
    "2. Compute engineered features per component",
    "3. Score with XGBoost model",
    "4. Assign risk tier → trigger action",
    "5. Log outcome for model retraining",
]
add_rect(slide, 0.2, 5.5, 12.9, 1.3, fill_color=DARK_BLUE)
for i, item in enumerate(flow_items):
    add_text_box(slide, item, 0.4 + i * 2.6, 5.6, 2.5, 0.9,
                 font_size=9.5, color=WHITE, align=PP_ALIGN.CENTER)

footer(slide, 10)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Challenges & Assumptions
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
header_bar(slide, "Challenges & Assumptions",
           "Honest assessment of limitations and deployment considerations")

left_items = [
    (0, "42:1 class imbalance — accuracy is\n   misleading; PR-AUC is the right metric"),
    (0, "Temporal structure needs careful\n   per-group rolling feature computation"),
    (0, "Small positive class (140) limits\n   statistical confidence"),
    (0, "Possible feature leakage: cumulative\n   features require historical-only data"),
]
right_items = [
    (0, "Humidity & ambient temp appear noisy\n   in this dataset — may be excluded"),
    (0, "Assumes flight_cycles correctly orders\n   records chronologically per aircraft"),
    (0, "last_maintenance_cycles = 0 treated\n   as 'freshly maintained', not missing"),
    (0, "Model needs quarterly retraining\n   as fleet ages and usage changes"),
]

add_rect(slide, 0.2, 1.6, 6.3, 5.2, fill_color=LIGHT_BLUE)
add_text_box(slide, "Technical Challenges", 0.4, 1.7, 5.9, 0.42,
             font_size=13, bold=True, color=DARK_BLUE)
bullet_list(slide, left_items, 0.3, 2.2, 6.0, 4.0, font_size=11.5, spacing=0.74)

add_rect(slide, 6.8, 1.6, 6.3, 5.2, fill_color=RGBColor(0xfb, 0xf3, 0xe0))
add_text_box(slide, "Key Assumptions", 7.0, 1.7, 5.9, 0.42,
             font_size=13, bold=True, color=ORANGE)
bullet_list(slide, right_items, 6.9, 2.2, 6.0, 4.0, font_size=11.5, spacing=0.74)
footer(slide, 11)

# ────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Summary & Next Steps
# ────────────────────────────────────────────────────────────────────────────
slide = blank_slide(prs)
add_rect(slide, 0, 0, 13.33, 7.5, fill_color=DARK_BLUE)
add_rect(slide, 0, 5.8, 13.33, 1.7, fill_color=MED_BLUE)

add_text_box(slide, "Summary", 0.5, 0.3, 12.3, 0.7,
             font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             font_name="Calibri Light")
add_rect(slide, 3.0, 1.05, 7.3, 0.06, fill_color=WHITE)

kpis_final = [
    ("0.9993", "ROC-AUC"),
    ("0.9687", "PR-AUC"),
    ("1.0000", "Recall"),
    ("0.9333", "F1"),
]
for i, (val, lbl) in enumerate(kpis_final):
    bg = [MED_BLUE, MED_BLUE, RED, GREEN][i]
    add_rect(slide, 0.5 + i * 3.1, 1.3, 2.8, 1.5, fill_color=bg)
    add_text_box(slide, val, 0.5 + i * 3.1, 1.35, 2.8, 0.85,
                 font_size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, lbl, 0.5 + i * 3.1, 2.15, 2.8, 0.5,
                 font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

summary_points = [
    "XGBoost with temporal feature engineering achieves near-perfect failure detection",
    "Rolling fault accumulation (fault_roll5_sum) is the decisive predictor",
    "Model is ready for proof-of-concept integration with FDMS",
    "Threshold can be tuned to match the airline's risk tolerance",
]
for i, pt in enumerate(summary_points):
    add_text_box(slide, f"✓  {pt}", 0.5, 3.05 + i * 0.58, 12.3, 0.55,
                 font_size=13, color=WHITE)

add_text_box(slide, "Thank you — Questions welcome",
             0.5, 6.0, 12.3, 0.55, font_size=20, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, font_name="Calibri Light")
add_text_box(slide, "Rwanda Civil Aviation Authority | March 2026",
             0.5, 6.65, 12.3, 0.45, font_size=11, color=LIGHT_BLUE,
             align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────────────
output = "RCAA_Predictive_Maintenance_Slides.pptx"
prs.save(output)
print(f"Slides saved: {output}")
