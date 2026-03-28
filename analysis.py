"""
Predictive Maintenance for Aircraft Components
Rwanda Civil Aviation Authority - Data Science Assessment
Author: Data Science Candidate
Date: March 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, average_precision_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap
import json
import pickle

warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "aircraft_maintenance_dataset.csv"
FIGURES_DIR = "figures"
MODELS_DIR = "models"
RANDOM_STATE = 42
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE = ['#1f4e79', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad']
sns.set_palette(PALETTE)


# ─── 1. Data Loading ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
# Drop fully-empty columns (trailing commas in CSV)
df = df.dropna(axis=1, how='all')
df.columns = df.columns.str.strip().str.replace('"', '')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")


# ─── 2. Initial Data Quality Assessment ──────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATA QUALITY ASSESSMENT")
print("=" * 60)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nTarget distribution:\n{df['failure_within_10_cycles'].value_counts()}")
print(f"Class imbalance ratio: {df['failure_within_10_cycles'].value_counts()[0] / df['failure_within_10_cycles'].value_counts()[1]:.1f}:1")

print(f"\nUnique aircraft: {df['aircraft_id'].nunique()}")
print(f"Unique components: {df['component_id'].nunique()}")
print(f"Unique aircraft-component pairs: {df.groupby(['aircraft_id','component_id']).ngroups}")

print(f"\nNumerical summary:\n{df.describe().round(2)}")

# Save summary stats
df.describe().round(3).to_csv("data_summary.csv")


# ─── 3. Exploratory Data Analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 3.1 Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df['failure_within_10_cycles'].value_counts()
axes[0].bar(['No Failure\n(0)', 'Failure\n(1)'], counts.values,
            color=[PALETTE[0], PALETTE[1]], edgecolor='white', linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, f'{v:,}\n({v/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Target Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, counts.max() * 1.2)

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=['No Failure (0)', 'Failure (1)'],
    colors=[PALETTE[0], PALETTE[1]], autopct='%1.1f%%',
    startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2)
)
axes[1].set_title('Class Proportion', fontweight='bold')
plt.suptitle('Target Variable: Failure Within 10 Flight Cycles', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_target_distribution.png", bbox_inches='tight')
plt.close()

# 3.2 Sensor distributions by class
numeric_features = ['engine_hours', 'temperature_sensor_1', 'temperature_sensor_2',
                    'vibration_sensor', 'pressure_sensor', 'fault_code_count',
                    'last_maintenance_cycles', 'ambient_temperature', 'humidity']
feature_labels = {
    'engine_hours': 'Engine Hours',
    'temperature_sensor_1': 'Temperature Sensor 1 (°C)',
    'temperature_sensor_2': 'Temperature Sensor 2 (°C)',
    'vibration_sensor': 'Vibration (g)',
    'pressure_sensor': 'Pressure (psi)',
    'fault_code_count': 'Fault Code Count',
    'last_maintenance_cycles': 'Cycles Since Last Maintenance',
    'ambient_temperature': 'Ambient Temperature (°C)',
    'humidity': 'Humidity (%)'
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat in enumerate(numeric_features):
    for label, color in [(0, PALETTE[0]), (1, PALETTE[1])]:
        subset = df[df['failure_within_10_cycles'] == label][feat].dropna()
        axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                     label=f'{"No Failure" if label == 0 else "Failure"}',
                     density=True, edgecolor='none')
    axes[i].set_title(feature_labels[feat], fontweight='bold')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)
plt.suptitle('Sensor Distributions by Failure Status', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/02_feature_distributions.png", bbox_inches='tight')
plt.close()

# 3.3 Correlation matrix
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[numeric_features + ['failure_within_10_cycles']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 8}, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/03_correlation_matrix.png", bbox_inches='tight')
plt.close()

# 3.4 Failure rate by component
failure_by_component = df.groupby('component_id')['failure_within_10_cycles'].agg(['mean', 'sum', 'count']).reset_index()
failure_by_component.columns = ['Component', 'Failure Rate', 'Failures', 'Total']
failure_by_component = failure_by_component.sort_values('Failure Rate', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(failure_by_component['Component'], failure_by_component['Failure Rate'] * 100,
              color=PALETTE[:len(failure_by_component)], edgecolor='white', linewidth=1.5)
for bar, row in zip(bars, failure_by_component.itertuples()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'{row.Failures}/{row.Total}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Failure Rate by Component Type', fontweight='bold', fontsize=13)
ax.set_ylabel('Failure Rate (%)')
ax.set_xlabel('Component')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/04_failure_by_component.png", bbox_inches='tight')
plt.close()

# 3.5 Boxplots - key sensors vs failure
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
key_features = ['vibration_sensor', 'temperature_sensor_1', 'temperature_sensor_2',
                'pressure_sensor', 'fault_code_count', 'last_maintenance_cycles']
for i, feat in enumerate(key_features):
    data_fail = df[df['failure_within_10_cycles'] == 1][feat]
    data_nofail = df[df['failure_within_10_cycles'] == 0][feat]
    bp = axes[i].boxplot([data_nofail, data_fail],
                         labels=['No Failure', 'Failure'],
                         patch_artist=True,
                         medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(PALETTE[0])
    bp['boxes'][1].set_facecolor(PALETTE[1])
    axes[i].set_title(feature_labels[feat], fontweight='bold')
plt.suptitle('Key Sensor Readings by Failure Status', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_boxplots_by_failure.png", bbox_inches='tight')
plt.close()

# 3.6 Flight cycle trends for sample aircraft
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sample_aircraft = df['aircraft_id'].unique()[:4]
sensor_pairs = [
    ('flight_cycles', 'vibration_sensor', 'Vibration over Flight Cycles'),
    ('flight_cycles', 'temperature_sensor_1', 'Temp Sensor 1 over Flight Cycles'),
    ('flight_cycles', 'pressure_sensor', 'Pressure over Flight Cycles'),
    ('flight_cycles', 'fault_code_count', 'Fault Codes over Flight Cycles'),
]
for ax, (x, y, title) in zip(axes.flatten(), sensor_pairs):
    for ac, color in zip(sample_aircraft, PALETTE):
        subset = df[df['aircraft_id'] == ac].sort_values('flight_cycles')
        failure_pts = subset[subset['failure_within_10_cycles'] == 1]
        ax.plot(subset[x], subset[y], alpha=0.5, linewidth=1, color=color, label=ac)
        if len(failure_pts) > 0:
            ax.scatter(failure_pts[x], failure_pts[y], color='red', s=30, zorder=5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Flight Cycles')
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.legend(fontsize=7)
plt.suptitle('Sensor Trends Over Flight Cycles (Red dots = Failure Events)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/06_sensor_trends.png", bbox_inches='tight')
plt.close()

print("EDA figures saved.")


# ─── 4. Feature Engineering ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. FEATURE ENGINEERING")
print("=" * 60)

df_fe = df.copy()

# Sort for time-series operations
df_fe = df_fe.sort_values(['aircraft_id', 'component_id', 'flight_cycles']).reset_index(drop=True)

# Group key for per-aircraft-component operations
group_key = ['aircraft_id', 'component_id']

# 4.1 Rolling statistics (window = 3 cycles)
sensor_cols = ['temperature_sensor_1', 'temperature_sensor_2',
               'vibration_sensor', 'pressure_sensor', 'fault_code_count']

print("Computing rolling features...")
for col in sensor_cols:
    grp = df_fe.groupby(group_key)[col]
    df_fe[f'{col}_roll3_mean'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df_fe[f'{col}_roll3_std']  = grp.transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df_fe[f'{col}_roll5_mean'] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_fe[f'{col}_diff1']      = grp.transform(lambda x: x.diff().fillna(0))  # rate of change

# 4.2 Cumulative fault codes
df_fe['cumulative_faults'] = df_fe.groupby(group_key)['fault_code_count'].transform('cumsum')

# 4.3 Engine hours per flight cycle (efficiency proxy)
df_fe['hours_per_cycle'] = df_fe['engine_hours'] / df_fe['flight_cycles'].replace(0, np.nan)

# 4.4 Temperature differential
df_fe['temp_differential'] = df_fe['temperature_sensor_1'] - df_fe['temperature_sensor_2']

# 4.5 Sensor stress index (combined z-score proxy)
for col in ['vibration_sensor', 'pressure_sensor']:
    mean_val = df_fe[col].mean()
    std_val  = df_fe[col].std()
    df_fe[f'{col}_zscore'] = (df_fe[col] - mean_val) / std_val

df_fe['stress_index'] = (
    df_fe['vibration_sensor_zscore'].abs() +
    df_fe['pressure_sensor_zscore'].abs()
)

# 4.6 Maintenance urgency score
df_fe['maint_urgency'] = df_fe['last_maintenance_cycles'] * (1 + df_fe['fault_code_count'])

# 4.7 Interaction: vibration × temperature (stress under heat)
df_fe['vib_x_temp1'] = df_fe['vibration_sensor'] * df_fe['temperature_sensor_1']

# 4.8 Interaction: sensor_drift × vibration (drifting sensor under stress)
df_fe['drift_x_vib']  = df_fe['sensor_drift_flag'] * df_fe['vibration_sensor']
df_fe['drift_x_faults'] = df_fe['sensor_drift_flag'] * df_fe['fault_code_count']

# 4.9 Cumulative anomaly flag (rolling sum of fault codes over 5 cycles)
df_fe['fault_roll5_sum'] = df_fe.groupby(group_key)['fault_code_count'].transform(
    lambda x: x.rolling(5, min_periods=1).sum()
)

print(f"Features after engineering: {df_fe.shape[1]} columns")
new_features = [c for c in df_fe.columns if c not in df.columns]
print(f"New features created: {len(new_features)}")
print(new_features)


# ─── 5. Preprocessing ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PREPROCESSING")
print("=" * 60)

# Encode categoricals
le_aircraft  = LabelEncoder()
le_component = LabelEncoder()
df_fe['aircraft_id_enc']  = le_aircraft.fit_transform(df_fe['aircraft_id'])
df_fe['component_id_enc'] = le_component.fit_transform(df_fe['component_id'])

# Define final feature set (exclude non-features)
exclude_cols = ['aircraft_id', 'component_id', 'failure_within_10_cycles',
                'vibration_sensor_zscore', 'pressure_sensor_zscore']
feature_cols = [c for c in df_fe.columns if c not in exclude_cols]

# Drop rows where target is missing
df_fe = df_fe.dropna(subset=['failure_within_10_cycles']).reset_index(drop=True)

X = df_fe[feature_cols].copy()
y = df_fe['failure_within_10_cycles'].astype(int)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Handle any remaining NaN (from rolling operations at boundaries)
X = X.fillna(X.median())

# Train/test split — stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples, {y_train.sum()} failures")
print(f"Test set:  {X_test.shape[0]} samples, {y_test.sum()} failures")

# Scale features (for LR; tree models don't require it but doesn't hurt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─── 6. Model Training ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. MODEL TRAINING")
print("=" * 60)

# Calculate class weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Scale pos weight: {scale_pos_weight:.2f}")

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', max_depth=10,
        min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, use_label_encoder=False,
        eval_metric='aucpr', random_state=RANDOM_STATE, verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                                    scoring='average_precision', n_jobs=-1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='average_precision', n_jobs=-1)

    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    f1       = f1_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'avg_precision': avg_prec,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'cm': cm,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    print(f"  ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f} | "
          f"F1: {f1:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  CV Avg Precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─── 7. Model Evaluation & Visualizations ─────────────────────────────────────
print("\n" + "=" * 60)
print("7. MODEL EVALUATION")
print("=" * 60)

# 7.1 Metrics comparison table
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'ROC-AUC': [r['roc_auc'] for r in results.values()],
    'Avg Precision (PR-AUC)': [r['avg_precision'] for r in results.values()],
    'F1 Score': [r['f1'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'CV Avg Prec (mean)': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
}).set_index('Model').round(4)
print(f"\nMetrics Summary:\n{metrics_df}")
metrics_df.to_csv("model_metrics.csv")

# 7.2 ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for (name, res), color in zip(results.items(), PALETTE):
    # ROC
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[0].plot(fpr, tpr, lw=2, color=color,
                 label=f"{name} (AUC={res['roc_auc']:.3f})")
    # PR
    prec_c, rec_c, _ = precision_recall_curve(y_test, res['y_proba'])
    axes[1].plot(rec_c, prec_c, lw=2, color=color,
                 label=f"{name} (AP={res['avg_precision']:.3f})")

# Baselines
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].legend(fontsize=8)

base_ap = y_test.mean()
axes[1].axhline(y=base_ap, color='k', linestyle='--', lw=1,
                label=f'Random (AP={base_ap:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].legend(fontsize=8)

plt.suptitle('Model Performance Curves', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/07_roc_pr_curves.png", bbox_inches='tight')
plt.close()

# 7.3 Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm_norm = res['cm'].astype(float) / res['cm'].sum(axis=1)[:, np.newaxis]
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'],
                linewidths=0.5)
    ax.set_title(f'{name}\nF1={res["f1"]:.3f} | Prec={res["precision"]:.3f} | Rec={res["recall"]:.3f}',
                 fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/08_confusion_matrices.png", bbox_inches='tight')
plt.close()

# 7.4 Metrics bar chart
fig, ax = plt.subplots(figsize=(12, 6))
metric_names = ['ROC-AUC', 'Avg Precision (PR-AUC)', 'F1 Score', 'Precision', 'Recall']
x = np.arange(len(metric_names))
width = 0.2
for i, (name, res) in enumerate(results.items()):
    vals = [res['roc_auc'], res['avg_precision'], res['f1'], res['precision'], res['recall']]
    ax.bar(x + i * width, vals, width, label=name, color=PALETTE[i], alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontweight='bold')
ax.set_ylabel('Score')
ax.set_title('Model Comparison Across Metrics', fontweight='bold', fontsize=13)
ax.legend()
ax.set_ylim(0, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', lw=0.5)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/09_metrics_comparison.png", bbox_inches='tight')
plt.close()


# ─── 8. Best Model & Feature Importance ───────────────────────────────────────
print("\n" + "=" * 60)
print("8. BEST MODEL ANALYSIS")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['avg_precision'])
best_res  = results[best_name]
best_model = best_res['model']
print(f"Best model: {best_name} (Avg Precision = {best_res['avg_precision']:.4f})")

# Feature importance (tree-based)
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1],
                   color=PALETTE[0], alpha=0.85)
    ax.set_xlabel('Feature Importance (Gain)', fontweight='bold')
    ax.set_title(f'Top 20 Feature Importances — {best_name}', fontweight='bold', fontsize=13)
    for bar in bars:
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.4f}', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_feature_importance.png", bbox_inches='tight')
    plt.close()
    print(f"\nTop 10 features:\n{importance_df.head(10).to_string(index=False)}")
    importance_df.to_csv("feature_importance.csv", index=False)

# 8.1 SHAP values
print("\nComputing SHAP values (this may take a moment)...")
try:
    if 'XGBoost' in best_name or 'LightGBM' in best_name or 'Forest' in best_name:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Summary plot (beeswarm)
        plt.sca(axes[0])
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          max_display=15, show=False, plot_type='dot')
        axes[0].set_title(f'SHAP Summary — {best_name}', fontweight='bold')

        # Bar plot
        plt.sca(axes[1])
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          max_display=15, show=False, plot_type='bar')
        axes[1].set_title('SHAP Mean |Value|', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/11_shap_summary.png", bbox_inches='tight')
        plt.close()
        print("SHAP figure saved.")
except Exception as e:
    print(f"SHAP skipped: {e}")


# ─── 9. Threshold Optimization ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. THRESHOLD OPTIMIZATION")
print("=" * 60)

thresholds = np.arange(0.1, 0.9, 0.02)
threshold_metrics = []
for t in thresholds:
    y_pred_t = (best_res['y_proba'] >= t).astype(int)
    prec  = precision_score(y_test, y_pred_t, zero_division=0)
    rec   = recall_score(y_test, y_pred_t)
    f1    = f1_score(y_test, y_pred_t, zero_division=0)
    threshold_metrics.append({'threshold': t, 'precision': prec, 'recall': rec, 'f1': f1})

thresh_df = pd.DataFrame(threshold_metrics)
best_thresh_idx = thresh_df['f1'].idxmax()
best_thresh = thresh_df.loc[best_thresh_idx, 'threshold']
print(f"Optimal threshold (max F1): {best_thresh:.2f}")
print(thresh_df.loc[best_thresh_idx])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_df['threshold'], thresh_df['precision'], label='Precision', color=PALETTE[0], lw=2)
ax.plot(thresh_df['threshold'], thresh_df['recall'],    label='Recall',    color=PALETTE[1], lw=2)
ax.plot(thresh_df['threshold'], thresh_df['f1'],        label='F1 Score',  color=PALETTE[2], lw=2)
ax.axvline(x=best_thresh, color='gray', linestyle='--', lw=1.5,
           label=f'Optimal threshold ({best_thresh:.2f})')

# Add operational zones
ax.axvspan(0.1, 0.3, alpha=0.07, color='red', label='High Recall Zone\n(safety-critical)')
ax.axvspan(0.5, 0.9, alpha=0.07, color='blue', label='High Precision Zone\n(cost-reduction)')

ax.set_xlabel('Decision Threshold', fontweight='bold')
ax.set_ylabel('Score')
ax.set_title('Threshold Optimization for Maintenance Decision Support',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=8)
ax.set_xlim(0.1, 0.9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/12_threshold_optimization.png", bbox_inches='tight')
plt.close()


# ─── 10. Risk Scoring for Maintenance Decisions ────────────────────────────────
print("\n" + "=" * 60)
print("10. MAINTENANCE DECISION SUPPORT")
print("=" * 60)

# Apply best model to all data (for demonstration of the scoring system)
X_all = df_fe[feature_cols].fillna(df_fe[feature_cols].median())
if best_name == 'Logistic Regression':
    risk_scores = best_model.predict_proba(scaler.transform(X_all))[:, 1]
else:
    risk_scores = best_model.predict_proba(X_all)[:, 1]

df_fe['failure_risk_score'] = risk_scores

# Assign risk tier
def risk_tier(score):
    if score >= 0.6:
        return 'CRITICAL'
    elif score >= 0.35:
        return 'HIGH'
    elif score >= 0.15:
        return 'MEDIUM'
    else:
        return 'LOW'

df_fe['risk_tier'] = df_fe['failure_risk_score'].apply(risk_tier)

# Risk distribution
tier_counts = df_fe['risk_tier'].value_counts()
print(f"\nRisk tier distribution:\n{tier_counts}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

tier_order  = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
tier_colors = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60']
tier_vals   = [tier_counts.get(t, 0) for t in tier_order]

bars = axes[0].bar(tier_order, tier_vals, color=tier_colors, edgecolor='white', linewidth=2)
for bar, val in zip(bars, tier_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{val:,}\n({val/len(df_fe)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Risk Tier Distribution', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Number of Records')

axes[1].hist(risk_scores, bins=60, color=PALETTE[0], edgecolor='none', alpha=0.7)
axes[1].axvline(0.15, color=tier_colors[3], lw=2, linestyle='--', label='LOW/MEDIUM (0.15)')
axes[1].axvline(0.35, color=tier_colors[2], lw=2, linestyle='--', label='MEDIUM/HIGH (0.35)')
axes[1].axvline(0.60, color=tier_colors[0], lw=2, linestyle='--', label='HIGH/CRITICAL (0.60)')
axes[1].set_title('Distribution of Failure Risk Scores', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Predicted Failure Probability')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=8)

plt.suptitle('Predictive Maintenance Risk Scoring System', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/13_risk_scoring.png", bbox_inches='tight')
plt.close()

# Sample output for top at-risk components
top_risk = df_fe[df_fe['failure_within_10_cycles'] == 0].nlargest(10, 'failure_risk_score')[
    ['aircraft_id', 'component_id', 'flight_cycles', 'failure_risk_score', 'risk_tier',
     'vibration_sensor', 'fault_code_count', 'last_maintenance_cycles']
]
print(f"\nTop 10 at-risk components (currently not in failure window):\n{top_risk.to_string(index=False)}")

# Save final dataset with risk scores
df_fe[['aircraft_id', 'component_id', 'flight_cycles',
       'failure_within_10_cycles', 'failure_risk_score', 'risk_tier']].to_csv(
    "risk_scores_output.csv", index=False
)


# ─── 11. Save Model ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. SAVING ARTIFACTS")
print("=" * 60)

# Save model, scaler, feature list
import pickle
with open(f"{MODELS_DIR}/best_model.pkl", 'wb') as f:
    pickle.dump({'model': best_model, 'model_name': best_name,
                 'scaler': scaler, 'feature_cols': feature_cols,
                 'best_threshold': float(best_thresh),
                 'le_aircraft': le_aircraft, 'le_component': le_component}, f)
print(f"Model saved: {MODELS_DIR}/best_model.pkl")

# Save metrics
with open("final_results_summary.json", 'w') as f:
    summary = {}
    for name, res in results.items():
        summary[name] = {
            'roc_auc': round(res['roc_auc'], 4),
            'avg_precision': round(res['avg_precision'], 4),
            'f1': round(res['f1'], 4),
            'precision': round(res['precision'], 4),
            'recall': round(res['recall'], 4),
            'cv_mean_avg_precision': round(res['cv_mean'], 4),
            'cv_std': round(res['cv_std'], 4),
        }
    json.dump({'models': summary, 'best_model': best_name,
               'best_threshold': float(best_thresh)}, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nBest model: {best_name}")
print(f"  ROC-AUC:        {best_res['roc_auc']:.4f}")
print(f"  Avg Precision:  {best_res['avg_precision']:.4f}")
print(f"  F1 Score:       {best_res['f1']:.4f}")
print(f"  Recall:         {best_res['recall']:.4f}")
print(f"  Precision:      {best_res['precision']:.4f}")
print(f"\nAll figures saved to: {FIGURES_DIR}/")
print(f"Risk scores saved to: risk_scores_output.csv")
