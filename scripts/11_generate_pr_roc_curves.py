

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # change to 'TkAgg' in Spyder if you want
                               # interactive pop-up windows
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")


RES_DIR   = os.path.join('outputs', 'results')
FIG_DIR   = os.path.join('outputs', 'figures')
SYNTH_DIR = os.path.join('outputs', 'synthesiser')
os.makedirs(FIG_DIR, exist_ok=True)

TEST_SET_PATH  = os.path.join(RES_DIR, 'test_set.csv')
TRAIN_SET_PATH = os.path.join(RES_DIR, 'train_set.csv')
HP_PATH        = os.path.join(RES_DIR, 'best_hyperparameters.json')
SYNTH_PATH        = os.path.join(SYNTH_DIR, 'best_fraud_250k.csv')   # fraud-only (for Exp 3 augmentation)
SYNTH_300K_PATH   = os.path.join(SYNTH_DIR, 'ctgan_synthetic_300k.csv')  # full synth dataset (for Exp 2)


CURVE_SEED = 42

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'legend.fontsize':    9.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'figure.dpi':         150,
})

# Consistent colours across all figures
COLORS = {
    'LR':      '#5C6BC0',
    'RF':      '#2E7D32',
    'XGB':     '#E65100',
    'LGBM':    '#6A1B9A',
    'ResNet':  '#00838F',   # placeholder — ResNet needs its own script
    'TabNet':  '#F9A825',   # placeholder — TabNet needs its own script
    'TabPFN':  '#C62828',   # placeholder — TabPFN needs its own script
    'Exp1':    '#1565C0',
    'Exp2':    '#2E7D32',
    'Exp3':    '#E65100',
}



#Loading Data

print("LOADING DATA")


test  = pd.read_csv(TEST_SET_PATH)
train = pd.read_csv(TRAIN_SET_PATH)

X_test  = test.drop(columns=['Class']).values
y_test  = test['Class'].values
X_train = train.drop(columns=['Class']).values
y_train = train['Class'].values

print(f"  Test set  : {X_test.shape}  | fraud = {y_test.sum()}")
print(f"  Train set : {X_train.shape} | fraud = {y_train.sum()}")

FRAUD_RATE = y_test.sum() / len(y_test)
print(f"  Test fraud rate: {FRAUD_RATE:.4f}")

with open(HP_PATH) as f:
    best_hp = json.load(f)
print(f"  Hyperparameters loaded from: {HP_PATH}")


def make_classifiers(seed):
    """
    Returns the 4 classical classifiers with saved best hyperparameters.
    Deep learning models (TabNet, ResNet, TabPFN) are NOT included here
    as they require their own training pipelines.
    """
    lr_hp   = best_hp.get('LogisticRegression', {})
    rf_hp   = best_hp.get('RandomForest', {})
    xgb_hp  = best_hp.get('XGBoost', {})
    lgbm_hp = best_hp.get('LightGBM', {})

    # Strip keys we set explicitly to avoid "multiple values" errors
    lr_hp   = {k: v for k, v in lr_hp.items()
               if k not in ('random_state', 'max_iter', 'solver')}
    rf_hp   = {k: v for k, v in rf_hp.items()
               if k not in ('random_state', 'n_jobs')}
    xgb_hp  = {k: v for k, v in xgb_hp.items()
               if k not in ('random_state', 'eval_metric', 'verbosity',
                             'use_label_encoder')}
    lgbm_hp = {k: v for k, v in lgbm_hp.items()
               if k not in ('random_state', 'verbose')}

    clfs = {
        'LR': LogisticRegression(
            **lr_hp, random_state=seed, max_iter=1000, solver='lbfgs'
        ),
        'RF': RandomForestClassifier(
            **rf_hp, random_state=seed, n_jobs=-1
        ),
        'XGB': xgb.XGBClassifier(
            **xgb_hp, random_state=seed, eval_metric='logloss',
            verbosity=0
        ),
        'LGBM': lgb.LGBMClassifier(
            **lgbm_hp, random_state=seed, verbose=-1
        ),
    }
    return clfs


def get_y_prob(clf, X_tr, y_tr, X_te):
    """Fit classifier and return fraud probability scores on test set."""
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def plot_pr_roc(ax_pr, ax_roc, recall, precision, fpr, tpr,
                label_pr, label_roc, color, lw=2.2, ls='-', alpha=1.0):
    """Plot one PR curve and one ROC curve onto given axes."""
    pr_auc  = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    ax_pr.plot(recall, precision,
               color=color, linewidth=lw, linestyle=ls, alpha=alpha,
               label=f'{label_pr}  (AUC-PR = {pr_auc:.4f})')
    ax_roc.plot(fpr, tpr,
                color=color, linewidth=lw, linestyle=ls, alpha=alpha,
                label=f'{label_roc}  (AUC-ROC = {roc_auc:.4f})')
    return pr_auc, roc_auc


def finalise_axes(ax_pr, ax_roc, fraud_rate):
    """Add baselines, labels, and legends to PR and ROC axes.
    Legends are placed BELOW each subplot to avoid blocking the curves.
    """
    ax_pr.axhline(y=fraud_rate, color='black', linestyle=':', linewidth=1.2,
                  label=f'Random classifier (fraud rate = {fraud_rate:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='black', linestyle=':', linewidth=1.2,
                label='Random classifier (AUC = 0.5000)')

    ax_pr.set_xlabel('Recall');  ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve', fontweight='bold')
    ax_pr.set_xlim([0, 1]);      ax_pr.set_ylim([0, 1.05])
    ax_pr.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),   # place below the x-axis
        ncol=1,
        framealpha=0.9,
        fontsize=9,
        borderaxespad=0.
    )

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve', fontweight='bold')
    ax_roc.set_xlim([0, 1]);     ax_roc.set_ylim([0, 1.05])
    ax_roc.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),   # place below the x-axis
        ncol=1,
        framealpha=0.9,
        fontsize=9,
        borderaxespad=0.
    )


def save_fig(fig, fname):
    path = os.path.join(FIG_DIR, fname)
    # bottom margin gives space for below-axis legends
    fig.subplots_adjust(bottom=0.32, wspace=0.35)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")




print("FIGURE 1 — Experiment 1: Real Data, No Handling, 4 Classifiers")


fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle(
    'Experiment 1 — Real Data Baseline (No Handling)\n'
    'PR and ROC Curves — 4 Classical Classifiers',
    fontsize=13, fontweight='bold', y=1.01
)

clfs = make_classifiers(CURVE_SEED)
exp1_probs = {}   # store for later use in cross-experiment figure

for name, clf in clfs.items():
    print(f"  Fitting {name}...", end=' ')
    y_prob = get_y_prob(clf, X_train, y_train, X_test)
    exp1_probs[name] = y_prob

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _          = roc_curve(y_test, y_prob)

    plot_pr_roc(ax_pr, ax_roc,
                recall, precision, fpr, tpr,
                label_pr=name, label_roc=name,
                color=COLORS[name])
    print("done")

finalise_axes(ax_pr, ax_roc, FRAUD_RATE)
save_fig(fig, 'fig_exp1_pr_roc.png')



print("FIGURE 2 — Experiment 2: Synthetic-Only, Top Classical Models")


# Load FULL synthetic dataset (both classes) for Experiment 2
# Note: best_fraud_250k.csv is fraud-ONLY — that is for Exp 3 augmentation only
synth = pd.read_csv(SYNTH_300K_PATH)
print(f"  Synthetic data loaded: {synth.shape}")

feature_cols = [c for c in train.columns if c != 'Class']
X_synth_raw  = synth[feature_cols].values
y_synth      = synth['Class'].values
print(f"  Synthetic fraud: {y_synth.sum()} / {len(y_synth)}")
print(f"  Synthetic legit: {(y_synth==0).sum()} / {len(y_synth)}")
assert y_synth.sum() > 0 and (y_synth==0).sum() > 0, \
    "Synthetic dataset must contain both classes for Experiment 2"
# Also load fraud-only file for use in Exp 3 augmentation later
synth_fraud_df_raw = pd.read_csv(SYNTH_PATH)
print(f"  Fraud-only file loaded for Exp 3: {synth_fraud_df_raw.shape}")

# Scale synthetic data using the same scaler parameters (refit on synthetic train)
# Note: ideally use the saved scaler.pkl — loading it here for correctness
scaler_path = os.path.join('outputs', 'models', 'standard_scaler.pkl')
if os.path.exists(scaler_path):
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_synth = scaler.transform(X_synth_raw)
    print(f"  Loaded saved scaler from: {scaler_path}")
else:
    # Fallback: refit scaler on synthetic data (slightly different but acceptable)
    scaler = StandardScaler()
    X_synth = scaler.fit_transform(X_synth_raw)
    print("  WARNING: Saved scaler not found. Refitting on synthetic data.")
    print("           For best accuracy, ensure outputs/models/standard_scaler.pkl exists.")

# Build synthetic + SMOTE conditions
exp2_conditions = {
    'LR — No Handling':  ('LR',   X_synth, y_synth, None),
    'XGB — No Handling': ('XGB',  X_synth, y_synth, None),
    'RF — No Handling':  ('RF',   X_synth, y_synth, None),
    'LGBM — No Handling':('LGBM', X_synth, y_synth, None),
}

exp2_colors = {
    'LR — No Handling':   COLORS['LR'],
    'XGB — No Handling':  COLORS['XGB'],
    'RF — No Handling':   COLORS['RF'],
    'LGBM — No Handling': COLORS['LGBM'],
}

fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle(
    'Experiment 2 — Synthetic-Only Training\n'
    'PR and ROC Curves — Top Classical Models',
    fontsize=13, fontweight='bold', y=1.01
)

exp2_probs = {}

for label, (clf_name, X_tr, y_tr, _) in exp2_conditions.items():
    print(f"  Fitting {label}...", end=' ')
    clf    = make_classifiers(CURVE_SEED)[clf_name]
    y_prob = get_y_prob(clf, X_tr, y_tr, X_test)
    exp2_probs[label] = y_prob

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _          = roc_curve(y_test, y_prob)

    plot_pr_roc(ax_pr, ax_roc,
                recall, precision, fpr, tpr,
                label_pr=label, label_roc=label,
                color=exp2_colors[label])
    print("done")

# Add Exp1 RF reference dashed line
precision_ref, recall_ref, _ = precision_recall_curve(y_test, exp1_probs['RF'])
fpr_ref, tpr_ref, _          = roc_curve(y_test, exp1_probs['RF'])
pr_auc_ref  = auc(recall_ref, precision_ref)
roc_auc_ref = auc(fpr_ref, tpr_ref)
ax_pr.plot(recall_ref, precision_ref, color=COLORS['RF'],
           linewidth=1.6, linestyle='--', alpha=0.55,
           label=f'Exp 1 RF No Handling (ref, AUC-PR = {pr_auc_ref:.4f})')
ax_roc.plot(fpr_ref, tpr_ref, color=COLORS['RF'],
            linewidth=1.6, linestyle='--', alpha=0.55,
            label=f'Exp 1 RF No Handling (ref, AUC-ROC = {roc_auc_ref:.4f})')

finalise_axes(ax_pr, ax_roc, FRAUD_RATE)
save_fig(fig, 'fig_exp2_pr_roc.png')



print("FIGURE 3 — Experiment 3: Augmented Training, Best Ratio per Model")




# Build augmented training sets
# Real train fraud count
real_fraud_count = int(y_train.sum())
real_legit_count = int((y_train == 0).sum())

synth_fraud_df = synth_fraud_df_raw[feature_cols]  # use dedicated fraud-only file

def build_augmented(ratio, seed):
    """
    Augment real training set by injecting synthetic fraud until
    the fraud proportion in the training set equals `ratio`.
    Returns X_aug, y_aug (scaled, same scaler as original training).
    """
    # Real training data (already scaled in train_set.csv)
    X_real = X_train.copy()
    y_real = y_train.copy()

    total_rows  = len(y_real)
    legit_count  = int((y_real == 0).sum())
    # Target fraud count to reach `ratio`
    target_fraud = int(np.round(ratio * total_rows / (1 - ratio)))
    synth_needed = max(0, target_fraud - real_fraud_count)

    if synth_needed == 0:
        return X_real, y_real

    # Sample synthetic fraud rows (with replacement if needed)
    sample = synth_fraud_df.sample(
        n=min(synth_needed, len(synth_fraud_df)),
        replace=(synth_needed > len(synth_fraud_df)),
        random_state=seed
    )
    X_synth_fraud = scaler.transform(sample.values)
    y_synth_fraud = np.ones(len(X_synth_fraud), dtype=int)

    X_aug = np.vstack([X_real, X_synth_fraud])
    y_aug = np.concatenate([y_real, y_synth_fraud])
    return X_aug, y_aug


exp3_conditions = {
    'RF 5% Aug':   ('RF',   0.05),
    'XGB 1% Aug':  ('XGB',  0.01),
    'LGBM 1% Aug': ('LGBM', 0.01),
    'LR 1% Aug':   ('LR',   0.01),
}

exp3_colors = {
    'RF 5% Aug':   COLORS['RF'],
    'XGB 1% Aug':  COLORS['XGB'],
    'LGBM 1% Aug': COLORS['LGBM'],
    'LR 1% Aug':   COLORS['LR'],
}

fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle(
    'Experiment 3 — CTGAN Augmented Training\n'
    'PR and ROC Curves — Best Ratio per Classical Model',
    fontsize=13, fontweight='bold', y=1.01
)

exp3_probs = {}

for label, (clf_name, ratio) in exp3_conditions.items():
    print(f"  Building {ratio*100:.0f}% augmented set for {clf_name}...", end=' ')
    X_aug, y_aug = build_augmented(ratio, CURVE_SEED)
    print(f"{len(y_aug)} rows, {int(y_aug.sum())} fraud...", end=' ')

    clf    = make_classifiers(CURVE_SEED)[clf_name]
    y_prob = get_y_prob(clf, X_aug, y_aug, X_test)
    exp3_probs[label] = y_prob

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _          = roc_curve(y_test, y_prob)

    plot_pr_roc(ax_pr, ax_roc,
                recall, precision, fpr, tpr,
                label_pr=label, label_roc=label,
                color=exp3_colors[label])
    print("done")

# Add Exp1 RF reference
ax_pr.plot(recall_ref, precision_ref, color='grey',
           linewidth=1.6, linestyle='--', alpha=0.6,
           label=f'Exp 1 RF No Handling (ref, AUC-PR = {pr_auc_ref:.4f})')
ax_roc.plot(fpr_ref, tpr_ref, color='grey',
            linewidth=1.6, linestyle='--', alpha=0.6,
            label=f'Exp 1 RF No Handling (ref, AUC-ROC = {roc_auc_ref:.4f})')

finalise_axes(ax_pr, ax_roc, FRAUD_RATE)
save_fig(fig, 'fig_exp3_pr_roc.png')



print("FIGURE 4 — Cross-Experiment: Best Model per Experiment")




cross_conditions = {
    'Exp 1: RF No Handling':  (exp1_probs['RF'],       COLORS['Exp1']),
    'Exp 2: LR No Handling':  (exp2_probs['LR — No Handling'], COLORS['Exp2']),
    'Exp 3: RF 5% Aug':       (exp3_probs['RF 5% Aug'],COLORS['Exp3']),
}

fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle(
    'Cross-Experiment Comparison — Best Model per Experiment\n'
    'PR and ROC Curves',
    fontsize=13, fontweight='bold', y=1.01
)

for label, (y_prob, color) in cross_conditions.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _          = roc_curve(y_test, y_prob)
    plot_pr_roc(ax_pr, ax_roc,
                recall, precision, fpr, tpr,
                label_pr=label, label_roc=label,
                color=color, lw=2.8)
    print(f"  Plotted: {label}")

finalise_axes(ax_pr, ax_roc, FRAUD_RATE)
save_fig(fig, 'fig_cross_experiment_pr_roc.png')



print(f"\nOutput figures saved to:  {os.path.abspath(FIG_DIR)}")


