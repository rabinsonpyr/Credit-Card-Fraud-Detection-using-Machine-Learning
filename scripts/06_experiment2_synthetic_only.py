

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, json, warnings, pickle
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, average_precision_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb

# Configuration 
# Automatically read best synthesiser from quality check recommendation
_BEST_SYN_FILE = os.path.join('outputs', 'synthesiser', 'best_synthesiser.txt')
if not os.path.exists(_BEST_SYN_FILE):
    print("ERROR: best_synthesiser.txt not found. Run 04_synthesiser_quality_check.py first.")
    sys.exit(1)

with open(_BEST_SYN_FILE, 'r') as _f:
    _BEST_SYN_NAME = _f.read().strip()

_FILENAME_MAP = {
    'GaussianCopula': 'gc_synthetic_300k.csv',
    'CTGAN':          'ctgan_synthetic_300k.csv',
    'CopulaGAN':      'copulagan_synthetic_300k.csv'
}

if _BEST_SYN_NAME not in _FILENAME_MAP:
    print(f"ERROR: Unrecognised synthesiser name in best_synthesiser.txt: '{_BEST_SYN_NAME}'")
    sys.exit(1)

SYNTHETIC_FILE = os.path.join('outputs', 'synthesiser', _FILENAME_MAP[_BEST_SYN_NAME])
print(f"Best synthesiser (from quality check): {_BEST_SYN_NAME}")
print(f"Synthetic training file: {SYNTHETIC_FILE}")
SCALER_FILE    = os.path.join('outputs', 'models',   'standard_scaler.pkl')
HP_FILE        = os.path.join('outputs', 'results',  'best_hyperparameters.json')
TEST_SET_FILE  = os.path.join('outputs', 'results',  'test_set.csv')
REAL_TRAIN_FILE= os.path.join('outputs', 'results',  'train_set.csv')

# Expected test set values from Experiment 1 
EXPECTED_TEST_ROWS  = 56746
EXPECTED_TEST_FRAUD = 95

SEEDS        = [42, 123, 456, 789, 1234]
RANDOM_STATE = 42

OUT_DIR = 'outputs'
RES_DIR = os.path.join(OUT_DIR, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')

for d in [RES_DIR, FIG_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)

PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURES + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL   = 'Class'
SEP = "=" * 60

def evaluate(y_true, y_pred, y_prob):
    """Binary F1 (fraud class), AUC-PR, MCC, Precision, Recall, AUC-ROC, FPR."""
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    return {
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'AUC_PR':    average_precision_score(y_true, y_prob),
        'MCC':       matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'AUC_ROC':   roc_auc_score(y_true, y_prob),
        'FPR': fpr, 'TP':tp, 'FP':fp, 'TN':tn, 'FN':fn
    }

def print_metrics(name, m):
    print(f"\n  {name}")
    print(f"    F1 (binary, fraud class) : {m['F1']*100:.2f}%")
    print(f"    AUC-PR                   : {m['AUC_PR']*100:.2f}%")
    print(f"    MCC                      : {m['MCC']*100:.2f}%")
    print(f"    Precision                : {m['Precision']*100:.2f}%")
    print(f"    Recall                   : {m['Recall']*100:.2f}%")
    print(f"    FPR                      : {m['FPR']*100:.2f}%  ({m['FP']} false alarms)")
    print(f"    TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

# SECTION 1 — Load dependencies 
print(SEP)
print("EXPERIMENT 2 — SYNTHETIC-ONLY TRAINING")
print(SEP)
print("\nSECTION 1 — LOADING EXPERIMENT 1 DEPENDENCIES")
print(SEP)

for path in [SYNTHETIC_FILE, SCALER_FILE, HP_FILE, TEST_SET_FILE, REAL_TRAIN_FILE]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)

# Load test set
test_df = pd.read_csv(TEST_SET_FILE)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

print(f"Test set loaded : {len(y_test):,} transactions")
print(f"  Fraud         : {y_test.sum()} ({y_test.mean()*100:.4f}%)")
print(f"  Legitimate    : {(y_test==0).sum():,}")

# Sanity check — verify test set matches Experiment 1
assert len(y_test) == EXPECTED_TEST_ROWS, \
    f"SANITY FAIL: Test set has {len(y_test)} rows, expected {EXPECTED_TEST_ROWS}. " \
    f"Re-run 02_experiment1_real_data.py or update EXPECTED_TEST_ROWS."
assert y_test.sum() == EXPECTED_TEST_FRAUD, \
    f"SANITY FAIL: Test set has {y_test.sum()} fraud cases, expected {EXPECTED_TEST_FRAUD}. " \
    f"Re-run 02_experiment1_real_data.py or update EXPECTED_TEST_FRAUD."
print(f"  ✓ Sanity: test set matches Experiment 1 ({EXPECTED_TEST_ROWS} rows, {EXPECTED_TEST_FRAUD} fraud)")
print("  NOTE: This is the REAL held-out test set — never used in training")

with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)
with open(HP_FILE) as f:
    best_params = json.load(f)
print("Scaler loaded (transform only — no refit).")
print("Tuned hyperparameters loaded.")

real_train = pd.read_csv(REAL_TRAIN_FILE)
real_imbalance = int((real_train['Class']==0).sum()) / int((real_train['Class']==1).sum())
print(f"Real data imbalance ratio: {real_imbalance:.1f}")

# SECTION 2 — Load synthetic training data 
print(f"\nSECTION 2 — LOADING SYNTHETIC TRAINING DATA")
print(SEP)

syn_df = pd.read_csv(SYNTHETIC_FILE)
print(f"Synthetic dataset loaded: {os.path.basename(SYNTHETIC_FILE)}")
print(f"Shape: {syn_df.shape}")

assert set(FEATURE_COLS + [TARGET_COL]).issubset(set(syn_df.columns)), \
    "SANITY FAIL: Synthetic dataset missing required columns"
assert syn_df.isnull().sum().sum() == 0, \
    "SANITY FAIL: Synthetic dataset contains missing values"
assert syn_df[TARGET_COL].nunique() == 2, \
    "SANITY FAIL: Synthetic dataset Class column should have 2 values"
print(f"  Fraud rows : {(syn_df[TARGET_COL]==1).sum():,}")
print(f"  Legit rows : {(syn_df[TARGET_COL]==0).sum():,}")
print("  ✓ Sanity: required columns present")
print("  ✓ Sanity: no missing values")

# Leakage check — no synthetic row should exactly match a real training row
real_fraud_rows = set(
    real_train[real_train['Class']==1][FEATURE_COLS].round(6).apply(
        lambda r: tuple(r), axis=1)
)
syn_fraud_rows = set(
    syn_df[syn_df[TARGET_COL]==1][FEATURE_COLS].round(6).apply(
        lambda r: tuple(r), axis=1).head(1000)
)
overlap = real_fraud_rows & syn_fraud_rows
assert len(overlap) == 0, \
    f"SANITY FAIL: {len(overlap)} synthetic rows exactly match real training rows (leakage)"
print("  ✓ Sanity: no real rows in synthetic data (leakage check passed)")

# Feature engineering if not already present
if 'log_Amount' not in syn_df.columns:
    syn_df['log_Amount'] = np.log1p(syn_df['Amount'])
    hour = (syn_df['Time'] / 3600) % 24
    syn_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    syn_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

X_syn_raw = syn_df[FEATURE_COLS].values
y_syn     = syn_df[TARGET_COL].values

X_syn = scaler.transform(X_syn_raw)

assert not np.any(np.isnan(X_syn)), "SANITY FAIL: NaN in scaled synthetic data"
assert not np.any(np.isinf(X_syn)), "SANITY FAIL: Inf in scaled synthetic data"
print(f"  ✓ Sanity: scaled with real-data scaler, no NaN/Inf")
print(f"\nSynthetic training pool: {(y_syn==1).sum():,} fraud, {(y_syn==0).sum():,} legit")
print("NOTE: No real training data used in classifiers — synthetic only")

# SECTION 3 — Build models 
def build_models(params, seed):
    return {
        'LR':   LogisticRegression(**params['LogisticRegression'], random_state=seed),
        'RF':   RandomForestClassifier(**params['RandomForest'], random_state=seed, n_jobs=-1),
        'XGB':  xgb.XGBClassifier(**params['XGBoost'], random_state=seed,
                                   eval_metric='aucpr', use_label_encoder=False, verbosity=0),
        'LGBM': lgb.LGBMClassifier(**params['LightGBM'], random_state=seed, verbosity=-1)
    }

def build_cs_models(params, seed, spw):
    return {
        'LR':   LogisticRegression(**params['LogisticRegression'],
                                   class_weight='balanced', random_state=seed),
        'RF':   RandomForestClassifier(**params['RandomForest'],
                                       class_weight='balanced', random_state=seed, n_jobs=-1),
        'XGB':  xgb.XGBClassifier(**params['XGBoost'], scale_pos_weight=spw,
                                   random_state=seed, eval_metric='aucpr',
                                   use_label_encoder=False, verbosity=0),
        'LGBM': lgb.LGBMClassifier(**params['LightGBM'], class_weight='balanced',
                                    random_state=seed, verbosity=-1)
    }

# SECTION 4 — Run experiments 
conditions = {
    '2A_NoHandling':     'No Handling',
    '2B_SMOTE':          'SMOTE',
    '2C_RUS':            'Random Undersampling',
    '2D_CostSensitive':  'Cost-Sensitive',
    '2E_ADASYN':         'ADASYN',
}

all_results = []

for cond_key, cond_name in conditions.items():
    print(f"\n{SEP}")
    print(f"EXPERIMENT {cond_key} — {cond_name}")
    print(f"Train: Synthetic only | Test: Real held-out")
    print(SEP)

    seed_results = {clf: [] for clf in ['LR','RF','XGB','LGBM']}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        if cond_key == '2A_NoHandling':
            X_tr, y_tr = X_syn.copy(), y_syn.copy()

        elif cond_key == '2B_SMOTE':
            print(f"Before SMOTE : {Counter(y_syn)}")
            sm = SMOTE(random_state=seed)
            X_tr, y_tr = sm.fit_resample(X_syn, y_syn)
            print(f"After SMOTE  : {Counter(y_tr)}")

        elif cond_key == '2C_RUS':
            print(f"Before RUS : {Counter(y_syn)}")
            rus = RandomUnderSampler(random_state=seed)
            X_tr, y_tr = rus.fit_resample(X_syn, y_syn)
            print(f"After RUS  : {Counter(y_tr)}")

        elif cond_key == '2D_CostSensitive':
            X_tr, y_tr = X_syn.copy(), y_syn.copy()

        elif cond_key == '2E_ADASYN':
            print(f"Before ADASYN : {Counter(y_syn)}")
            ada = ADASYN(random_state=seed)
            X_tr, y_tr = ada.fit_resample(X_syn, y_syn)
            print(f"After ADASYN  : {Counter(y_tr)}")

        assert not np.any(np.isnan(X_tr)), \
            f"SANITY FAIL: NaN in training data after {cond_name}"

        n_fraud_tr = (y_tr==1).sum()
        n_legit_tr = (y_tr==0).sum()
        spw = n_legit_tr / n_fraud_tr if n_fraud_tr > 0 else 1

        if cond_key == '2D_CostSensitive':
            models = build_cs_models(best_params, seed, spw)
        else:
            models = build_models(best_params, seed)

        for clf_name, clf in models.items():
            label = f"{clf_name} (Syn, {cond_name})"
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_pred, y_prob)
            seed_results[clf_name].append(m)
            print_metrics(label, m)
            all_results.append({
                'Condition': cond_key, 'Condition_Name': cond_name,
                'Classifier': clf_name, 'Seed': seed, **m
            })

    print(f"\n=== EXPERIMENT {cond_key} SUMMARY (mean ± std across 5 seeds) ===")
    print(f"  NOTE: F1 is binary (positive class = fraud). Not micro or macro averaged.")
    for clf_name, metrics_list in seed_results.items():
        f1s  = [m['F1']     for m in metrics_list]
        aprs = [m['AUC_PR'] for m in metrics_list]
        mccs = [m['MCC']    for m in metrics_list]
        label = f"{clf_name} + {cond_name} (Syn)"
        print(f"  {label:<40}: "
              f"F1={np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%  "
              f"MCC={np.mean(mccs)*100:.2f}%  AUC-PR={np.mean(aprs)*100:.2f}%")

    pd.DataFrame([r for r in all_results if r['Condition']==cond_key]).to_csv(
        os.path.join(RAW_DIR, f'{cond_key.lower()}_raw.csv'), index=False)

# SECTION 5 — Master results 
print(f"\n{SEP}")
print("SECTION 5 — MASTER RESULTS TABLE")
print(SEP)

results_df = pd.DataFrame(all_results)
master = results_df.groupby(['Condition_Name','Classifier']).agg(
    F1_mean=('F1','mean'), F1_std=('F1','std'),
    AUC_PR_mean=('AUC_PR','mean'), AUC_PR_std=('AUC_PR','std'),
    MCC_mean=('MCC','mean'), MCC_std=('MCC','std'),
    Precision_mean=('Precision','mean'),
    Recall_mean=('Recall','mean'),
    AUC_ROC_mean=('AUC_ROC','mean'),
    FPR_mean=('FPR','mean'),
).reset_index()
master['Model'] = master['Classifier'] + ' + ' + master['Condition_Name'] + ' (Syn)'
master = master.sort_values('F1_mean', ascending=False).reset_index(drop=True)

print("\n=== MASTER RESULTS TABLE — ALL SYNTHETIC TRAINING EXPERIMENTS ===")
print("  NOTE: F1 is binary (positive class = fraud). Not micro or macro averaged.")
for _, row in master.iterrows():
    print(f"  {row['Model']:<45} "
          f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
          f"AUC-PR={row['AUC_PR_mean']*100:.2f}%  MCC={row['MCC_mean']*100:.2f}%")

master.to_csv(os.path.join(RES_DIR, 'master_results_synthetic.csv'), index=False)

# SECTION 6 — Exp 1 vs Exp 2 comparison 
print(f"\n{SEP}")
print("SECTION 6 — EXPERIMENT 1 vs EXPERIMENT 2 COMPARISON (RQ2)")
print(SEP)

exp1 = pd.read_csv(os.path.join(RES_DIR, 'master_results_real_data.csv'))

comparison_rows = []
for tech in ['No Handling','SMOTE','Random Undersampling','Cost-Sensitive','ADASYN']:
    e1_rows = exp1[exp1['Condition_Name'] == tech] \
              if 'Condition_Name' in exp1.columns \
              else exp1[exp1['Model'].str.contains(tech)]
    e2_rows = master[master['Condition_Name'] == tech]

    e1_best = e1_rows['F1_mean'].max() if len(e1_rows) > 0 else np.nan
    e2_best = e2_rows['F1_mean'].max() if len(e2_rows) > 0 else np.nan
    gap     = e1_best - e2_best if not (np.isnan(e1_best) or np.isnan(e2_best)) else np.nan
    gap_pct = (gap / e1_best * 100) if (e1_best and e1_best > 0) else np.nan

    comparison_rows.append({
        'Technique':                    tech,
        'Exp1_Best_F1_%_Real':          round(e1_best*100, 2),
        'Exp2_Best_F1_%_Synthetic':     round(e2_best*100, 2),
        'F1_Gap_%points':               round(gap*100, 2),
        'Gap_Relative_%':               round(gap_pct, 1)
    })
    print(f"  {tech:<25}: Real={e1_best*100:.2f}%  Syn={e2_best*100:.2f}%  "
          f"Gap={gap*100:.2f}pp ({gap_pct:.1f}%)")

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(RES_DIR, 'exp1_vs_exp2_comparison.csv'), index=False)
print(f"\nSaved: {os.path.join(RES_DIR, 'exp1_vs_exp2_comparison.csv')}")

print(f"\n{SEP}")
print("ALL DONE — EXPERIMENT 2")

