

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, json, time, warnings, pickle
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

#Configuration 
SYNTH_FRAUD_FILE = os.path.join('outputs', 'synthesiser', 'best_fraud_250k.csv')
SCALER_FILE      = os.path.join('outputs', 'models',     'standard_scaler.pkl')
HP_FILE          = os.path.join('outputs', 'results',    'best_hyperparameters.json')
TEST_SET_FILE    = os.path.join('outputs', 'results',    'test_set.csv')
REAL_TRAIN_FILE  = os.path.join('outputs', 'results',    'train_set.csv')

# Expected test set values from Experiment 1 — update if Exp 1 is re-run
EXPECTED_TEST_ROWS  = 56746
EXPECTED_TEST_FRAUD = 95

AUGMENTATION_RATIOS = [0.01, 0.05, 0.10, 0.25, 0.50]
SEEDS               = [42, 123, 456, 789, 1234]

OUT_DIR = 'outputs'
RES_DIR = os.path.join(OUT_DIR, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')

for d in [RES_DIR, FIG_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)

PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURES + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL   = 'Class'


#Helper
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
    print(f"  {name}")
    print(f"    F1 (binary, fraud class) : {m['F1']*100:.2f}%  "
          f"AUC-PR: {m['AUC_PR']*100:.2f}%  "
          f"MCC: {m['MCC']*100:.2f}%  "
          f"FPR: {m['FPR']*100:.2f}%")

#Load dependencies 

print("EXPERIMENT 3 — SYNTHETIC AUGMENTATION + BENCHMARKS")
print(f"Augmentation ratios: {[f'{r*100:.0f}%' for r in AUGMENTATION_RATIOS]}")


for path in [SYNTH_FRAUD_FILE, SCALER_FILE, HP_FILE, TEST_SET_FILE, REAL_TRAIN_FILE]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)

# Load test set
test_df = pd.read_csv(TEST_SET_FILE)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values
print(f"Test set loaded : {len(y_test):,} transactions")
print(f"  Fraud         : {y_test.sum()} ({y_test.mean()*100:.4f}%)")

# Sanity check — verify test set matches Experiment 1
assert len(y_test) == EXPECTED_TEST_ROWS, \
    f"SANITY FAIL: Test set has {len(y_test)} rows, expected {EXPECTED_TEST_ROWS}. " \
    f"Re-run 02_experiment1_real_data.py or update EXPECTED_TEST_ROWS."
assert y_test.sum() == EXPECTED_TEST_FRAUD, \
    f"SANITY FAIL: Test set has {y_test.sum()} fraud cases, expected {EXPECTED_TEST_FRAUD}. " \
    f"Re-run 02_experiment1_real_data.py or update EXPECTED_TEST_FRAUD."
print(f"  ✓ Sanity: test set matches Experiment 1 "
      f"({EXPECTED_TEST_ROWS} rows, {EXPECTED_TEST_FRAUD} fraud)")

# Load real training set
real_train   = pd.read_csv(REAL_TRAIN_FILE)
X_real_train = real_train[FEATURE_COLS].values
y_real_train = real_train[TARGET_COL].values
n_real_legit = int((y_real_train==0).sum())
n_real_fraud = int((y_real_train==1).sum())
print(f"\nReal training set: {len(y_real_train):,} rows")
print(f"  Legitimate : {n_real_legit:,}  ({n_real_legit/len(y_real_train)*100:.3f}%)")
print(f"  Fraud      : {n_real_fraud}  ({n_real_fraud/len(y_real_train)*100:.3f}%)")

with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)
with open(HP_FILE) as f:
    best_params = json.load(f)
print("Scaler and hyperparameters loaded.")

scale_pos_weight = n_real_legit / n_real_fraud

#Load synthetic fraud pool
print(f"\nLOADING SYNTHETIC FRAUD POOL")


synth_df = pd.read_csv(SYNTH_FRAUD_FILE)
print(f"Synthetic fraud file: {os.path.basename(SYNTH_FRAUD_FILE)}")
print(f"Shape: {synth_df.shape}")

# Sanity checks on fraud pool
assert (synth_df[TARGET_COL] == 1).all(), \
    f" Fraud pool contains non-fraud rows."
assert synth_df.isnull().sum().sum() == 0, \
    "Fraud pool contains missing values"
print(f" all {len(synth_df):,} rows confirmed Class=1 (fraud only)")
print(" no missing values")

max_ratio   = max(AUGMENTATION_RATIOS)
max_needed  = int((max_ratio * n_real_legit) / (1 - max_ratio)) - n_real_fraud
assert len(synth_df) >= max_needed, \
    f"Fraud pool ({len(synth_df):,}) insufficient for {max_ratio*100:.0f}% ratio"
print(f"pool ({len(synth_df):,}) sufficient for all ratios")
print(f"Max needed at {max_ratio*100:.0f}% ratio: {max_needed:,}")

real_fraud_sample = set(
    real_train[real_train['Class']==1][FEATURE_COLS].round(6).apply(
        lambda r: tuple(r), axis=1)
)
synth_sample = set(
    synth_df[FEATURE_COLS].round(6).head(1000).apply(
        lambda r: tuple(r), axis=1)
)
overlap = real_fraud_sample & synth_sample
assert len(overlap) == 0, \
    f"SANITY FAIL: {len(overlap)} synthetic rows match real fraud rows (leakage)"
print("  ✓ Sanity: no real rows in synthetic pool (leakage check passed)")

if 'log_Amount' not in synth_df.columns:
    synth_df['log_Amount'] = np.log1p(synth_df['Amount'])
    hour = (synth_df['Time'] / 3600) % 24
    synth_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    synth_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

X_synth = scaler.transform(synth_df[FEATURE_COLS].values)
print(f"\nSynthetic fraud pool ready: {len(synth_df):,} cases")

#Model builders
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

all_results  = []
ratio_results = []

# Augmentation ratio sensitivity
print("\nAUGMENTATION RATIO SENSITIVITY")


for ratio in AUGMENTATION_RATIOS:
    print(f"\n{'='*50}")
    print(f"  Target fraud rate: {ratio*100:.0f}%")
    print(f"{'='*50}")

    n_syn_needed = int((ratio * n_real_legit) / (1 - ratio)) - n_real_fraud
    n_syn_needed = max(0, min(n_syn_needed, len(synth_df)))

    seed_results = {clf: [] for clf in ['LR','RF','XGB','LGBM']}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        syn_idx      = rng.choice(len(X_synth), size=n_syn_needed, replace=False)
        X_syn_sample = X_synth[syn_idx]
        y_syn_sample = np.ones(n_syn_needed, dtype=int)

        X_combined = np.vstack([X_real_train, X_syn_sample])
        y_combined = np.concatenate([y_real_train, y_syn_sample])
        actual_ratio = y_combined.mean()

        if seed == SEEDS[0]:
            print(f"\n  Seed {seed} — Combined training set:")
            print(f"    Total            : {len(y_combined):,}")
            print(f"    Fraud            : {y_combined.sum():,}  ({actual_ratio*100:.3f}%)")
            print(f"    Synthetic added  : {n_syn_needed:,}")
            assert abs(actual_ratio - ratio) < 0.005, \
                f"SANITY FAIL: Actual ratio {actual_ratio:.4f} ≠ target {ratio}"
            assert not np.any(np.isnan(X_combined)), \
                "SANITY FAIL: NaN in augmented training data"
            print(f"    ✓ Sanity: {ratio*100:.0f}% ratio achieved")

        models = build_models(best_params, seed)
        for clf_name, clf in models.items():
            clf.fit(X_combined, y_combined)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_pred, y_prob)
            seed_results[clf_name].append(m)
            print_metrics(f"  {clf_name} (ratio={ratio*100:.0f}%)", m)
            all_results.append({
                'Condition': '3A_SynthAug',
                'Condition_Name': f'Synth_Aug_{ratio*100:.0f}pct',
                'Augmentation_Ratio': ratio,
                'Classifier': clf_name, 'Seed': seed, **m
            })

    print(f"\n  Ratio {ratio*100:.0f}% — Mean F1 by classifier:")
    for clf_name, metrics_list in seed_results.items():
        f1s  = [m['F1']     for m in metrics_list]
        aprs = [m['AUC_PR'] for m in metrics_list]
        mccs = [m['MCC']    for m in metrics_list]
        print(f"    {clf_name:<8}: "
              f"F1={np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%  "
              f"MCC={np.mean(mccs)*100:.2f}%  AUC-PR={np.mean(aprs)*100:.2f}%")
        ratio_results.append({
            'Ratio': ratio, 'Classifier': clf_name,
            'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s),
            'AUC_PR_mean': np.mean(aprs), 'MCC_mean': np.mean(mccs)
        })

ratio_df = pd.DataFrame(ratio_results)
print("\n=== BEST AUGMENTATION RATIO PER CLASSIFIER (by mean F1) ===")
for clf in ['LR','RF','XGB','LGBM']:
    sub  = ratio_df[ratio_df['Classifier']==clf]
    best = sub.loc[sub['F1_mean'].idxmax()]
    print(f"  {clf:<8}: best ratio = {best['Ratio']*100:.0f}%  "
          f"mean F1 = {best['F1_mean']*100:.2f}%")

ratio_df.to_csv(os.path.join(RAW_DIR, 'exp3a_augmentation_raw.csv'), index=False)

#Benchmark conditions
benchmark_conditions = {
    '3B_SMOTE':        ('SMOTE',                'SMOTE'),
    '3C_ADASYN':       ('ADASYN',               'ADASYN'),
    '3D_CostSensitive':('Cost-Sensitive',        'CostSensitive'),
    '3E_RUS':          ('Random Undersampling',  'RUS'),
}

for cond_key, (cond_name, short) in benchmark_conditions.items():

    print(f"EXPERIMENT {cond_key} — {cond_name} on Real Training Data")


    seed_results = {clf: [] for clf in ['LR','RF','XGB','LGBM']}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        if short == 'SMOTE':
            print(f"Before SMOTE : {Counter(y_real_train)}")
            X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_real_train, y_real_train)
            print(f"After SMOTE  : {Counter(y_tr)}")
        elif short == 'ADASYN':
            print(f"Before ADASYN : {Counter(y_real_train)}")
            X_tr, y_tr = ADASYN(random_state=seed).fit_resample(X_real_train, y_real_train)
            print(f"After ADASYN  : {Counter(y_tr)}")
        elif short == 'CostSensitive':
            X_tr, y_tr = X_real_train.copy(), y_real_train.copy()
        elif short == 'RUS':
            print(f"Before RUS : {Counter(y_real_train)}")
            X_tr, y_tr = RandomUnderSampler(random_state=seed).fit_resample(
                X_real_train, y_real_train)
            print(f"After RUS  : {Counter(y_tr)}")

        assert not np.any(np.isnan(X_tr)), f"SANITY FAIL: NaN after {cond_name}"

        if short == 'CostSensitive':
            models = build_cs_models(best_params, seed, scale_pos_weight)
        else:
            models = build_models(best_params, seed)

        for clf_name, clf in models.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_pred, y_prob)
            seed_results[clf_name].append(m)
            print_metrics(f"{clf_name} + {short}", m)
            all_results.append({
                'Condition': cond_key, 'Condition_Name': cond_name,
                'Augmentation_Ratio': None, 'Classifier': clf_name,
                'Seed': seed, **m
            })

    print(f"\n=== EXPERIMENT {cond_key} SUMMARY (mean ± std across 5 seeds) ===")

    for clf_name, metrics_list in seed_results.items():
        f1s  = [m['F1']     for m in metrics_list]
        aprs = [m['AUC_PR'] for m in metrics_list]
        mccs = [m['MCC']    for m in metrics_list]
        label = f"{clf_name} + {short}"
        print(f"  {label:<35}: "
              f"F1={np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%  "
              f"MCC={np.mean(mccs)*100:.2f}%  AUC-PR={np.mean(aprs)*100:.2f}%")

    pd.DataFrame([r for r in all_results if r['Condition']==cond_key]).to_csv(
        os.path.join(RAW_DIR, f'{cond_key.lower()}_raw.csv'), index=False)

# Master results

print("MASTER RESULTS TABLE")


results_df = pd.DataFrame(all_results)

aug_results = results_df[results_df['Condition']=='3A_SynthAug'].copy()
best_ratio_rows = []
for clf in ['LR','RF','XGB','LGBM']:
    sub    = aug_results[aug_results['Classifier']==clf]
    best_r = sub.groupby('Augmentation_Ratio')['F1'].mean().idxmax()
    best_ratio_rows.append(sub[sub['Augmentation_Ratio']==best_r].copy())
aug_best = pd.concat(best_ratio_rows)
aug_best['Condition_Name'] = aug_best.apply(
    lambda r: f"Synth Aug (best {r['Augmentation_Ratio']*100:.0f}%)", axis=1)

bench_results = results_df[results_df['Condition']!='3A_SynthAug'].copy()
master_input  = pd.concat([aug_best, bench_results], ignore_index=True)

master = master_input.groupby(['Condition_Name','Classifier']).agg(
    F1_mean=('F1','mean'), F1_std=('F1','std'),
    AUC_PR_mean=('AUC_PR','mean'), AUC_PR_std=('AUC_PR','std'),
    MCC_mean=('MCC','mean'), MCC_std=('MCC','std'),
    Precision_mean=('Precision','mean'),
    Recall_mean=('Recall','mean'),
    FPR_mean=('FPR','mean'),
).reset_index()
master['Model'] = master['Classifier'] + ' (' + master['Condition_Name'] + ')'
master = master.sort_values('F1_mean', ascending=False).reset_index(drop=True)

print("\n=== MASTER RESULTS TABLE — EXPERIMENT 3 ===")
print("  NOTE: F1 is binary (positive class = fraud). Not micro or macro averaged.")
for _, row in master.iterrows():
    print(f"  {row['Model']:<55} "
          f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
          f"MCC={row['MCC_mean']*100:.2f}%")

master.to_csv(os.path.join(RES_DIR, 'master_results_exp3.csv'), index=False)

#Cross-experiment summary

print(" CROSS-EXPERIMENT BEST PERFORMANCE SUMMARY")


exp1 = pd.read_csv(os.path.join(RES_DIR, 'master_results_real_data.csv'))
exp2 = pd.read_csv(os.path.join(RES_DIR, 'master_results_synthetic.csv'))

cross_rows = [
    {'Experiment': 'Exp 1: Real Only (Baseline)',
     'Best_Model': exp1.iloc[0]['Model'],
     'F1_mean': exp1.iloc[0]['F1_mean'], 'F1_std': exp1.iloc[0]['F1_std']},
    {'Experiment': 'Exp 2: Synthetic Only',
     'Best_Model': exp2.iloc[0]['Model'],
     'F1_mean': exp2.iloc[0]['F1_mean'], 'F1_std': exp2.iloc[0]['F1_std']},
]

for _, row in master.groupby('Condition_Name').apply(
        lambda x: x.loc[x['F1_mean'].idxmax()]).iterrows():
    cross_rows.append({
        'Experiment': f'Exp 3: {row["Condition_Name"]}',
        'Best_Model': row['Model'],
        'F1_mean': row['F1_mean'], 'F1_std': row['F1_std']
    })

cross = pd.DataFrame(cross_rows)
print("\n=== CROSS-EXPERIMENT BEST PERFORMANCE SUMMARY ===")
for _, row in cross.iterrows():
    print(f"  {row['Experiment']:<45} {row['Best_Model']:<35} "
          f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%")

cross.to_csv(os.path.join(RES_DIR, 'cross_experiment_comparison.csv'), index=False)


