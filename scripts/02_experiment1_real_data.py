
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, json, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, average_precision_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import pickle

RAW_FILE     = 'data/creditcard.csv'
RANDOM_STATE = 42
TEST_SIZE    = 0.20
SEEDS        = [42, 123, 456, 789, 1234]
N_ITER_TUNE  = 20
CV_FOLDS     = 5

OUT_DIR = 'outputs'
RES_DIR = os.path.join(OUT_DIR, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
MOD_DIR = os.path.join(OUT_DIR, 'models')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')

for d in [RES_DIR, FIG_DIR, MOD_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)



# Evaluator
def evaluate(y_true, y_pred, y_prob):
    """
    Binary F1 (fraud class only), AUC-PR, MCC, Precision, Recall, AUC-ROC, FPR.
    F1 is binary (positive class = fraud = 1), not micro or macro averaged.
    """
    tp = ((y_pred==1)&(y_true==1)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'AUC_PR':    average_precision_score(y_true, y_prob),
        'MCC':       matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'AUC_ROC':   roc_auc_score(y_true, y_prob),
        'FPR':       fpr,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

def print_metrics(name, m):
    print(f"\n  {name}")
    print(f"    F1 (binary, fraud class) : {m['F1']*100:.2f}%")
    print(f"    AUC-PR                   : {m['AUC_PR']*100:.2f}%")
    print(f"    MCC                      : {m['MCC']*100:.2f}%")
    print(f"    Precision                : {m['Precision']*100:.2f}%")
    print(f"    Recall                   : {m['Recall']*100:.2f}%")
    print(f"    AUC-ROC                  : {m['AUC_ROC']*100:.2f}%")
    print(f"    FPR                      : {m['FPR']*100:.2f}%  ({m['FP']} false alarms)")
    print(f"    TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

# Section 1 — Load


print("\nSECTION 1 — Loading")


df = pd.read_csv(RAW_FILE)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Legitimate : {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.4f}%)")
print(f"  Fraudulent : {(df['Class']==1).sum():,}  ({(df['Class']==1).mean()*100:.4f}%)")
print(f"  Missing    : {df.isnull().sum().sum()}")

n_before = len(df)
df = df.drop_duplicates()
print(f"\n  Duplicates removed: {n_before - len(df):,}")
print(f"  Clean dataset     : {len(df):,} rows")

assert df.isnull().sum().sum() == 0, "Missing Values"
assert n_before > len(df),           "No duplicates, need to check raw file"
assert df['Class'].nunique() == 2,   "Wrong: Class column must have 2 values"
print("no missing values")
print("duplicates removed")
print("Only binary class exists")

# Section 2 - Feature Engineering
print("\nSection 2 - Feature Engineering")


df['log_Amount'] = np.log1p(df['Amount'])
df['hour_of_day'] = (df['Time'] / 3600) % 24
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL   = 'Class'

print(f"Feature set: {len(FEATURE_COLS)} features")
print("  PCA features: V1–V28")
print("  Engineered: log_Amount, hour_sin, hour_cos")

assert len(FEATURE_COLS) == 31, f"Expected 31 features, got {len(FEATURE_COLS)}"
assert all(f in df.columns for f in FEATURE_COLS), "Missing feature columns"
print("31 features confirmed")

#Section 3 — Stratified 80/20 split
print("\nSection 3 Stratified 80/20 split")


X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train_raw, X_test_raw = X[idx_train], X[idx_test]
y_train,     y_test      = y[idx_train], y[idx_test]

print(f"Train : {len(y_train):,} rows | Fraud: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
print(f"Test  : {len(y_test):,} rows  | Fraud: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

assert abs(y_train.mean() - y_test.mean()) < 0.0005, \
    "Fraud rate not preserved."
assert y_test.sum() > 0, "No fraud cases in test set"
print("  fraud rate is same in both splits")

#SECTION 4
print("\nSection 4: Scaling and saving train/test set")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

assert not np.any(np.isnan(X_train)), "NaN in scaled training data"
assert not np.any(np.isinf(X_train)), "Inf in scaled training data"
assert not np.any(np.isnan(X_test)),  "NaN in scaled test data"


# Saving scaler
scaler_path = os.path.join(MOD_DIR, 'standard_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved: {scaler_path}")


train_unscaled = pd.DataFrame(X_train_raw, columns=FEATURE_COLS)
train_unscaled['Class'] = y_train
train_unscaled_path = os.path.join(RES_DIR, 'train_set_unscaled.csv')
train_unscaled.to_csv(train_unscaled_path, index=False)
print(f"\nTrain set (unscaled) saved : {train_unscaled_path}")
print("for synthesiser scripts 03a, 03b, 03c")



train_scaled = pd.DataFrame(X_train, columns=FEATURE_COLS)
train_scaled['Class'] = y_train
train_scaled_path = os.path.join(RES_DIR, 'train_set.csv')
train_scaled.to_csv(train_scaled_path, index=False)
print(f"\nTrain set (scaled)   saved : {train_scaled_path}")
print("for experiment scripts 06, 07, 08")

# Save test set
test_out = pd.DataFrame(X_test, columns=FEATURE_COLS)
test_out['Class'] = y_test
test_out.to_csv(os.path.join(RES_DIR, 'test_set.csv'), index=False)
print(f"\nTest set saved: {os.path.join(RES_DIR, 'test_set.csv')}")

# Save test set checksum — Exp 2, 3, 4 verify this before running
   # test_checksum = int(pd.util.hash_pandas_object(
    #    pd.DataFrame(X_test, columns=FEATURE_COLS).round(6)).sum())
   # with open(os.path.join(RES_DIR, 'test_set_checksum.txt'), 'w') as f:
    #    f.write(f"{test_checksum}\n{len(y_test)}\n{y_test.sum()}")
    #print(f"Test set checksum saved    : {os.path.join(RES_DIR, 'test_set_checksum.txt')}") This sanity check is not needed.

# Section 5: Hyperparameter tuning
print("\nSection 5 - Hyperparameter Tuning")
print(f"Strategy: RandomizedSearchCV, AUC-PR scoring, {CV_FOLDS}-fold CV, {N_ITER_TUNE} iterations")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

param_grids = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        'max_iter': [1000]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'num_leaves': [20, 31, 50],
        'max_depth': [5, 10, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [20, 50]
    }
}

base_estimators = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    'RandomForest':       RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost':            xgb.XGBClassifier(random_state=RANDOM_STATE,
                                             eval_metric='aucpr',
                                             use_label_encoder=False, verbosity=0),
    'LightGBM':           lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
}

best_params    = {}
tuning_results = []

for name, estimator in base_estimators.items():
    print(f"\nTuning {name}...")
    t0 = time.time()
    search = RandomizedSearchCV(
        estimator, param_grids[name],
        n_iter=N_ITER_TUNE, scoring='average_precision',
        cv=skf, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    best_params[name] = search.best_params_
    best_score = search.best_score_
    tuning_results.append({
        'Classifier': name,
        'Best_CV_AUC_PR': best_score,
        'Tuning_Time_s': elapsed
    })
    print(f"  Best CV AUC-PR : {best_score*100:.2f}%  (in {elapsed:.0f}s)")
    print(f"  Best params    : {search.best_params_}")

hp_path = os.path.join(RES_DIR, 'best_hyperparameters.json')
with open(hp_path, 'w') as f:
    json.dump(best_params, f, indent=2)
print(f"\nHyperparameters saved: {hp_path}")

with open(hp_path) as f:
    hp_verify = json.load(f)
assert set(hp_verify.keys()) == set(base_estimators.keys()), \
    "Hyperparameter file missing some classifiers"
print("all 4 classifier hyperparameters saved")

pd.DataFrame(tuning_results).to_csv(
    os.path.join(RES_DIR, 'tuning_summary.csv'), index=False)

print(f"\nTunning Summary")
for r in tuning_results:
    print(f"  {r['Classifier']:<22} AUC-PR={r['Best_CV_AUC_PR']*100:.2f}%  "
          f"time={r['Tuning_Time_s']:.0f}s")

# Building tuned models
def build_models(params, seed):
    return {
        'LR':   LogisticRegression(**params['LogisticRegression'], random_state=seed),
        'RF':   RandomForestClassifier(**params['RandomForest'], random_state=seed, n_jobs=-1),
        'XGB':  xgb.XGBClassifier(**params['XGBoost'], random_state=seed,
                                   eval_metric='aucpr', use_label_encoder=False, verbosity=0),
        'LGBM': lgb.LGBMClassifier(**params['LightGBM'], random_state=seed, verbosity=-1)
    }

def build_cs_models(params, seed):
    n_fraud = (y_train == 1).sum()
    n_legit = (y_train == 0).sum()
    spw = n_legit / n_fraud
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

#SECTION 7 — Run experiments ───────────────────────────────────────────────
conditions = {
    '1A_NoHandling':     'No Handling',
    '1B_SMOTE':          'SMOTE',
    '1C_RUS':            'Random Undersampling',
    '1D_CostSensitive':  'Cost-Sensitive',
    '1E_ADASYN':         'ADASYN',
}

all_results = []

for cond_key, cond_name in conditions.items():

    print(f"Experiment {cond_key} — {cond_name}")


    seed_results = {clf: [] for clf in ['LR', 'RF', 'XGB', 'LGBM']}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        if cond_key == '1A_NoHandling':
            X_tr, y_tr = X_train.copy(), y_train.copy()

        elif cond_key == '1B_SMOTE':
            print(f"Before SMOTE : {Counter(y_train)}")
            sm = SMOTE(random_state=seed)
            X_tr, y_tr = sm.fit_resample(X_train, y_train)
            print(f"After SMOTE  : {Counter(y_tr)}")
            assert (y_tr==1).sum() == (y_tr==0).sum(), \
                "SMOTE did not balance classes"

        elif cond_key == '1C_RUS':
            print(f"Before RUS : {Counter(y_train)}")
            rus = RandomUnderSampler(random_state=seed)
            X_tr, y_tr = rus.fit_resample(X_train, y_train)
            print(f"After RUS  : {Counter(y_tr)}")
            assert (y_tr==1).sum() == (y_tr==0).sum(), \
                "RUS did not balance classes"

        elif cond_key == '1D_CostSensitive':
            X_tr, y_tr = X_train.copy(), y_train.copy()

        elif cond_key == '1E_ADASYN':
            print(f"Before ADASYN : {Counter(y_train)}")
            ada = ADASYN(random_state=seed)
            X_tr, y_tr = ada.fit_resample(X_train, y_train)
            print(f"After ADASYN  : {Counter(y_tr)}")

        assert not np.any(np.isnan(X_tr)), \
            f"NaN in training data after {cond_name}"

        if cond_key == '1D_CostSensitive':
            models = build_cs_models(best_params, seed)
        else:
            models = build_models(best_params, seed)

        for clf_name, clf in models.items():
            label = f"{clf_name} + {cond_name}" \
                    if cond_key != '1A_NoHandling' else clf_name
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            m = evaluate(y_test, y_pred, y_prob)
            seed_results[clf_name].append(m)
            print_metrics(f"{label} (seed={seed})", m)

            all_results.append({
                'Condition': cond_key, 'Condition_Name': cond_name,
                'Classifier': clf_name, 'Seed': seed, **m,
                'Training_Time_s': elapsed
            })

    print("\n Experiment {cond_key} Summary (mean ± std across 5 seeds)")
    for clf_name, metrics_list in seed_results.items():
        f1s  = [m['F1']     for m in metrics_list]
        aprs = [m['AUC_PR'] for m in metrics_list]
        mccs = [m['MCC']    for m in metrics_list]
        label = f"{clf_name} + {cond_name}" \
                if cond_key != '1A_NoHandling' else clf_name
        print(f"  {label:<35}: "
              f"F1={np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%  "
              f"MCC={np.mean(mccs)*100:.2f}%  AUC-PR={np.mean(aprs)*100:.2f}%")

    pd.DataFrame([r for r in all_results if r['Condition']==cond_key]).to_csv(
        os.path.join(RAW_DIR, f'{cond_key.lower()}_raw.csv'), index=False)

# Master results table
print("Section 8 - Master Results Table")


results_df = pd.DataFrame(all_results)

master = results_df.groupby(['Condition_Name', 'Classifier']).agg(
    F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
    AUC_PR_mean=('AUC_PR', 'mean'), AUC_PR_std=('AUC_PR', 'std'),
    MCC_mean=('MCC', 'mean'), MCC_std=('MCC', 'std'),
    Precision_mean=('Precision', 'mean'),
    Recall_mean=('Recall', 'mean'),
    AUC_ROC_mean=('AUC_ROC', 'mean'),
    FPR_mean=('FPR', 'mean'),
).reset_index()

master['Model'] = master['Classifier'] + ' (' + master['Condition_Name'] + ')'
master = master.sort_values('F1_mean', ascending=False).reset_index(drop=True)

print("\nMaster Results - All Real Data Experiments")
print("  NOTE: F1 is binary (positive class = fraud). Not micro or macro averaged.")
for _, row in master.iterrows():
    print(f"  {row['Model']:<40} "
          f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
          f"AUC-PR={row['AUC_PR_mean']*100:.2f}%  "
          f"MCC={row['MCC_mean']*100:.2f}%")

master_path = os.path.join(RES_DIR, 'master_results_real_data.csv')
master.to_csv(master_path, index=False)
print(f"\nSaved: {master_path}")

best     = master.iloc[0]
best_apr = master.loc[master['AUC_PR_mean'].idxmax()]
best_mcc = master.loc[master['MCC_mean'].idxmax()]


print("Final Summary: Experimnet 1")

print(f"Best F1 (binary, fraud class) : "
      f"{best['Model']:<35} {best['F1_mean']*100:.2f}% ± {best['F1_std']*100:.2f}%")
print(f"Best AUC-PR                   : "
      f"{best_apr['Model']:<35} {best_apr['AUC_PR_mean']*100:.2f}%")
print(f"Best MCC                      : "
      f"{best_mcc['Model']:<35} {best_mcc['MCC_mean']*100:.2f}%")

