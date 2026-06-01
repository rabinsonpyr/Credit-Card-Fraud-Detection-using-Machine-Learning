

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, warnings, json
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.metrics import (f1_score, average_precision_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

# ── TabPFN Client Import ───────────────────────────────────────────────────
try:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier, init
except ImportError:
    print("ERROR: tabpfn-client not installed.")
    print("Run: pip install tabpfn-client")
    sys.exit(1)


TABPFN_TOKEN = os.environ.get("TABPFN_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiOWUxNDk5N2YtNmZhZS00ZGQyLTllMzktYjc2ZGJiOGFjNWQ1IiwiZXhwIjoxODA5Mjc0NDUxfQ.kpVRf0IrCTmEgzQV69G9xX6KzF7dHUzTZLQgf9-8ODo")  # set env var or paste token here

MAX_TRAIN_ROWS  = 50_000     # Hard limit imposed by tabpfn-client library
TEST_BATCH_SIZE = 45_000     # Test set batched below 50k limit
SEEDS = [42, 123, 456, 789, 1234]


# File paths
TEST_SET_FILE = os.path.join('outputs', 'results',     'test_set.csv')
REAL_TRAIN_FILE = os.path.join('outputs', 'results',     'train_set.csv')
SYNTHETIC_300K_FILE = os.path.join('outputs', 'synthesiser', 'ctgan_synthetic_300k.csv')
FRAUD_POOL_FILE  = os.path.join('outputs', 'synthesiser', 'best_fraud_250k.csv')
OUT_DIR   = os.path.join('outputs', 'results')
RES_FILE  = os.path.join(OUT_DIR,   'tabpfn_client_results.csv')
PARTIAL_FILE  = os.path.join(OUT_DIR,   'tabpfn_client_partial.csv')

# test set configuration
EXPECTED_TEST_ROWS  = 56746
EXPECTED_TEST_FRAUD = 95

PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURES + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL= 'Class'





if TABPFN_TOKEN:
    print(f"  Token found — authenticating...")
    try:
        tabpfn_client.init(use_server=True)
        # Set token via environment if not already done
        os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
        print("Authentication done")
    except Exception as e:
        print(f"  Token auth failed: {e}")
        print("  Falling back to interactive login...")
        tabpfn_client.init()
else:
    print("  No token found...")
    tabpfn_client.init()
    token = tabpfn_client.get_access_token()
    


def evaluate(y_true, y_pred, y_prob):
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
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
    print(f"    AUC-ROC                  : {m['AUC_ROC']*100:.2f}%")
    print(f"    MCC                      : {m['MCC']*100:.2f}%")
    print(f"    Precision                : {m['Precision']*100:.2f}%")
    print(f"    Recall                   : {m['Recall']*100:.2f}%")
    print(f"    FPR                      : {m['FPR']*100:.2f}%  ({m['FP']} false alarms)")
    print(f"    TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

def append_result(result_dict):
    """Append a single seed result to the partial CSV immediately after completion.
    This ensures no results are lost if the script crashes or hits the daily limit."""
    os.makedirs(OUT_DIR, exist_ok=True)
    row_df = pd.DataFrame([result_dict])
    if os.path.exists(PARTIAL_FILE):
        row_df.to_csv(PARTIAL_FILE, mode='a', header=False, index=False)
    else:
        row_df.to_csv(PARTIAL_FILE, mode='w', header=True, index=False)
    print(f"Seed result saved to {PARTIAL_FILE}")

def stratified_subsample(X, y, n_rows, seed):
    """
    Stratified subsample of n_rows from (X, y).
    Preserves fraud rate. If n_rows >= len(y), returns full dataset.
    """
    if len(y) <= n_rows:
        print(f"  Dataset has {len(y):,} rows — using full dataset (below {n_rows:,} limit)")
        return X.copy(), y.copy()

    rng        = np.random.default_rng(seed)
    fraud_idx  = np.where(y == 1)[0]
    legit_idx  = np.where(y == 0)[0]
    fraud_rate = len(fraud_idx) / len(y)

    n_fraud    = max(1, int(n_rows * fraud_rate))
    n_legit    = n_rows - n_fraud

    fraud_samp = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
    legit_samp = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)

    idx = np.concatenate([fraud_samp, legit_samp])
    rng.shuffle(idx)

    print(f"  Subsampled {len(idx):,} rows "
          f"({(y[idx]==1).sum()} fraud, {(y[idx]==0).sum():,} legit) "
          f"from {len(y):,} total")
    return X[idx], y[idx]

def run_tabpfn_seeds(X_train_full, y_train_full, X_test, y_test,
                     condition_name, experiment_label, seeds):
    """
    Runs TabPFN across all seeds for one experimental condition.
    - Appends each seed result to PARTIAL_FILE immediately after completion
    - Skips seeds already present in PARTIAL_FILE (safe resume)
    - Batches test set into TEST_BATCH_SIZE chunks to stay under 50k limit
    Returns list of result dicts for seeds run in this session.
    """
    results = []

    # Check which seeds are already completed in the partial file
    completed_seeds = set()
    if os.path.exists(PARTIAL_FILE):
        try:
            partial_df    = pd.read_csv(PARTIAL_FILE)
            already_done  = partial_df[
                (partial_df['Experiment'] == experiment_label) &
                (partial_df['Condition']  == condition_name)
            ]['Seed'].tolist()
            completed_seeds = set(already_done)
            if completed_seeds:
                print(f"  Resuming — already completed seeds: {sorted(completed_seeds)}")
        except Exception:
            pass

    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n--- Seed {seed} --- SKIPPED (already saved in partial CSV)")
            continue

        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        # Stratified subsample to MAX_TRAIN_ROWS
        X_tr, y_tr = stratified_subsample(
            X_train_full, y_train_full, MAX_TRAIN_ROWS, seed
        )
        print(f"  Fraud in training subsample : {(y_tr==1).sum()} ({(y_tr==1).mean()*100:.4f}%)")

        # Build TabPFN classifier
        try:
            clf = TabPFNClassifier(balance_probabilities=True)
        except TypeError:
            print("   balance_probabilities not supported — using default")
            clf = TabPFNClassifier()

        # Fit — sends training data to PriorLabs server
        print(f"  Sending {len(y_tr):,} training rows to TabPFN server...")
        clf.fit(X_tr, y_tr)
        print(f" Fit complete")

        # Predict on test set in batches (50k row limit applies to test too)
        print(f"  Running inference on {len(y_test):,} test rows (batched)...")
        all_probs = []
        for batch_start in range(0, len(X_test), TEST_BATCH_SIZE):
            batch_end  = min(batch_start + TEST_BATCH_SIZE, len(X_test))
            X_batch    = X_test[batch_start:batch_end]
            print(f"    Batch {batch_start:,}–{batch_end:,} ({len(X_batch):,} rows)...")
            prob_batch = clf.predict_proba(X_batch)
            all_probs.append(prob_batch)

        y_prob_2d = np.vstack(all_probs)
        y_prob    = y_prob_2d[:, 1]
        y_pred    = (y_prob >= 0.5).astype(int)
        print(f"  Inference complete ({len(y_prob):,} predictions)")

        m = evaluate(y_test, y_pred, y_prob)
        print_metrics(f"TabPFN | {condition_name} (seed={seed})", m)

        result = {
            'Experiment':      experiment_label,
            'Condition':       condition_name,
            'Seed':            seed,
            'Train_rows_used': len(y_tr),
            'Train_fraud':     int((y_tr==1).sum()),
            **m
        }

        # Save immediately
        append_result(result)
        results.append(result)

    return results

# Load data 

print("LOADING DATA")


# Check required files
required = [TEST_SET_FILE, REAL_TRAIN_FILE]
for f in required:
    if not os.path.exists(f):
        print(f"ERROR: Required file not found: {f}")
        print("Run 02_experiment1_real_data.py first.")
        sys.exit(1)

# Load test set
test_df = pd.read_csv(TEST_SET_FILE)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

print(f"Test set: {len(y_test):,} rows  ({y_test.sum()} fraud, {(y_test==0).sum():,} legit)")
assert len(y_test) == EXPECTED_TEST_ROWS,  f"SANITY FAIL: Expected {EXPECTED_TEST_ROWS} test rows"
assert y_test.sum() == EXPECTED_TEST_FRAUD, f"SANITY FAIL: Expected {EXPECTED_TEST_FRAUD} fraud in test"
print(f"test set matches Experiment 1 ({EXPECTED_TEST_ROWS} rows, {EXPECTED_TEST_FRAUD} fraud)")

# Load real training set
train_df  = pd.read_csv(REAL_TRAIN_FILE)
X_train   = train_df[FEATURE_COLS].values
y_train   = train_df[TARGET_COL].values
print(f"Real train: {len(y_train):,} rows  ({y_train.sum()} fraud, {(y_train==0).sum():,} legit)")

# Load synthetic data if available
synthetic_available = os.path.exists(SYNTHETIC_300K_FILE)
fraud_pool_available = os.path.exists(FRAUD_POOL_FILE)

if synthetic_available:
    syn_df    = pd.read_csv(SYNTHETIC_300K_FILE)
    X_syn     = syn_df[FEATURE_COLS].values
    y_syn     = syn_df[TARGET_COL].values
    print(f"CTGAN synthetic: {len(y_syn):,} rows  ({y_syn.sum()} fraud, {(y_syn==0).sum():,} legit)")
else:
    print(f" Synthetic data not found at {SYNTHETIC_300K_FILE}")
    print(f"Condition C (synthetic-only) will be skipped.")

if fraud_pool_available:
    pool_df   = pd.read_csv(FRAUD_POOL_FILE)
    X_pool    = pool_df[FEATURE_COLS].values
    y_pool    = pool_df[TARGET_COL].values
    print(f"Fraud pool: {len(y_pool):,} synthetic fraud cases")
else:
    print(f"Fraud pool not found at {FRAUD_POOL_FILE}")
    print(f"  Condition B (50% augmented) will be skipped.")

# ── Condition A — Real Data Only (Experiment 1 setup) ─────────────────────

print("CONDITION A — REAL DATA ONLY (Experiment 1 setup)")
print("Train: Stratified 50k subsample of real training data")
print("Test : Fixed real held-out test set (56,746 rows)")


all_results = []

results_a = run_tabpfn_seeds(
    X_train, y_train, X_test, y_test,
    condition_name   = "No Handling (Real)",
    experiment_label = "Exp1_Real",
    seeds            = SEEDS
)
all_results.extend(results_a)

# Summary A
f1s_a  = [r['F1']     for r in results_a]
aprs_a = [r['AUC_PR'] for r in results_a]
mccs_a = [r['MCC']    for r in results_a]
print(f"\n=== CONDITION A SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")

print(f"  F1 (binary, fraud class) : {np.mean(f1s_a)*100:.2f}% ± {np.std(f1s_a)*100:.2f}%")
print(f"  AUC-PR                   : {np.mean(aprs_a)*100:.2f}% ± {np.std(aprs_a)*100:.2f}%")
print(f"  MCC                      : {np.mean(mccs_a)*100:.2f}% ± {np.std(mccs_a)*100:.2f}%")
print(f"  Train rows used          : {MAX_TRAIN_ROWS:,} (stratified subsample)")

# Condition B — 50% Augmented (Experiment 3 setup)
if fraud_pool_available:

    print("CONDITION B — 50% AUGMENTED (Experiment 3 setup)")
    print("Train: Real data + CTGAN synthetic fraud → 50% fraud rate")
    print("       Capped at 50,000 rows (stratified)")
    print("Test : Fixed real held-out test set (56,746 rows)")


    # Build 50% augmented dataset
    n_legit_train = int((y_train == 0).sum())  # 226,602 legitimate
    n_fraud_50pct = n_legit_train              # equal fraud and legit = 50%
    n_synthetic_needed = n_fraud_50pct - int((y_train == 1).sum())

    # Cap synthetic fraud at pool size
    n_synthetic_needed = min(n_synthetic_needed, len(y_pool))

    # Sample from fraud pool (stratified by seed inside run_tabpfn_seeds)
    rng_aug    = np.random.default_rng(42)  # fixed seed for pool sampling
    pool_idx   = rng_aug.choice(len(y_pool), size=n_synthetic_needed, replace=False)
    X_aug_full = np.vstack([X_train, X_pool[pool_idx]])
    y_aug_full = np.concatenate([y_train, y_pool[pool_idx]])

    print(f"  Combined dataset before subsampling:")
    print(f"    Total rows : {len(y_aug_full):,}")
    print(f"    Fraud      : {(y_aug_full==1).sum():,} ({(y_aug_full==1).mean()*100:.2f}%)")
    print(f"    Legitimate : {(y_aug_full==0).sum():,}")

    results_b = run_tabpfn_seeds(
        X_aug_full, y_aug_full, X_test, y_test,
        condition_name   = "50% Augmented (Real + CTGAN)",
        experiment_label = "Exp3_Aug50pct",
        seeds            = SEEDS
    )
    all_results.extend(results_b)

    # Summary B
    f1s_b  = [r['F1']     for r in results_b]
    aprs_b = [r['AUC_PR'] for r in results_b]
    mccs_b = [r['MCC']    for r in results_b]
    print(f"\n=== CONDITION B SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")
    print(f"  F1 (binary, fraud class) : {np.mean(f1s_b)*100:.2f}% ± {np.std(f1s_b)*100:.2f}%")
    print(f"  AUC-PR                   : {np.mean(aprs_b)*100:.2f}% ± {np.std(aprs_b)*100:.2f}%")
    print(f"  MCC                      : {np.mean(mccs_b)*100:.2f}% ± {np.std(mccs_b)*100:.2f}%")
    print(f"  Train rows used          : {MAX_TRAIN_ROWS:,} (stratified subsample from augmented set)")

else:
    print(f"\nSkipping Condition B — fraud pool file not found.")

# Condition C — Synthetic Only (Experiment 2 setup)
if synthetic_available:

    print("CONDITION C — SYNTHETIC ONLY (Experiment 2 setup)")
    print("Train: Stratified 50k subsample of CTGAN synthetic data")
    print("Test : Fixed real held-out test set (56,746 rows)")


    results_c = run_tabpfn_seeds(
        X_syn, y_syn, X_test, y_test,
        condition_name   = "No Handling (Synthetic)",
        experiment_label = "Exp2_Synthetic",
        seeds            = SEEDS
    )
    all_results.extend(results_c)

    # Summary C
    f1s_c  = [r['F1']     for r in results_c]
    aprs_c = [r['AUC_PR'] for r in results_c]
    mccs_c = [r['MCC']    for r in results_c]
    print(f"\n=== CONDITION C SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")
    print(f"  F1 (binary, fraud class) : {np.mean(f1s_c)*100:.2f}% ± {np.std(f1s_c)*100:.2f}%")
    print(f"  AUC-PR                   : {np.mean(aprs_c)*100:.2f}% ± {np.std(aprs_c)*100:.2f}%")
    print(f"  MCC                      : {np.mean(mccs_c)*100:.2f}% ± {np.std(mccs_c)*100:.2f}%")
    print(f"  Train rows used          : {MAX_TRAIN_ROWS:,} (stratified subsample from synthetic data)")

else:
    print(f"\nSkipping Condition C — synthetic data file not found.")

# Master Results Table

print("MASTER RESULTS TABLE — TABPFN CLIENT")


# Load ALL results including previous sessions from partial CSV
os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(PARTIAL_FILE):
    all_results_df = pd.read_csv(PARTIAL_FILE)
    print(f"  Loaded {len(all_results_df)} total seed results from {PARTIAL_FILE}")
else:
    all_results_df = pd.DataFrame(all_results)

# Also include any results from this session not yet in partial file
if all_results:
    session_df     = pd.DataFrame(all_results)
    all_results_df = pd.concat([all_results_df, session_df]).drop_duplicates(
        subset=['Experiment', 'Condition', 'Seed']
    ).reset_index(drop=True)

master = all_results_df.groupby(['Experiment', 'Condition']).agg(
    F1_mean=('F1','mean'),           F1_std=('F1','std'),
    AUC_PR_mean=('AUC_PR','mean'),   AUC_PR_std=('AUC_PR','std'),
    MCC_mean=('MCC','mean'),         MCC_std=('MCC','std'),
    AUC_ROC_mean=('AUC_ROC','mean'), AUC_ROC_std=('AUC_ROC','std'),
    Precision_mean=('Precision','mean'),
    Recall_mean=('Recall','mean'),
    FPR_mean=('FPR','mean'),
    Train_rows_mean=('Train_rows_used','mean'),
    Train_fraud_mean=('Train_fraud','mean'),
    Seeds_completed=('Seed','count'),
).reset_index()
master = master.sort_values('F1_mean', ascending=False).reset_index(drop=True)



print()

for _, row in master.iterrows():
    seeds_note = f"({int(row['Seeds_completed'])}/5 seeds)" if row['Seeds_completed'] < 5 else "(5/5 seeds)"
    print(f"  [{row['Experiment']}] {row['Condition']} {seeds_note}")
    print(f"    F1      : {row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%")
    print(f"    AUC-PR  : {row['AUC_PR_mean']*100:.2f}%")
    print(f"    AUC-ROC : {row['AUC_ROC_mean']*100:.2f}%")
    print(f"    MCC     : {row['MCC_mean']*100:.2f}%")
    print(f"    FPR     : {row['FPR_mean']*100:.3f}%")
    print(f"    Avg train rows : {row['Train_rows_mean']:.0f} "
          f"({row['Train_fraud_mean']:.0f} fraud)")
    print()


exp1_baseline_f1 = 84.11
print(f"  Reference — Experiment 1 best (RF No Handling): F1 = {exp1_baseline_f1:.2f}%")
for _, row in master.iterrows():
    gap       = exp1_baseline_f1 - row['F1_mean']*100
    direction = "below" if gap > 0 else "ABOVE"
    print(f"  TabPFN [{row['Experiment']}]: {abs(gap):.2f} pp {direction} RF baseline")

# Save final results
all_results_df.to_csv(RES_FILE, index=False)
master_path = os.path.join(OUT_DIR, 'tabpfn_client_master.csv')
master.to_csv(master_path, index=False)

print(f"\n  Raw results saved  : {RES_FILE}")
print(f"  Master table saved : {master_path}")
print(f"  Partial CSV        : {PARTIAL_FILE} (append-safe resume file)")

