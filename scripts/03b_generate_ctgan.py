

import os
import time
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_FILE   = 'outputs/results/train_set_unscaled.csv'
OUT_FILE     = 'outputs/synthesiser/ctgan_synthetic_300k.csv'
CKPT_FILE    = 'outputs/synthesiser/ctgan_synthesizer.pkl'
N_GENERATE   = 300_000
CTGAN_EPOCHS = 300

os.makedirs('outputs/synthesiser', exist_ok=True)



print("Generating CTGAN Data")



print("\nLoading unscaled training data")

if not os.path.exists(TRAIN_FILE):
    print(f"\nERROR: {TRAIN_FILE} not found.")
    print("Run 02_experiment1_real_data.py first.")
    raise FileNotFoundError(TRAIN_FILE)

train_df = pd.read_csv(TRAIN_FILE)
print(f"  Loaded: {len(train_df):,} rows")
print(f"  Fraud : {(train_df['Class']==1).sum()} "
      f"({(train_df['Class']==1).mean()*100:.4f}%)")
print(f"  Legit : {(train_df['Class']==0).sum():,}")


assert train_df.isnull().sum().sum() == 0, "Missing values in training data"



print(f"\nFitting CTGANSynthesizer ({CTGAN_EPOCHS} epochs)")
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition

train_sdv = train_df.copy()
train_sdv['Class'] = train_sdv['Class'].astype(bool)

meta = SingleTableMetadata()
meta.detect_from_dataframe(train_sdv)
meta.update_column('Class', sdtype='boolean')
print(f"  Metadata built. Columns: {len(meta.columns)}")

synth = CTGANSynthesizer(meta, epochs=CTGAN_EPOCHS, verbose=True)
t0 = time.time()
synth.fit(train_sdv)
elapsed = (time.time() - t0) / 60
print(f"  Fitted in {elapsed:.1f} minutes")

synth.save(CKPT_FILE)
print(f"  Checkpoint saved: {CKPT_FILE}")


print(f"\nGenerating {N_GENERATE:,} rows")

real_fraud_rate = (train_df['Class'] == 1).mean()
n_fraud_target  = int(N_GENERATE * real_fraud_rate)
n_legit_target  = N_GENERATE - n_fraud_target


print(f"  Target fraud rate: {real_fraud_rate*100:.4f}%")
print(f"  Fraud rows to generate: {n_fraud_target}")
print(f"  Legit rows to generate: {n_legit_target:,}")

fraud_condition = Condition(num_rows=n_fraud_target, column_values={'Class': True})
legit_condition = Condition(num_rows=n_legit_target, column_values={'Class': False})

t0 = time.time()
synthetic = synth.sample_from_conditions(
    conditions=[fraud_condition, legit_condition]
)
synthetic['Class'] = synthetic['Class'].astype(int)
print(f"  Generated in {time.time()-t0:.1f}s")


print("\nChecks")

assert len(synthetic) == N_GENERATE, \
    f"SANITY FAIL: Expected {N_GENERATE} rows, got {len(synthetic)}"
assert synthetic.isnull().sum().sum() == 0, \
    "SANITY FAIL: Missing values in synthetic data"
assert set(synthetic.columns) == set(train_df.columns), \
    "SANITY FAIL: Column mismatch between synthetic and real data"

fraud_n    = (synthetic['Class']==1).sum()
fraud_rate = (synthetic['Class']==1).mean() * 100
real_rate  = (train_df['Class']==1).mean() * 100

# Verify class balance is close to real data
assert abs(fraud_rate - real_rate) < 0.05, \
    f"SANITY FAIL: Fraud rate {fraud_rate:.3f}% too far from real rate {real_rate:.3f}%"

print(f"Row count   : {len(synthetic):,}")
print(f"Fraud rows  : {fraud_n:,} ({fraud_rate:.3f}%)")
print(f"Legit rows  : {(synthetic['Class']==0).sum():,}")
print(f" Real rate   : {real_rate:.4f}% | Synthetic rate: {fraud_rate:.4f}%")
print("Class balance ok")
print("Columns matched real training data")


synthetic.to_csv(OUT_FILE, index=False)
size_mb = os.path.getsize(OUT_FILE) / (1024*1024)
print(f"\n  Saved: {OUT_FILE}  ({size_mb:.1f} MB)")


