import os
import time
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_FILE = 'outputs/results/train_set_unscaled.csv'
OUT_FILE   = 'outputs/synthesiser/gc_synthetic_300k.csv'
N_GENERATE = 300_000

os.makedirs('outputs/synthesiser', exist_ok=True)



print("Synthetic Data Generation -- GaussianCopula")


print("\n Loading unscaled training split")

if not os.path.exists(TRAIN_FILE):
    print(f"\nERROR: {TRAIN_FILE} not found.")
    raise FileNotFoundError(TRAIN_FILE)

train_df = pd.read_csv(TRAIN_FILE)
print(f"  Loaded: {len(train_df):,} rows")
print(f"  Fraud : {(train_df['Class']==1).sum()} "
      f"({(train_df['Class']==1).mean()*100:.4f}%)")
print(f"  Legit : {(train_df['Class']==0).sum():,}")


assert train_df.isnull().sum().sum() == 0, "Missing values in training data"
print("No missing values")


print("\nFitting GaussianCopulaSynthesizer")
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

train_sdv = train_df.copy()
train_sdv['Class'] = train_sdv['Class'].astype(bool)

meta = SingleTableMetadata()
meta.detect_from_dataframe(train_sdv)
meta.update_column('Class', sdtype='boolean')
print(f"  Metadata built done. Columns: {len(meta.columns)}")

synth = GaussianCopulaSynthesizer(meta)
t0 = time.time()
synth.fit(train_sdv)
print(f"  Fitted in {time.time()-t0:.1f}s")


print(f"\nGenerating {N_GENERATE:,} rows")


t0 = time.time()
synthetic = synth.sample(num_rows=N_GENERATE)
synthetic['Class'] = synthetic['Class'].astype(int)
print(f"  Generated in {time.time()-t0:.1f}s")


print("\nchecks")

assert len(synthetic) == N_GENERATE, \
    f"Expected {N_GENERATE} rows, got {len(synthetic)}"
assert synthetic.isnull().sum().sum() == 0, \
    "Missing values in synthetic data"
assert set(synthetic.columns) == set(train_df.columns), \
    "Column mismatch between synthetic and real data"

fraud_n    = (synthetic['Class']==1).sum()
fraud_rate = (synthetic['Class']==1).mean() * 100
real_rate  = (train_df['Class']==1).mean() * 100

print(f"Row count: {len(synthetic):,}")
print(f"Fraud rows: {fraud_n:,} ({fraud_rate:.3f}%)")
print(f"Legit rows: {(synthetic['Class']==0).sum():,}")
print(f"Real rate: {real_rate:.3f}% | Synthetic rate: {fraud_rate:.3f}%")
print("Columns matched real training data")


# Save
synthetic.to_csv(OUT_FILE, index=False)
size_mb = os.path.getsize(OUT_FILE) / (1024*1024)
print(f"\n  Saved: {OUT_FILE}  ({size_mb:.1f} MB)")



print(f"""
Output : {OUT_FILE}
Rows   : {len(synthetic):,}
Fraud  : {fraud_n:,} ({fraud_rate:.3f}%)

""")
