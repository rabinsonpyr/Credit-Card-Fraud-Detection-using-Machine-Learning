

import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

# ── Configuration ─────────────────────────────────────────────────────────────
RECOMMENDATION_FILE = 'outputs/synthesiser/best_synthesiser.txt'
TRAIN_FILE          = 'outputs/results/train_set_unscaled.csv'
N_FRAUD             = 250_000
OUT_FILE            = 'outputs/synthesiser/best_fraud_250k.csv'

CKPT_MAP = {
    'CTGAN':      'outputs/synthesiser/ctgan_synthesizer.pkl',
    'CopulaGAN':  'outputs/synthesiser/copulagan_synthesizer.pkl',
}





# Read recommendation file
print("\Loading quality check recommendation")

if not os.path.exists(RECOMMENDATION_FILE):
    print(f"ERROR: {RECOMMENDATION_FILE} not found.")
    print("Run 04_synthesiser_quality_check.py first.")
    raise FileNotFoundError(RECOMMENDATION_FILE)

with open(RECOMMENDATION_FILE) as f:
    best_name = f.read().strip()

print(f"  Best synthesiser: {best_name}")


#  Load synthesiser 
print(f"\nLoading {best_name} synthesiser")

from sdv.metadata import SingleTableMetadata
from sdv.single_table import (GaussianCopulaSynthesizer,
                               CTGANSynthesizer,
                               CopulaGANSynthesizer)
from sdv.sampling import Condition

if best_name == 'GaussianCopula':
    print("  Refitting now from train_set_unscaled.csv...")

    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        raise FileNotFoundError(TRAIN_FILE)

    train_df  = pd.read_csv(TRAIN_FILE)
    train_sdv = train_df.copy()
    train_sdv['Class'] = train_sdv['Class'].astype(bool)

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(train_sdv)
    meta.update_column('Class', sdtype='boolean')

    synth = GaussianCopulaSynthesizer(meta)
    t0 = time.time()
    synth.fit(train_sdv)
    print(f"  Refitted in {time.time()-t0:.1f}s")

else:
    ckpt_file = CKPT_MAP[best_name]

    if not os.path.exists(ckpt_file):
        print(f"Checkpoint not found: {ckpt_file}")
        raise FileNotFoundError(ckpt_file)

    CLASS_MAP = {
        'CTGAN':     CTGANSynthesizer,
        'CopulaGAN': CopulaGANSynthesizer,
    }

    t0 = time.time()
    synth = CLASS_MAP[best_name].load(ckpt_file)
    print(f"Loaded from checkpoint: {ckpt_file}")
    print(f"Loaded in {time.time()-t0:.1f}s — no retraining needed")

#Generating fraud-only rows using class-conditional generation
print(f"\nGenerating {N_FRAUD:,} fraud-only rows")

fraud_condition = Condition(
    num_rows=N_FRAUD,
    column_values={'Class': True}
)

t0 = time.time()
fraud_only = synth.sample_from_conditions(conditions=[fraud_condition])
fraud_only['Class'] = fraud_only['Class'].astype(int)
elapsed = time.time() - t0
print(f"  Generated in {elapsed:.1f}s")


print(f"\nSanity checks")

assert len(fraud_only) == N_FRAUD, \
    f"Expected {N_FRAUD} rows, got {len(fraud_only)}"
print(f"Row count correct: {len(fraud_only):,}")

assert (fraud_only['Class'] == 1).all(), \
    f"SANITY FAIL: Not all rows are Class=1. " \
    f"Found {(fraud_only['Class']!=1).sum()} non-fraud rows."
print(f"All {len(fraud_only):,} rows confirmed Class=1 (fraud only)")

assert fraud_only.isnull().sum().sum() == 0, \
    "Missing values in generated data"
print(f"No missing values")

# Leakage check — compare against real training fraud rows
# Uses train_set_unscaled.csv (consistent — no dependency on creditcard.csv)
if not os.path.exists(TRAIN_FILE):
    print(" Skipping leakage check — train_set_unscaled.csv not found")
else:
    train_check = pd.read_csv(TRAIN_FILE)
    real_fraud_rows = set(
        train_check[train_check['Class'] == 1].round(6).apply(
            lambda r: tuple(r), axis=1)
    )
    syn_sample = set(
        fraud_only.round(6).head(500).apply(
            lambda r: tuple(r), axis=1)
    )
    overlap = real_fraud_rows & syn_sample
    assert len(overlap) == 0, \
        f"{len(overlap)} synthetic rows exactly match real training rows (leakage)"
    print("No Data Leakage")


fraud_only.to_csv(OUT_FILE, index=False)
size_mb = os.path.getsize(OUT_FILE) / (1024*1024)
print(f"\n  Saved: {OUT_FILE}  ({size_mb:.1f} MB)")

