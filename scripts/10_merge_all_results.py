
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RES_DIR = os.path.join('outputs', 'results')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')



METRICS = ['F1', 'AUC_PR', 'MCC', 'AUC_ROC', 'Precision', 'Recall', 'FPR']
COUNT_COLS = ['TP', 'FP', 'TN', 'FN']

def load(path, label):
    """Load CSV if exists, print status."""
    full = os.path.join(*path) if isinstance(path, (list, tuple)) else path
    if os.path.exists(full):
        df = pd.read_csv(full)
        print(f"  ✓ {label:45s} {len(df):>5} rows")
        return df
    print(f"  ✗ {label:45s} NOT FOUND: {full}")
    return None

def aggregate(df, group_cols):
    """Aggregate per-seed results to mean ± std, sorted by F1_mean desc."""
    if df is None or df.empty:
        return pd.DataFrame()
    agg_dict = {}
    for m in METRICS:
        if m in df.columns:
            agg_dict[m] = ['mean', 'std']
    for c in COUNT_COLS:
        if c in df.columns:
            agg_dict[c] = 'mean'
    if 'Seed' in df.columns:
        agg_dict['Seed'] = 'count'
    # Only group by columns that exist
    group_cols = [c for c in group_cols if c in df.columns]
    result = df.groupby(group_cols).agg(agg_dict).reset_index()
    result.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) and c[1]
                      else (c[0] if isinstance(c, tuple) else c)
                      for c in result.columns]
    if 'Seed_count' in result.columns:
        result = result.rename(columns={'Seed_count': 'Seeds'})
    if 'F1_mean' in result.columns:
        result = result.sort_values('F1_mean', ascending=False).reset_index(drop=True)
    return result

def print_top(df, label, n=10):
    """Print top N results neatly."""
    if df is None or df.empty:
        print(f"  {label}: No data")
        return
    print(f"\n  {label} — Top {min(n, len(df))} results:")
    for _, row in df.head(n).iterrows():
        parts = []
        for col in ['Model', 'Classifier', 'Condition', 'Ratio']:
            if col in row.index and pd.notna(row.get(col)):
                parts.append(str(row[col]))
        label_str = ' | '.join(parts)
        seeds_str = f"({int(row['Seeds'])}/5)" if 'Seeds' in row else ""
        f1_str    = f"F1={row['F1_mean']*100:.2f}%±{row.get('F1_std',0)*100:.2f}%" if 'F1_mean' in row else ""
        apr_str   = f"AUC-PR={row['AUC_PR_mean']*100:.2f}%" if 'AUC_PR_mean' in row else ""
        mcc_str   = f"MCC={row['MCC_mean']*100:.2f}%" if 'MCC_mean' in row else ""
        print(f"    {label_str:<50} {f1_str}  {apr_str}  {mcc_str}  {seeds_str}")

def standardise(df, experiment, model_type):
    """Add standard columns to any raw results dataframe."""
    if df is None:
        return None
    df = df.copy()
    # Add experiment label if missing
    if 'Experiment' not in df.columns:
        df['Experiment'] = experiment
    # Add model type
    df['ModelType'] = model_type
    # Rename Classifier to Model if needed
    if 'Classifier' in df.columns and 'Model' not in df.columns:
        df = df.rename(columns={'Classifier': 'Model'})
    # Add Ratio column if missing
    if 'Ratio' not in df.columns:
        df['Ratio'] = None
    # Add Condition column if missing
    if 'Condition' not in df.columns:
        df['Condition'] = 'Unknown'
    return df



print("LOADING ALL RAW SEED FILES")


print("\n  Experiment 1 — Classical ML:")
e1_1a = load(os.path.join(RAW_DIR, '1a_nohandling_raw.csv'),    'Exp1 No Handling')
e1_1b = load(os.path.join(RAW_DIR, '1b_smote_raw.csv'),         'Exp1 SMOTE')
e1_1c = load(os.path.join(RAW_DIR, '1c_rus_raw.csv'),           'Exp1 Random Undersampling')
e1_1d = load(os.path.join(RAW_DIR, '1d_costsensitive_raw.csv'), 'Exp1 Cost-Sensitive')
e1_1e = load(os.path.join(RAW_DIR, '1e_adasyn_raw.csv'),        'Exp1 ADASYN')

print("\n  Experiment 1 — Deep Learning:")
e1_dl = load(os.path.join(RAW_DIR, 'exp4_deep_learning_raw.csv'), 'Exp1 Deep Learning (Script 08)')

print("\n  Experiment 2 — Classical ML:")
e2_2a = load(os.path.join(RAW_DIR, '2a_nohandling_raw.csv'),    'Exp2 No Handling')
e2_2b = load(os.path.join(RAW_DIR, '2b_smote_raw.csv'),         'Exp2 SMOTE')
e2_2c = load(os.path.join(RAW_DIR, '2c_rus_raw.csv'),           'Exp2 Random Undersampling')
e2_2d = load(os.path.join(RAW_DIR, '2d_costsensitive_raw.csv'), 'Exp2 Cost-Sensitive')
e2_2e = load(os.path.join(RAW_DIR, '2e_adasyn_raw.csv'),        'Exp2 ADASYN')

print("\n  Experiment 2 — Deep Learning:")
e2_dl = load(os.path.join(RAW_DIR, 'dl_exp2_raw.csv'), 'Exp2 Deep Learning (Script 08d)')

print("\n  Experiment 3 — Classical ML:")
e3_aug = load(os.path.join(RAW_DIR, 'exp3a_augmentation_raw.csv'), 'Exp3 Augmentation ratios')
e3_3b  = load(os.path.join(RAW_DIR, '3b_smote_raw.csv'),            'Exp3 SMOTE benchmark')
e3_3c  = load(os.path.join(RAW_DIR, '3c_adasyn_raw.csv'),           'Exp3 ADASYN benchmark')
e3_3d  = load(os.path.join(RAW_DIR, '3d_costsensitive_raw.csv'),    'Exp3 Cost-Sensitive benchmark')
e3_3e  = load(os.path.join(RAW_DIR, '3e_rus_raw.csv'),              'Exp3 RUS benchmark')

print("\n  Experiment 3 — Deep Learning:")
e3_dl  = load(os.path.join(RAW_DIR, 'dl_exp3_raw.csv'), 'Exp3 Deep Learning (Script 08d)')

print("\n  TabPFN Client:")
tabpfn = load(os.path.join(RES_DIR, 'tabpfn_client_partial.csv'), 'TabPFN (Conditions A and B)')


if tabpfn is not None:
    print(f"\n  TabPFN conditions in partial CSV:")
    for exp in tabpfn['Experiment'].unique():
        n = tabpfn[tabpfn['Experiment']==exp]['Seed'].nunique()
        cond = tabpfn[tabpfn['Experiment']==exp]['Condition'].iloc[0] \
               if 'Condition' in tabpfn.columns else '—'
        print(f"    {exp}: {n}/5 seeds  ({cond})")



print("COLUMN INSPECTION")


sample_files = {
    'Exp1 Classical (1a)': e1_1a,
    'Exp1 DL':             e1_dl,
    'Exp2 Classical (2a)': e2_2a,
    'Exp2 DL':             e2_dl,
    'Exp3 Augmentation':   e3_aug,
    'Exp3 DL':             e3_dl,
    'TabPFN':              tabpfn,
}
for name, df in sample_files.items():
    if df is not None:
        print(f"  {name}: {list(df.columns)[:10]}")



print("BUILDING EXPERIMENT 1 — Real Data Baseline")


# Add condition labels to classical ML raw files
def label_condition(df, condition_name, experiment='Exp1'):
    if df is None:
        return None
    df = df.copy()
    df['Condition']  = condition_name
    df['Experiment'] = experiment
    df['ModelType']  = 'Classical'
    if 'Classifier' in df.columns and 'Model' not in df.columns:
        df = df.rename(columns={'Classifier': 'Model'})
    if 'Ratio' not in df.columns:
        df['Ratio'] = None
    return df

e1_parts = []
for df, cond in [
    (e1_1a, 'No Handling'),
    (e1_1b, 'SMOTE'),
    (e1_1c, 'Random Undersampling'),
    (e1_1d, 'Cost-Sensitive'),
    (e1_1e, 'ADASYN'),
]:
    labeled = label_condition(df, cond, 'Exp1')
    if labeled is not None:
        e1_parts.append(labeled)
        print(f"  Added Exp1 Classical {cond}: {len(labeled)} rows")

# Deep learning Exp1
if e1_dl is not None:
    dl1 = e1_dl.copy()
    dl1['ModelType']  = 'DeepLearning'
    dl1['Experiment'] = 'Exp1'
    if 'Ratio' not in dl1.columns:
        dl1['Ratio'] = None
    e1_parts.append(dl1)
    print(f"  Added Exp1 DL: {len(dl1)} rows")
    # Show what models/conditions are in DL
    if 'Model' in dl1.columns and 'Condition' in dl1.columns:
        combos = dl1.groupby(['Model','Condition']).size().reset_index()
        for _, r in combos.iterrows():
            print(f"    DL: {r['Model']} + {r['Condition']}: {r[0]} rows")

# TabPFN Exp1
if tabpfn is not None:
    mask = tabpfn['Experiment'] == 'Exp1_Real'
    tp1  = tabpfn[mask].copy()
    if len(tp1) == 0:
        # Try alternate label
        mask = tabpfn['Experiment'].str.contains('Real', na=False)
        tp1  = tabpfn[mask].copy()
    if len(tp1) > 0:
        tp1['Model']      = 'TabPFN'
        tp1['ModelType']  = 'TabPFN'
        tp1['Experiment'] = 'Exp1'
        if 'Condition' not in tp1.columns:
            tp1['Condition'] = 'No Handling'
        if 'Ratio' not in tp1.columns:
            tp1['Ratio'] = None
        e1_parts.append(tp1)
        print(f"  Added Exp1 TabPFN: {len(tp1)} rows ({tp1['Seed'].nunique()}/5 seeds)")

if e1_parts:
    exp1_all    = pd.concat(e1_parts, ignore_index=True, sort=False)
    print(f"\n  Total Exp1 rows: {len(exp1_all)}")
    group_cols  = [c for c in ['Model','Condition','Ratio'] if c in exp1_all.columns]
    exp1_master = aggregate(exp1_all, group_cols)
    print_top(exp1_master, "Experiment 1 Top Results")
    exp1_master.to_csv(os.path.join(RES_DIR, 'merged_exp1_master.csv'), index=False)
    exp1_all.to_csv(os.path.join(RAW_DIR, 'merged_exp1_raw.csv'), index=False)
    print(f"\n  ✓ Saved merged_exp1_master.csv ({len(exp1_master)} configurations)")
else:
    print("  ERROR: No Exp1 data loaded")
    exp1_master = pd.DataFrame()



print("BUILDING EXPERIMENT 2 — Synthetic-Only Training")


e2_parts = []
for df, cond in [
    (e2_2a, 'No Handling'),
    (e2_2b, 'SMOTE'),
    (e2_2c, 'Random Undersampling'),
    (e2_2d, 'Cost-Sensitive'),
    (e2_2e, 'ADASYN'),
]:
    labeled = label_condition(df, cond, 'Exp2')
    if labeled is not None:
        e2_parts.append(labeled)
        print(f"  Added Exp2 Classical {cond}: {len(labeled)} rows")

if e2_dl is not None:
    dl2 = e2_dl.copy()
    dl2['ModelType']  = 'DeepLearning'
    dl2['Experiment'] = 'Exp2'
    if 'Ratio' not in dl2.columns:
        dl2['Ratio'] = None
    e2_parts.append(dl2)
    print(f"  Added Exp2 DL: {len(dl2)} rows")

# TabPFN Exp2 (synthetic only — Condition C, pending)
if tabpfn is not None:
    mask = tabpfn['Experiment'].isin(['Exp2_Synthetic', 'Exp2'])
    tp2  = tabpfn[mask].copy()
    if len(tp2) > 0:
        tp2['Model']      = 'TabPFN'
        tp2['ModelType']  = 'TabPFN'
        tp2['Experiment'] = 'Exp2'
        if 'Condition' not in tp2.columns:
            tp2['Condition'] = 'No Handling'
        if 'Ratio' not in tp2.columns:
            tp2['Ratio'] = None
        e2_parts.append(tp2)
        print(f"  Added Exp2 TabPFN: {len(tp2)} rows ({tp2['Seed'].nunique()}/5 seeds)")
    else:
        print(f"  Exp2 TabPFN: Not yet available (Condition C still in progress)")

if e2_parts:
    exp2_all    = pd.concat(e2_parts, ignore_index=True, sort=False)
    print(f"\n  Total Exp2 rows: {len(exp2_all)}")
    group_cols  = [c for c in ['Model','Condition'] if c in exp2_all.columns]
    exp2_master = aggregate(exp2_all, group_cols)
    print_top(exp2_master, "Experiment 2 Top Results")
    exp2_master.to_csv(os.path.join(RES_DIR, 'merged_exp2_master.csv'), index=False)
    exp2_all.to_csv(os.path.join(RAW_DIR, 'merged_exp2_raw.csv'), index=False)
    print(f"\n  ✓ Saved merged_exp2_master.csv ({len(exp2_master)} configurations)")
else:
    print("  ERROR: No Exp2 data loaded")
    exp2_master = pd.DataFrame()



print("BUILDING EXPERIMENT 3 — Augmented Training")


e3_parts = []

# Augmentation ratios (classical ML)
if e3_aug is not None:
    aug = e3_aug.copy()
    aug['Experiment'] = 'Exp3'
    aug['ModelType']  = 'Classical'
    if 'Classifier' in aug.columns and 'Model' not in aug.columns:
        aug = aug.rename(columns={'Classifier': 'Model'})
    # Make sure Condition and Ratio columns exist
    # The augmentation file may have a 'Ratio' or 'Augmentation_Ratio' column
    if 'Ratio' not in aug.columns:
        if 'Augmentation_Ratio' in aug.columns:
            aug = aug.rename(columns={'Augmentation_Ratio': 'Ratio'})
        elif 'augmentation_ratio' in aug.columns:
            aug = aug.rename(columns={'augmentation_ratio': 'Ratio'})
    if 'Condition' not in aug.columns:
        if 'Ratio' in aug.columns:
            aug['Condition'] = aug['Ratio'].apply(
                lambda r: f"Synth Aug ({str(r)})" if pd.notna(r) else 'Synth Aug'
            )
        else:
            aug['Condition'] = 'Synth Aug'
    e3_parts.append(aug)
    print(f"  Added Exp3 Augmentation: {len(aug)} rows")
    if 'Ratio' in aug.columns:
        print(f"    Ratios found: {sorted(aug['Ratio'].dropna().unique().tolist())}")
    if 'Condition' in aug.columns:
        print(f"    Conditions: {aug['Condition'].unique().tolist()[:5]}")

# Benchmark conditions (classical ML)
for df, cond in [
    (e3_3b, 'SMOTE'),
    (e3_3c, 'ADASYN'),
    (e3_3d, 'Cost-Sensitive'),
    (e3_3e, 'Random Undersampling'),
]:
    labeled = label_condition(df, cond, 'Exp3')
    if labeled is not None:
        e3_parts.append(labeled)
        print(f"  Added Exp3 Classical {cond}: {len(labeled)} rows")

# Deep learning Exp3
if e3_dl is not None:
    dl3 = e3_dl.copy()
    dl3['ModelType']  = 'DeepLearning'
    dl3['Experiment'] = 'Exp3'
    if 'Ratio' not in dl3.columns:
        dl3['Ratio'] = None
    e3_parts.append(dl3)
    print(f"  Added Exp3 DL: {len(dl3)} rows")

# TabPFN Exp3 (50% augmented — Condition B)
if tabpfn is not None:
    mask = tabpfn['Experiment'].isin(['Exp3_Aug50pct', 'Exp3'])
    tp3  = tabpfn[mask].copy()
    if len(tp3) == 0:
        # Try matching by condition name
        if 'Condition' in tabpfn.columns:
            mask = tabpfn['Condition'].str.contains('Aug|augment|50', case=False, na=False)
            tp3  = tabpfn[mask].copy()
    if len(tp3) > 0:
        tp3['Model']      = 'TabPFN'
        tp3['ModelType']  = 'TabPFN'
        tp3['Experiment'] = 'Exp3'
        if 'Condition' not in tp3.columns:
            tp3['Condition'] = 'Synth Aug (50%)'
        else:
            tp3['Condition'] = tp3['Condition'].replace(
                {'50% Augmented (Real + CTGAN)': 'Synth Aug (50%)'}
            )
        if 'Ratio' not in tp3.columns:
            tp3['Ratio'] = '50%'
        else:
            tp3['Ratio'] = tp3['Ratio'].fillna('50%')
        e3_parts.append(tp3)
        print(f"  Added Exp3 TabPFN 50%: {len(tp3)} rows ({tp3['Seed'].nunique()}/5 seeds)")

if e3_parts:
    exp3_all    = pd.concat(e3_parts, ignore_index=True, sort=False)
    print(f"\n  Total Exp3 rows: {len(exp3_all)}")
    group_cols  = [c for c in ['Model','Condition','Ratio'] if c in exp3_all.columns]
    exp3_master = aggregate(exp3_all, group_cols)
    print_top(exp3_master, "Experiment 3 Top Results", n=15)
    exp3_master.to_csv(os.path.join(RES_DIR, 'merged_exp3_master.csv'), index=False)
    exp3_all.to_csv(os.path.join(RAW_DIR, 'merged_exp3_raw.csv'), index=False)
    print(f"\n  ✓ Saved merged_exp3_master.csv ({len(exp3_master)} configurations)")
else:
    print("  ERROR: No Exp3 data loaded")
    exp3_master = pd.DataFrame()



print("CROSS-EXPERIMENT SUMMARY")


summary_rows = []
for label, master in [
    ("Exp1 — Real Baseline",  exp1_master),
    ("Exp2 — Synthetic Only", exp2_master),
    ("Exp3 — Augmented",      exp3_master),
]:
    if master is None or master.empty:
        continue
    best = master.iloc[0]
    row  = {'Scenario': label}
    for col in ['Model', 'Condition', 'Ratio']:
        if col in best:
            row[col] = best[col]
    for m in METRICS:
        for suffix in ['_mean', '_std']:
            col = f"{m}{suffix}"
            if col in best:
                row[col] = best[col]
    if 'Seeds' in best:
        row['Seeds'] = best['Seeds']
    summary_rows.append(row)

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(RES_DIR, 'merged_cross_experiment_summary.csv'), index=False
    )
    print("\n  Best result per experiment:")
    for _, row in summary_df.iterrows():
        ratio = f" ({row['Ratio']})" if 'Ratio' in row and pd.notna(row.get('Ratio')) else ""
        model = f"{row.get('Model','?')} + {row.get('Condition','?')}{ratio}"
        f1    = f"{row['F1_mean']*100:.2f}% ± {row.get('F1_std',0)*100:.2f}%" \
                if 'F1_mean' in row else "?"
        print(f"\n  [{row['Scenario']}]")
        print(f"    Model  : {model}")
        print(f"    F1     : {f1}")
        if 'AUC_PR_mean' in row:
            print(f"    AUC-PR : {row['AUC_PR_mean']*100:.2f}%")
        if 'MCC_mean' in row:
            print(f"    MCC    : {row['MCC_mean']*100:.2f}%")
        if 'FPR_mean' in row:
            print(f"    FPR    : {row['FPR_mean']*100:.3f}%")


print("DATA AVAILABILITY REPORT")


def report(df, label):
    if df is None or df.empty:
        print(f"  {label:<45} NO DATA")
        return
    seeds = df['Seed'].nunique() if 'Seed' in df.columns else '?'
    rows  = len(df)
    models = df['Model'].nunique() if 'Model' in df.columns else \
             df['Classifier'].nunique() if 'Classifier' in df.columns else '?'
    print(f"  {label:<45} {rows:>5} rows | {seeds} seeds | {models} models")

print("\n  Experiment 1:")
report(e1_1a, "Classical — No Handling")
report(e1_1b, "Classical — SMOTE")
report(e1_1c, "Classical — RUS")
report(e1_1d, "Classical — Cost-Sensitive")
report(e1_1e, "Classical — ADASYN")
report(e1_dl, "Deep Learning")
if tabpfn is not None:
    t = tabpfn[tabpfn['Experiment'].isin(['Exp1_Real','Exp1'])]
    report(t if len(t)>0 else None, "TabPFN (real data)")

print("\n  Experiment 2:")
report(e2_2a, "Classical — No Handling")
report(e2_2b, "Classical — SMOTE")
report(e2_2c, "Classical — RUS")
report(e2_2d, "Classical — Cost-Sensitive")
report(e2_2e, "Classical — ADASYN")
report(e2_dl, "Deep Learning")

print("\n  Experiment 3:")
report(e3_aug, "Classical — Augmentation ratios")
report(e3_3b,  "Classical — SMOTE benchmark")
report(e3_3c,  "Classical — ADASYN benchmark")
report(e3_3d,  "Classical — Cost-Sensitive benchmark")
report(e3_3e,  "Classical — RUS benchmark")
report(e3_dl,  "Deep Learning")
if tabpfn is not None:
    t = tabpfn[tabpfn['Experiment'].isin(['Exp3_Aug50pct','Exp3'])]
    report(t if len(t)>0 else None, "TabPFN (50% augmented)")


print("ALL DONE — MERGE SCRIPT")

print(f"""
Output files:
  {os.path.join(RES_DIR, 'merged_exp1_master.csv')}
  {os.path.join(RES_DIR, 'merged_exp2_master.csv')}
  {os.path.join(RES_DIR, 'merged_exp3_master.csv')}
  {os.path.join(RES_DIR, 'merged_cross_experiment_summary.csv')}

""")
