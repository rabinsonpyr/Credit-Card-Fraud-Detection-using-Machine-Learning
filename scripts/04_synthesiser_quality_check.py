

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")


TRAIN_FILE = 'outputs/results/train_set_unscaled.csv'

SYNTH_FILES = {
    'GaussianCopula': 'outputs/synthesiser/gc_synthetic_300k.csv',
    'CTGAN':          'outputs/synthesiser/ctgan_synthetic_300k.csv',
    'CopulaGAN':      'outputs/synthesiser/copulagan_synthetic_300k.csv',
}

FIG_DIR = 'outputs/synthesiser/figures'
os.makedirs(FIG_DIR, exist_ok=True)

VALIDATION_FEATS = ['V14', 'V17', 'V12', 'V10', 'V3', 'V4', 'V11']



print("SYNTHESISER QUALITY CHECK")


print("\nChecking all required files")
all_files = {**{'Training data (unscaled)': TRAIN_FILE}, **SYNTH_FILES}
missing = []
for label, path in all_files.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {label}: {path}")
    if not exists:
        missing.append(label)

if missing:
    print(f"\nERROR: Missing files: {missing}")
    print("Run 02_experiment1_real_data.py for training data.")
    print("Run 03a/03b/03c for synthetic CSVs.")
    raise FileNotFoundError(f"Missing: {missing}")


print("\nLoading reference data (unscaled training split)")
train_df = pd.read_csv(TRAIN_FILE)
print(f"  Loaded: {len(train_df):,} rows")
print(f"  Fraud : {(train_df['Class']==1).sum()} "
      f"({(train_df['Class']==1).mean()*100:.4f}%)")




from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

train_sdv = train_df.copy()
train_sdv['Class'] = train_sdv['Class'].astype(bool)
meta = SingleTableMetadata()
meta.detect_from_dataframe(train_sdv)
meta.update_column('Class', sdtype='boolean')

sdv_results = {}

for name, path in SYNTH_FILES.items():
    print(f"  Evaluating {name}...")
    syn = pd.read_csv(path)
    syn['Class'] = syn['Class'].astype(bool)

    t0 = time.time()
    report = evaluate_quality(train_sdv, syn, meta, verbose=True)
    elapsed = time.time() - t0

    overall = report.get_score()
    shapes  = report.get_details('Column Shapes')['Score'].mean()
    trends  = report.get_details('Column Pair Trends')['Score'].mean()

    sdv_results[name] = {
        'Overall':            overall,
        'Column_Shapes':      shapes,
        'Column_Pair_Trends': trends,
        'Eval_Time_s':        elapsed
    }

    print(f"    Overall Score      : {overall*100:.2f}%")
    print(f"    Column Shapes      : {shapes*100:.2f}%")
    print(f"    Column Pair Trends : {trends*100:.2f}%")
    print(f"    Evaluation time    : {elapsed:.1f}s\n")


print(f"Feature-Level Quality Comparison")


real_fraud = train_df[train_df['Class'] == 1]
real_legit = train_df[train_df['Class'] == 0]

synth_fraud_dfs = {}
for name, path in SYNTH_FILES.items():
    syn = pd.read_csv(path)
    synth_fraud_dfs[name] = syn[syn['Class'] == 1]

# Print comparison table
header = f"{'Feature':<12} {'Real Fraud':>12} {'Real Legit':>12}"
for name in SYNTH_FILES:
    header += f" {name[:14]:>16}"
print(header)
print("─" * (12 + 12 + 12 + 16 * len(SYNTH_FILES)))

feature_records = []
for feat in VALIDATION_FEATS:
    rf = real_fraud[feat].mean()
    rl = real_legit[feat].mean()
    row = f"{feat:<12} {rf:>12.4f} {rl:>12.4f}"
    feat_rec = {'Feature': feat, 'Real_Fraud_Mean': rf, 'Real_Legit_Mean': rl}
    flags = []
    for name in SYNTH_FILES:
        sv = synth_fraud_dfs[name][feat].mean()
        feat_rec[f'{name}_Mean'] = sv
        closer_legit = abs(sv - rl) < abs(sv - rf)
        feat_rec[f'{name}_Legit_Like'] = closer_legit
        row += f" {sv:>16.4f}"
        flags.append("L" if closer_legit else "✓")
    quality_str = "  " + " | ".join(f"{'🚨' if f=='L' else '✓'}" for f in flags)
    print(row + quality_str)
    feature_records.append(feat_rec)

feat_df = pd.DataFrame(feature_records)

print(f"\nSUMMARY — legitimate-like counts (lower = better fraud reproduction):")
for name in SYNTH_FILES:
    ll  = feat_df[f'{name}_Legit_Like'].sum()
    sdv = sdv_results[name]['Overall']
    print(f"  {name:<18}: {ll}/7 legitimate-like  |  SDV Overall: {sdv*100:.2f}%")


print(f"\nRanking synthesisers by SDV Overall Score")
print("  Higher SDV score = better overall quality\n")

ranked = sorted(sdv_results.items(), key=lambda x: x[1]['Overall'], reverse=True)

print(f"  {'Rank':<6} {'Synthesiser':<18} {'SDV Overall':>12} "
      f"{'Col Shapes':>12} {'Col Trends':>12} {'Legit-like':>12}")
print("  " + "─" * 74)

for rank, (name, scores) in enumerate(ranked, 1):
    ll = feat_df[f'{name}_Legit_Like'].sum()
    print(f"  {rank:<6} {name:<18} "
          f"{scores['Overall']*100:>11.2f}% "
          f"{scores['Column_Shapes']*100:>11.2f}% "
          f"{scores['Column_Pair_Trends']*100:>11.2f}% "
          f"{ll:>10}/7")

best_name = ranked[0][0]
best_sdv  = ranked[0][1]['Overall']
best_ll   = feat_df[f'{best_name}_Legit_Like'].sum()

print(f"\n  Recomended: {best_name}")
print(f"    SDV Overall Score : {best_sdv*100:.2f}%")
print(f"    Legitimate-like        : {best_ll}/7")


with open('outputs/synthesiser/best_synthesiser.txt', 'w') as f:
    f.write(best_name)
print("\n  Recommendation saved: outputs/synthesiser/best_synthesiser.txt")



print(f"\nSaving output files")

summary_rows = []
for name, scores in sdv_results.items():
    ll = feat_df[f'{name}_Legit_Like'].sum()
    summary_rows.append({
        'Synthesiser':           name,
        'SDV_Overall_%':         round(scores['Overall'] * 100, 2),
        'Column_Shapes_%':       round(scores['Column_Shapes'] * 100, 2),
        'Column_Pair_Trends_%':  round(scores['Column_Pair_Trends'] * 100, 2),
        'Legit_Like_Count':      int(ll),
        'Legit_Like_Out_Of':     7,
        'Recommended':           name == best_name
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/synthesiser/quality_summary.csv', index=False)
feat_df.to_csv('outputs/synthesiser/feature_comparison.csv', index=False)
print(f"  Saved: outputs/synthesiser/quality_summary.csv")
print(f"  Saved: outputs/synthesiser/feature_comparison.csv")


print(f"\nGenerating figures")

COLOURS = {
    'GaussianCopula': '#78909C',
    'CTGAN':          '#E65100',
    'CopulaGAN':      '#1565C0',
}

# Figure 1 — SDV Quality bar chart
fig, ax = plt.subplots(figsize=(10, 5))
labels  = ['Overall Score', 'Column Shapes', 'Column Pair Trends']
x = np.arange(len(labels))
w = 0.25
for i, row in summary_df.iterrows():
    vals = [row['SDV_Overall_%'], row['Column_Shapes_%'], row['Column_Pair_Trends_%']]
    bars = ax.bar(x + (i - 1) * w, vals, w,
                  label=row['Synthesiser'],
                  color=COLOURS[row['Synthesiser']], alpha=0.88)
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1) * w, v + 0.5, f"{v:.1f}%",
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('SDV Quality Score (%)', fontsize=12)
ax.set_ylim(0, 115)
ax.set_title('SDV Quality Evaluation — GaussianCopula vs CTGAN vs CopulaGAN\n'
             '(evaluated against real training data, unscaled)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out1 = os.path.join(FIG_DIR, 'quality_summary_bar.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out1}")

# Figure 2 — Feature means comparison
feats     = VALIDATION_FEATS
real_vals = [real_fraud[f].mean() for f in feats]
x = np.arange(len(feats))
w = 0.2

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - 1.5 * w, real_vals, w, label='Real Fraud', color='#C62828', alpha=0.90)
for i, (name, _) in enumerate(SYNTH_FILES.items()):
    syn_vals = [synth_fraud_dfs[name][f].mean() for f in feats]
    ax.bar(x + (i - 0.5) * w, syn_vals, w,
           label=name, color=list(COLOURS.values())[i], alpha=0.88)

ax.set_xticks(x)
ax.set_xticklabels(feats, fontsize=12)
ax.set_ylabel('Feature Mean Value', fontsize=12)
ax.set_title('Synthesiser Feature Means vs Real Fraud\n'
             'Ideal: all synthesiser bars should match Red (Real Fraud)',
             fontsize=13, fontweight='bold')
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out2 = os.path.join(FIG_DIR, 'quality_feature_means.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2}")



