

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RES_DIR = os.path.join('outputs', 'results')
FIG_DIR = os.path.join('outputs', 'figures')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')
os.makedirs(FIG_DIR, exist_ok=True)




# Classical models
CLASSICAL = ['LR', 'RF', 'XGB', 'LGBM']

DEEP_LEARNING = ['TabularResNet', 'ResNet', 'TabNet', 'TabPFN', 'VAE']
ALL_MODELS = CLASSICAL + DEEP_LEARNING

TECHNIQUES = ['No Handling', 'SMOTE', 'Random Undersampling', 'Cost-Sensitive', 'ADASYN']

# Colours — one consistent colour per model
C_MODEL = {
    'LR':            '#5C6BC0',
    'RF':            '#2E7D32',
    'XGB':           '#E65100',
    'LGBM':          '#6A1B9A',
    'TabularResNet': '#0277BD',
    'ResNet':        '#0277BD',   
    'TabNet':        '#558B2F',
    'TabPFN':        '#795548',
    'VAE':           '#607D8B',
}

C_TECH = {
    'No Handling':         '#1565C0',
    'SMOTE':               '#2E7D32',
    'Random Undersampling':'#E65100',
    'Cost-Sensitive':      '#6A1B9A',
    'ADASYN':              '#AD1457',
}


print("GENERATING THESIS FIGURES")


# Load files
print("\nLoading results files...")

def load(path):
    if not os.path.exists(path):
        print(f"  {path} not found — some figures will be skipped")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded: {path}  ({len(df)} rows)")
    return df

# Using merged files as primary source
exp1  = load(os.path.join(RES_DIR, 'merged_exp1_master.csv'))
exp2  = load(os.path.join(RES_DIR, 'merged_exp2_master.csv'))
exp3  = load(os.path.join(RES_DIR, 'merged_exp3_master.csv'))
comp  = load(os.path.join(RES_DIR, 'exp1_vs_exp2_comparison.csv'))
cross = load(os.path.join(RES_DIR, 'merged_cross_experiment_summary.csv'))

# Raw seeds for confusion matrices (Exp 1 classical only)
raw_1a = load(os.path.join(RAW_DIR, '1a_nohandling_raw.csv'))
raw_1b = load(os.path.join(RAW_DIR, '1b_smote_raw.csv'))
raw_1c = load(os.path.join(RAW_DIR, '1c_rus_raw.csv'))
raw_1d = load(os.path.join(RAW_DIR, '1d_costsensitive_raw.csv'))
raw_1e = load(os.path.join(RAW_DIR, '1e_adasyn_raw.csv'))


def normalise(df):
    """Rename columns so all code below uses Model + Condition."""
    if df is None:
        return None
    df = df.copy()
    if 'Classifier' in df.columns and 'Model' not in df.columns:
        df.rename(columns={'Classifier': 'Model'}, inplace=True)
    if 'Condition_Name' in df.columns and 'Condition' not in df.columns:
        df.rename(columns={'Condition_Name': 'Condition'}, inplace=True)
    return df

exp1  = normalise(exp1)
exp2  = normalise(exp2)
exp3  = normalise(exp3)

# ── Helper ─────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def model_colour(name):
    return C_MODEL.get(name, '#455A64')


def build_label(row):
    return f"{row['Model']}\n({row['Condition']})"


# Exp 1 heatmaps
if exp1 is not None:
    print("\nGenerating Experiment 1 heatmaps (classical models)...")
    classical_df = exp1[exp1['Model'].isin(CLASSICAL)].copy()

    metrics = [
        ('F1_mean',    'F1 Score (%) — binary, fraud class'),
        ('AUC_PR_mean','AUC-PR (%)'),
        ('MCC_mean',   'MCC (%)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Experiment 1 — Real Data Performance Heatmaps (Classical Models)\n'
        '(mean across 5 seeds, binary F1 for fraud class)',
        fontsize=13, fontweight='bold'
    )

    for ax, (metric, title) in zip(axes, metrics):
        pivot = classical_df.pivot_table(
            index='Condition', columns='Model',
            values=metric, aggfunc='mean'
        ) * 100
        row_order = [t for t in TECHNIQUES if t in pivot.index]
        col_order = [c for c in CLASSICAL if c in pivot.columns]
        pivot = pivot.reindex(index=row_order, columns=col_order)

        sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, linewidths=0.5, linecolor='white',
                    annot_kws={'size': 11})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Imbalance Technique', fontsize=10)
        ax.set_xticklabels(col_order, fontsize=10)
        ax.set_yticklabels(row_order, fontsize=9, rotation=0)

    plt.tight_layout()
    save(fig, 'exp1_heatmaps.png')


#Exp 1 master comparison ALL models
if exp1 is not None:
    print("Generating Experiment 1 master comparison (all models)...")

    df = exp1.copy()
    df['Label'] = df.apply(build_label, axis=1)
    df = df.sort_values('F1_mean', ascending=True)

    n = len(df)
    fig_height = max(10, n * 0.38)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    colours = [model_colour(r['Model']) for _, r in df.iterrows()]
    ax.barh(range(n), df['F1_mean'] * 100,
            xerr=df['F1_std'] * 100, color=colours, alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 1.2})

    ax.set_yticks(range(n))
    ax.set_yticklabels(df['Label'], fontsize=8)
    ax.set_xlabel('F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 1 — All Models × Technique Combinations\n'
        '(Classical + Deep Learning, mean ± std across 5 seeds)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0, 110)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['F1_mean'] * 100 + row['F1_std'] * 100 + 0.5, i,
                f"{row['F1_mean']*100:.1f}%", va='center', fontsize=7.5)

    # Legend — separate classical vs DL
    legend_handles = []
    for m in CLASSICAL:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (classical)"))
    for m in ['TabularResNet', 'TabNet', 'TabPFN', 'VAE']:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (deep learning)"))

    ax.legend(handles=legend_handles, title='Model', fontsize=8,
              loc='lower right', ncol=2)
    plt.tight_layout()
    save(fig, 'exp1_master_comparison.png')


# Exp 1 technique comparison (best F1 across all models)
if exp1 is not None:
    print("Generating Experiment 1 technique comparison...")

    tech_summary = (
        exp1[exp1['Condition'].isin(TECHNIQUES)]
        .groupby('Condition')
        .agg(F1_best=('F1_mean', 'max'))
        .reindex([t for t in TECHNIQUES if t in exp1['Condition'].values])
    )

    # Find std for the best model per technique
    def best_std(tech):
        sub = exp1[exp1['Condition'] == tech]
        if len(sub) == 0:
            return 0.0
        return sub.loc[sub['F1_mean'].idxmax(), 'F1_std']

    tech_summary['F1_best_std'] = [best_std(t) for t in tech_summary.index]

    x = np.arange(len(tech_summary))
    colours = [C_TECH.get(t, '#546E7A') for t in tech_summary.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, tech_summary['F1_best'] * 100,
           yerr=tech_summary['F1_best_std'] * 100,
           color=colours, alpha=0.88, width=0.6,
           error_kw={'ecolor': 'black', 'capsize': 5, 'linewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(tech_summary.index, fontsize=10)
    ax.set_ylabel('Best F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 1 — Best F1 per Imbalance Technique\n'
        '(best model across classical + deep learning, mean ± std)',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    for i, (tech, row) in enumerate(tech_summary.iterrows()):
        ax.text(i, row['F1_best'] * 100 + row['F1_best_std'] * 100 + 1,
                f"{row['F1_best']*100:.1f}%", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save(fig, 'exp1_technique_comparison.png')


# Exp 1 classical vs deep learning side-by-side 
if exp1 is not None:
    print("Generating Experiment 1 classical vs deep learning comparison...")

    # Best F1 per technique for classical group
    classical_best = (
        exp1[exp1['Model'].isin(CLASSICAL) & exp1['Condition'].isin(TECHNIQUES)]
        .groupby('Condition')['F1_mean'].max()
        .reindex([t for t in TECHNIQUES if t in exp1['Condition'].values])
        .fillna(0) * 100
    )

    dl_models_exp1 = [m for m in DEEP_LEARNING if m in exp1['Model'].values]
    dl_best = (
        exp1[exp1['Model'].isin(dl_models_exp1) & exp1['Condition'].isin(TECHNIQUES)]
        .groupby('Condition')['F1_mean'].max()
        .reindex(classical_best.index)
        .fillna(0) * 100
    )

    x = np.arange(len(classical_best))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, classical_best.values, w,
                label='Best Classical (LR/RF/XGB/LGBM)',
                color='#1565C0', alpha=0.88)
    b2 = ax.bar(x + w/2, dl_best.values, w,
                label='Best Deep Learning (ResNet/TabNet/TabPFN)',
                color='#BF360C', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(classical_best.index, fontsize=10)
    ax.set_ylabel('Best F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 1 — Classical vs Deep Learning\n'
        'Best F1 per Imbalance Technique',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.1f}%', ha='center', fontsize=8.5,
                    fontweight='bold', color='#1565C0')
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.1f}%', ha='center', fontsize=8.5,
                    fontweight='bold', color='#BF360C')

    plt.tight_layout()
    save(fig, 'exp1_classical_vs_dl.png')


# Exp 2 heatmaps (classical only)
if exp2 is not None:
    print("Generating Experiment 2 heatmaps (classical models)...")
    classical_df2 = exp2[exp2['Model'].isin(CLASSICAL)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Experiment 2 — Synthetic-Only Training Heatmaps (Classical Models)\n'
        '(mean across 5 seeds, binary F1 for fraud class)',
        fontsize=13, fontweight='bold'
    )

    for ax, (metric, title) in zip(axes, [
        ('F1_mean',    'F1 Score (%)'),
        ('AUC_PR_mean','AUC-PR (%)'),
        ('MCC_mean',   'MCC (%)'),
    ]):
        pivot = classical_df2.pivot_table(
            index='Condition', columns='Model',
            values=metric, aggfunc='mean'
        ) * 100
        row_order = [t for t in TECHNIQUES if t in pivot.index]
        col_order = [c for c in CLASSICAL if c in pivot.columns]
        pivot = pivot.reindex(index=row_order, columns=col_order)

        sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, linewidths=0.5, linecolor='white',
                    annot_kws={'size': 11})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Imbalance Technique', fontsize=10)
        ax.set_xticklabels(col_order, fontsize=10)
        ax.set_yticklabels(row_order, fontsize=9, rotation=0)

    plt.tight_layout()
    save(fig, 'exp2_heatmaps.png')


#Exp 2 master comparison ALL models
if exp2 is not None:
    print("Generating Experiment 2 master comparison (all models)...")

    df = exp2.copy()
    df['Label'] = df.apply(build_label, axis=1)
    df = df.sort_values('F1_mean', ascending=True)

    n = len(df)
    fig_height = max(10, n * 0.38)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    colours = [model_colour(r['Model']) for _, r in df.iterrows()]
    ax.barh(range(n), df['F1_mean'] * 100,
            xerr=df['F1_std'] * 100, color=colours, alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 1.2})

    ax.set_yticks(range(n))
    ax.set_yticklabels(df['Label'], fontsize=8)
    ax.set_xlabel('F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 2 — Synthetic-Only Training, All Models\n'
        '(Classical + Deep Learning, mean ± std across 5 seeds)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['F1_mean'] * 100 + row['F1_std'] * 100 + 0.5, i,
                f"{row['F1_mean']*100:.1f}%", va='center', fontsize=7.5)

    legend_handles = []
    for m in CLASSICAL:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (classical)"))
    for m in ['ResNet', 'TabNet', 'TabPFN', 'VAE']:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (deep learning)"))

    ax.legend(handles=legend_handles, title='Model', fontsize=8,
              loc='lower right', ncol=2)
    plt.tight_layout()
    save(fig, 'exp2_master_comparison.png')


# Exp 1 vs Exp 2 per-technique gap
if comp is not None:
    print("Generating Exp 1 vs Exp 2 per-technique comparison...")

    # Column names from exp1_vs_exp2_comparison.csv
    tech_col  = 'Technique'
    real_col  = 'Exp1_Best_F1_%_Real'
    syn_col   = 'Exp2_Best_F1_%_Synthetic'

    # Filter to the standard 5 techniques only
    comp_plot = comp[comp[tech_col].isin(TECHNIQUES)].copy()

    x = np.arange(len(comp_plot))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, comp_plot[real_col],  w,
                label='Exp 1 — Real Data', color='#1565C0', alpha=0.88)
    b2 = ax.bar(x + w/2, comp_plot[syn_col], w,
                label='Exp 2 — Synthetic Only', color='#F9A825', alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_plot[tech_col], fontsize=10)
    ax.set_ylabel('Best F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 1 vs Experiment 2\nBest F1 per Imbalance Technique (all models)',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%',
                ha='center', fontsize=9, fontweight='bold', color='#1565C0')
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%',
                ha='center', fontsize=9, fontweight='bold', color='#B8860B')

    plt.tight_layout()
    save(fig, 'exp1_vs_exp2_comparison.png')


# Exp 3 augmentation ratio sensitivity
ratio_raw = load(os.path.join(RAW_DIR, 'exp3a_augmentation_raw.csv'))
if ratio_raw is not None:
    print("Generating Experiment 3 ratio sensitivity...")
    ratio_raw = normalise(ratio_raw)
    model_col = 'Model' if 'Model' in ratio_raw.columns else 'Classifier'
    models_in_ratio = [m for m in (CLASSICAL + ['ResNet', 'TabNet', 'TabPFN'])
                       if m in ratio_raw[model_col].values]

    fig, ax = plt.subplots(figsize=(11, 6))
    for clf in models_in_ratio:
        sub = ratio_raw[ratio_raw[model_col] == clf].sort_values('Ratio')
        ax.errorbar(sub['Ratio'] * 100, sub['F1_mean'] * 100,
                    yerr=sub['F1_std'] * 100,
                    label=clf, color=model_colour(clf),
                    marker='o', linewidth=2, capsize=4, elinewidth=1.5)

    ax.set_xlabel('Augmentation Ratio — Target Fraud % in Combined Training', fontsize=11)
    ax.set_ylabel('Mean F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 3A — Augmentation Ratio Sensitivity\n'
        '(all models, mean ± std across 5 seeds)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(title='Model', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 5, 10, 25, 50])
    plt.tight_layout()
    save(fig, 'exp3_ratio_sensitivity.png')


#Exp 3 master comparison ALL models
if exp3 is not None:
    print("Generating Experiment 3 master comparison (all models)...")

    df = exp3.copy()
    # Skip VAE rows — constant output regardless of ratio
    df = df[df['Model'] != 'VAE']
    df['Label'] = df['Model'] + '\n(' + df['Condition'] + ')'
    df = df.sort_values('F1_mean', ascending=True)

    n = len(df)
    fig_height = max(10, n * 0.32)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    colours = [model_colour(r['Model']) for _, r in df.iterrows()]
    ax.barh(range(n), df['F1_mean'] * 100,
            xerr=df['F1_std'] * 100, color=colours, alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 1.2})

    ax.set_yticks(range(n))
    ax.set_yticklabels(df['Label'], fontsize=7.5)
    ax.set_xlabel('F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Experiment 3 — Real + Synthetic Augmentation, All Models\n'
        '(mean ± std across 5 seeds)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['F1_mean'] * 100 + row['F1_std'] * 100 + 0.5, i,
                f"{row['F1_mean']*100:.1f}%", va='center', fontsize=7)

    legend_handles = []
    for m in CLASSICAL:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (classical)"))
    for m in ['ResNet', 'TabNet', 'TabPFN']:
        if m in df['Model'].values:
            legend_handles.append(mpatches.Patch(color=model_colour(m), label=f"{m} (deep learning)"))

    ax.legend(handles=legend_handles, title='Model', fontsize=8,
              loc='lower right', ncol=2)
    plt.tight_layout()
    save(fig, 'exp3_master_comparison.png')


#Cross-experiment summary
if cross is not None:
    print("Generating cross-experiment summary...")

    # cross has columns: Scenario, Model, Condition, Ratio, F1_mean, F1_std, AUC_PR_mean, MCC_mean
    cross_plot = cross.copy()
    cross_plot = cross_plot.sort_values('F1_mean', ascending=True)

    scenario_colours = {
        'Exp1 Real Baseline':  '#1565C0',
        'Exp2 Synthetic Only': '#F9A825',
        'Exp3 Augmented':      '#2E7D32',
    }
    colours = [scenario_colours.get(s, '#546E7A') for s in cross_plot['Scenario']]

    labels = [
        f"{row['Scenario']}\n({row['Model']} — {row['Condition']})"
        for _, row in cross_plot.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.barh(range(len(cross_plot)), cross_plot['F1_mean'] * 100,
            xerr=cross_plot['F1_std'] * 100, color=colours, alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 4, 'linewidth': 1.5})

    ax.set_yticks(range(len(cross_plot)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('F1 Score (%) — binary, fraud class', fontsize=11)
    ax.set_title(
        'Cross-Experiment Best Performance Summary\n'
        '(mean ± std across 5 seeds)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(75, 90)
    ax.grid(axis='x', alpha=0.3)

    for i, (_, row) in enumerate(cross_plot.iterrows()):
        ax.text(row['F1_mean'] * 100 + row['F1_std'] * 100 + 0.2, i,
                f"{row['F1_mean']*100:.2f}%", va='center', fontsize=10, fontweight='bold')

    legend_handles = [
        mpatches.Patch(color='#1565C0', label='Experiment 1 — Real Data'),
        mpatches.Patch(color='#F9A825', label='Experiment 2 — Synthetic Only'),
        mpatches.Patch(color='#2E7D32', label='Experiment 3 — Augmented'),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    plt.tight_layout()
    save(fig, 'exp_cross_experiment.png')


# Confusion matrices Exp 1 (classical models)
CLASSICAL_4 = ['LR', 'RF', 'XGB', 'LGBM']

def plot_confusion_matrices(raw_df, condition_name, filename, seed=42):
    if raw_df is None:
        return
    print(f"Generating confusion matrices: {filename}...")

    raw_df = normalise(raw_df)
    model_col = 'Model' if 'Model' in raw_df.columns else 'Classifier'

    seed_df = raw_df[raw_df['Seed'] == seed]
    if len(seed_df) == 0:
        print(f"  Skipping — no data for seed {seed}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f'Experiment 1 — {condition_name}\nConfusion Matrices (seed={seed})',
        fontsize=13, fontweight='bold'
    )

    for ax, clf in zip(axes, CLASSICAL_4):
        row = seed_df[seed_df[model_col] == clf]
        if len(row) == 0:
            ax.set_visible(False)
            continue
        row = row.iloc[0]
        cm = np.array([[int(row['TN']), int(row['FP'])],
                       [int(row['FN']), int(row['TP'])]])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'],
                    annot_kws={'size': 14, 'fontweight': 'bold'},
                    linewidths=1, linecolor='white')
        ax.set_title(
            f"{clf}\nF1={row['F1']*100:.1f}%  Recall={row['Recall']*100:.1f}%\n"
            f"FPR={row['FPR']*100:.2f}%",
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)

    plt.tight_layout()
    save(fig, filename)


plot_confusion_matrices(raw_1a, 'No Handling',         'cm_exp1_nohandling.png')
plot_confusion_matrices(raw_1b,'SMOTE',  'cm_exp1_smote.png')
plot_confusion_matrices(raw_1c,'Random Undersampling', 'cm_exp1_rus.png')
plot_confusion_matrices(raw_1d, 'Cost-Sensitive',       'cm_exp1_costsensitive.png')
plot_confusion_matrices(raw_1e, 'ADASYN', 'cm_exp1_adasyn.png')


