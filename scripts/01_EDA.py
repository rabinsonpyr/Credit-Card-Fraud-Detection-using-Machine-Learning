

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

#Configuring..
RAW_FILE     = 'data/creditcard.csv'
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80/20 split

OUT_DIR = 'outputs'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
DAT_DIR = os.path.join(OUT_DIR, 'processed_data')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DAT_DIR, exist_ok=True)




print("EDA -- Exploratory Data Analysis")

#Loading credit card csv file
df = pd.read_csv(RAW_FILE)
print(f"\nLoaded: {RAW_FILE}\nLoaded. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nClass distribution:")
print(f"  Legitimate : {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.4f}%)")
print(f"  Fraudulent : {(df['Class']==1).sum():,}  ({(df['Class']==1).mean()*100:.4f}%)")
print(f"  Missing values : {df.isnull().sum().sum()}")

dup_count = df.duplicated().sum()
print(f"  Duplicate rows : {dup_count}")
df = df.drop_duplicates()
print(f" Dropped. New shape: {df.shape}")

#Doing anity checks
assert df.isnull().sum().sum() == 0, "SANITY FAIL: Missing values after dedup"
assert df['Class'].nunique() == 2,   "SANITY FAIL: Class have to be be binary"
print("\n   no missing values")
print("  binary class confirmed")

print("\n Summary Statistics")
print(df.describe().to_string())

#Class distribution

print("Section 3 — Class Distribution")


fraud_rate  = (df['Class']==1).mean()
imbal_ratio = int((df['Class']==0).sum() / (df['Class']==1).sum())
print(f"\nImbalance ratio: 1 fraud per {imbal_ratio} legitimate")
print(f"Naive accuracy: {(1-fraud_rate)*100:.2f}% (predict all = legitimate, catch zero fraud)")
print("Accuracy is NOT a valid metric. Need to use F1-score and AUC-PR.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
counts = df['Class'].value_counts()
ax1.bar(['Legitimate (0)', 'Fraudulent (1)'], counts.values, color=['#1565C0','#C62828'])
for i, v in enumerate(counts.values):
    ax1.text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold', fontsize=12)
ax1.set_title('Class Distribution (Raw Counts)', fontweight='bold')
ax1.set_ylabel('Number of Transactions')
ax2.pie(counts.values, labels=['Legitimate\n99.827%','Fraud\n0.173%'],
        colors=['#1565C0','#C62828'], autopct='%1.3f%%', startangle=90)
ax2.set_title('Class Distribution (Percentage)', fontweight='bold')
plt.suptitle('Kaggle ULB Credit Card Fraud Dataset — Class Imbalance', fontweight='bold', fontsize=14)
plt.tight_layout()
out = os.path.join(FIG_DIR, '01_class_distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# Analysing the amount

print("SECTION 4 — AMount Analysis")


legit_amt = df[df['Class']==0]['Amount']
fraud_amt = df[df['Class']==1]['Amount']
print("\nAmount stats:")
print(f"  Legitimate — Mean: EUR{legit_amt.mean():.2f}  Median: EUR{legit_amt.median():.2f}  Max: EUR{legit_amt.max():.2f}")
print(f"  Fraudulent — Mean: EUR{fraud_amt.mean():.2f}  Median: EUR{fraud_amt.median():.2f}  Max: EUR{fraud_amt.max():.2f}")
print("  Long-tail distribution. log_Amount will be used in experiments.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(legit_amt.clip(upper=2500), bins=60, density=True,
             alpha=0.6, color='#1565C0', label='Legitimate')
axes[0].hist(fraud_amt.clip(upper=2500), bins=60, density=True,
             alpha=0.6, color='#C62828', label='Fraud')
axes[0].set_title('Amount Distribution'); axes[0].legend()
axes[0].set_xlabel('Transaction Amount (EUR)')

log_legit = np.log1p(legit_amt); log_fraud = np.log1p(fraud_amt)
axes[1].hist(log_legit, bins=60, density=True, alpha=0.6, color='#1565C0', label='Legitimate')
axes[1].hist(log_fraud, bins=60, density=True, alpha=0.6, color='#C62828', label='Fraud')
axes[1].set_title('Log-Transformed Amount'); axes[1].legend()
axes[1].set_xlabel('log(1 + Amount)')

axes[2].boxplot([legit_amt.clip(upper=2000), fraud_amt.clip(upper=2000)],
                labels=['Legitimate','Fraud'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[2].set_title('Amount — Box Plot')
axes[2].set_ylabel('Transaction Amount (EUR)')
plt.suptitle('Transaction Amount: Legitimate vs Fraudulent', fontweight='bold', fontsize=13)
plt.tight_layout()
out = os.path.join(FIG_DIR, '02_amount_analysis.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# Temporal analysis

print("Section 5 — Time & Temporal Features")

df['hour_of_day'] = (df['Time'] / 3600) % 24
df['hour_sin']    = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos']    = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['log_Amount']  = np.log1p(df['Amount'])

hour_fraud_rate = df.groupby(df['hour_of_day'].astype(int))['Class'].mean() * 100
peak_hour = hour_fraud_rate.idxmax()
peak_rate = hour_fraud_rate.max()
print(f"\nPeak fraud hour : {peak_hour}:00  (rate = {peak_rate:.4f}%  vs overall {fraud_rate*100:.4f}%)")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
df[df['Class']==0]['hour_of_day'].plot.hist(ax=ax1, bins=24, density=True,
                                             alpha=0.6, color='#1565C0', label='Legitimate')
df[df['Class']==1]['hour_of_day'].plot.hist(ax=ax1, bins=24, density=True,
                                             alpha=0.6, color='#C62828', label='Fraud')
ax1.set_title('Transaction Hour Distribution'); ax1.legend(); ax1.set_xlabel('Hour of Day (0–24)')
ax2.bar(hour_fraud_rate.index, hour_fraud_rate.values, color='#E65100', alpha=0.85)
ax2.set_title('Fraud Rate by Hour of Day')
ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('Fraud Rate (%)')
plt.suptitle('Temporal Analysis of Transactions', fontweight='bold', fontsize=13)
plt.tight_layout()
out = os.path.join(FIG_DIR, '03_time_analysis.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
out = os.path.join(FIG_DIR, '04_time_density.png')
plt.figure(figsize=(14, 5))
plt.hist(df[df['Class']==0]['Time']/3600, bins=100, density=True, alpha=0.5, color='#1565C0', label='Legitimate')
plt.hist(df[df['Class']==1]['Time']/3600, bins=100, density=True, alpha=0.7, color='#C62828', label='Fraud')
plt.axvline(24, color='gray', ls='--', lw=1, label='Day 1')
plt.axvline(48, color='gray', ls='--', lw=1, label='Day 2')
plt.xlabel('Time (hours since first transaction)'); plt.ylabel('Density')
plt.title('Transaction Density Over Time', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

# PCA features analyzing

print("6 — PCA Feature Analyzing")


pca_feats = [f'V{i}' for i in range(1, 29)]
fraud_means = df[df['Class']==1][pca_feats].mean()
legit_means = df[df['Class']==0][pca_feats].mean()
discriminability = (fraud_means - legit_means).abs().sort_values(ascending=False)

print("\nTop 10 features:")
print(discriminability.head(10).to_string())

out = os.path.join(FIG_DIR, '05_feature_discriminability.png')
fig, ax = plt.subplots(figsize=(16, 5))
colors = ['#C62828' if v > discriminability.mean() else '#90CAF9'
          for v in discriminability.values]
ax.bar(discriminability.index, discriminability.values, color=colors)
ax.axhline(discriminability.mean(), color='navy', ls='--', lw=1.5, label='Average')
ax.set_title('Feature Discriminability  (Red = above average)', fontweight='bold')
ax.set_xlabel('PCA Feature'); ax.set_ylabel('|Mean(Fraud) - Mean(Legitimate)|')
ax.legend(); plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

top6 = discriminability.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, feat in enumerate(top6):
    ax = axes[i//3][i%3]
    ax.hist(df[df['Class']==0][feat], bins=60, density=True,
            alpha=0.6, color='#1565C0', label='Legitimate')
    ax.hist(df[df['Class']==1][feat], bins=60, density=True,
            alpha=0.6, color='#C62828', label='Fraud')
    diff = discriminability[feat]
    ax.set_title(f'{feat}  (diff={diff:.3f})', fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle('Top 6 Most Discriminative PCA Features', fontweight='bold', fontsize=14)
plt.tight_layout()
out = os.path.join(FIG_DIR, '06_top6_pca_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

fig, axes = plt.subplots(7, 4, figsize=(20, 28))
axes = axes.flatten()
for i, feat in enumerate(pca_feats):
    axes[i].hist(df[df['Class']==0][feat], bins=40, density=True,
                 alpha=0.6, color='#1565C0')
    axes[i].hist(df[df['Class']==1][feat], bins=40, density=True,
                 alpha=0.6, color='#C62828')
    axes[i].set_title(feat, fontweight='bold', fontsize=9)
for j in range(len(pca_feats), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('All V1-V28: Fraud (red) vs Legitimate (blue)', fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, '07_all_pca_features.png')
plt.savefig(out, dpi=100, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

#Correlation analysis
print("Section 7 — Correlation Analysis")


corr_features = pca_feats + ['log_Amount']
corr_matrix = df[corr_features + ['Class']].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, '08_correlation_heatmap.png')
plt.savefig(out, dpi=100, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

class_corr = df[corr_features + ['Class']].corr()['Class'].drop('Class').sort_values()
print("\nTop 10 features most correlated with fraud:")
print(class_corr.abs().sort_values(ascending=False).head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 10))
colors = ['#C62828' if v > 0 else '#1565C0' for v in class_corr.values]
ax.barh(class_corr.index, class_corr.values, color=colors, alpha=0.85)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Feature Correlation with Fraud (Class = 1)', fontweight='bold')
ax.set_xlabel('Pearson Correlation with Class Label')
for i, v in enumerate(class_corr.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), i,
            f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
plt.tight_layout()
out = os.path.join(FIG_DIR, '09_correlation_with_class.png')
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

# Outlier
print("Section 8 — Outlier")


outlier_results = {}
for col in pca_feats + ['Amount']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_results[col] = {'Outlier_Count': outliers, 'Outlier_pct': outliers/len(df)*100}

outlier_df = pd.DataFrame(outlier_results).T.sort_values('Outlier_pct', ascending=False)
print("\nOutlier Analysis (IQR method) — Top 15:")
print(outlier_df.head(15).to_string())

# Velocity Analyzing
print("Section 9 - Velocity Analyzing")
print("\nComputing global velocity features (may take ~30 seconds)...")

df_sorted = df.sort_values('Time').reset_index(drop=True)
times = df_sorted['Time'].values

for window_sec, col_name in [(60,'Velocity_60s'), (300,'Velocity_5min'), (1800,'Velocity_30min')]:
    counts = np.zeros(len(df_sorted), dtype=int)
    left = 0
    for right in range(len(df_sorted)):
        while times[right] - times[left] > window_sec:
            left += 1
        counts[right] = right - left + 1
    df_sorted[col_name] = counts

vel_cols = ['Velocity_60s', 'Velocity_5min', 'Velocity_30min']
print("\nVelocity stats (fraud mean vs legit mean):")
for col in vel_cols:
    fm = df_sorted[df_sorted['Class']==1][col].mean()
    lm = df_sorted[df_sorted['Class']==0][col].mean()
    print(f"  {col:<18} fraud={fm:.1f}   legit={lm:.1f}")

print("\nCorrelation of velocity features with Class:")
for col in vel_cols:
    corr = df_sorted[col].corr(df_sorted['Class'])
    print(f"  {col}: {corr:.6f}")

print("\nTop 5 PCA correlations (for comparison):")
print(class_corr.abs().sort_values(ascending=False).head(5).to_string())

# velocity correlations should be negligible
for col in vel_cols:
    corr = abs(df_sorted[col].corr(df_sorted['Class']))
    assert corr < 0.05, \
        f"Warning: Velocity {col} has higher correlation than expected ({corr:.4f})"
print("\n  all velocity correlations confirmed negligible (<0.05)")

print("  Negligible correlation. Card IDs are anonymised.")
print("  Velocity features EXCLUDED.")
print("  Limitation documented in methodology chapter.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(vel_cols):
    axes[i].hist(df_sorted[df_sorted['Class']==0][col], bins=50, density=True,
                 alpha=0.6, color='#1565C0', label='Legitimate')
    axes[i].hist(df_sorted[df_sorted['Class']==1][col], bins=50, density=True,
                 alpha=0.6, color='#C62828', label='Fraud')
    axes[i].set_title(f'Velocity ({col.split("_")[1]})', fontweight='bold')
    axes[i].legend()
plt.suptitle('Transaction Velocity: Fraud vs Legitimate', fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, '10_velocity_features.png')
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

# Split Strategy

print("Section 10 - Split Strategy")


idx = np.arange(len(df))
idx_train, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df['Class'].values
)

train_df = df.iloc[idx_train]
test_df  = df.iloc[idx_test]

print("\nStratified 80/20 split")
print(f"  Train : {len(train_df):,}  | Fraud: {(train_df['Class']==1).sum()} "
      f"({(train_df['Class']==1).mean()*100:.4f}%)")
print(f"  Test  : {len(test_df):,}  | Fraud: {(test_df['Class']==1).sum()} "
      f"({(test_df['Class']==1).mean()*100:.4f}%)")
print("  Fraud rate equal in both: YES")

# fraud rate preserved
assert abs((train_df['Class']==1).mean() - (test_df['Class']==1).mean()) < 0.0005, \
    "SANITY FAIL: Fraud rate not preserved"
print("  fraud rate preserved after split")



fig, ax = plt.subplots(figsize=(14, 4))
bins = np.linspace(0, 50, 50)
ax.hist(train_df['Time']/3600, bins=bins, density=True, alpha=0.6,
        color='#1565C0', label=f'Train split — {len(train_df):,} rows (Experiments 1–3)')
ax.hist(test_df['Time']/3600, bins=bins, density=True, alpha=0.6,
        color='#E65100', label=f'Test split  — {len(test_df):,} rows (ALL experiments)')
ax.set_xlabel('Time (hours since first transaction)')
ax.set_ylabel('Density')
ax.set_title('Stratified 80/20 Train/Test Split — Used Across All 3 Experiments',
             fontweight='bold')
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, '11_split_strategy.png')
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out}")

# Saving processed dataset

print("Section 11 — Saving Processed Data")


feature_cols = [f'V{i}' for i in range(1, 29)] + ['log_Amount', 'hour_sin', 'hour_cos']
assert len(feature_cols) == 31, "SANITY FAIL: Should have exactly 31 features"

processed = df[feature_cols + ['Class']].copy()
out_path = os.path.join(DAT_DIR, 'creditcard_processed.csv')
processed.to_csv(out_path, index=False)

print(f"\nShape   : {processed.shape}")
print(f"Features: {feature_cols}")
print(f"Fraud rate: {(processed['Class']==1).mean()*100:.4f}% ")
print(f"Saved: {out_path}")

# Summary

print("Key Findings")

print(f"""
1. Dataset Size
   Total          : {len(df):,}
   Legitimate (0) : {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.4f}%)
   Fraud      (1) : {(df['Class']==1).sum():,}  ({(df['Class']==1).mean()*100:.4f}%)
   Missing values : None,    Duplicates removed : {dup_count}

2. Class Imbalance
   Ratio         : 1 fraud per {imbal_ratio} legitimate
   Naive accuracy: {(1-fraud_rate)*100:.2f}%  (catch zero fraud)
   Use F1-score, AUC-PR, and MCC. Not accuracy.

3. Most Discriminative PCA Features
   Top 5 by mean difference: {', '.join(discriminability.head(5).index.tolist())}
   Top 3 positive corr with fraud : {class_corr.sort_values(ascending=False).head(3).index.tolist()}
   Top 3 negative corr with fraud : {class_corr.sort_values().head(3).index.tolist()}

4. Amount
   Fraud mean   : EUR {fraud_amt.mean():.2f}
   Fraud median : EUR {fraud_amt.median():.2f}
   Legit median : EUR {legit_amt.median():.2f}
   Long-tail. log_Amount used in all experiments.

5. Temporal Features
   Peak fraud hour : {peak_hour}:00  (rate = {peak_rate:.4f}%)
   hour_sin and hour_cos INCLUDED.

6. Veocity Features
   Excluded. Negligible correlation (<0.023). Card IDs anonymised.
   
""")


