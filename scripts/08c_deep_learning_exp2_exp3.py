

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, warnings, pickle
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (f1_score, average_precision_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("ERROR: pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
    sys.exit(1)

# Configuration
SCALER_FILE         = os.path.join('outputs', 'models',      'standard_scaler.pkl')
TEST_SET_FILE       = os.path.join('outputs', 'results',     'test_set.csv')
REAL_TRAIN_FILE     = os.path.join('outputs', 'results',     'train_set.csv')
SYNTHETIC_FILE      = os.path.join('outputs', 'synthesiser', 'ctgan_synthetic_300k.csv')
FRAUD_POOL_FILE     = os.path.join('outputs', 'synthesiser', 'best_fraud_250k.csv')

EXPECTED_TEST_ROWS  = 56746
EXPECTED_TEST_FRAUD = 95

SEEDS         = [42, 123, 456, 789, 1234]
INPUT_DIM     = 31
VAE_EPOCHS    = 50
TABNET_EPOCHS = 50
RESNET_EPOCHS = 50
BATCH_SIZE    = 512
LEARNING_RATE = 1e-3
AUG_RATIOS    = [0.01, 0.05, 0.10, 0.25, 0.50]

OUT_DIR      = 'outputs'
RES_DIR      = os.path.join(OUT_DIR, 'results')
RAW_DIR      = os.path.join(RES_DIR, 'raw_seeds')
EXP2_PARTIAL = os.path.join(RAW_DIR, 'dl_exp2_partial.csv')
EXP3_PARTIAL = os.path.join(RAW_DIR, 'dl_exp3_partial.csv')
for d in [RES_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)

PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURES + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL   = 'Class'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")


def load_partial(partial_file):
    """Load previously completed results. Returns list of result dicts."""
    if os.path.exists(partial_file):
        try:
            df      = pd.read_csv(partial_file)
            results = df.to_dict('records')
            print(f"  Resuming — loaded {len(results)} completed results "
                  f"from {os.path.basename(partial_file)}")
            return results
        except Exception as e:
            print(f"  Warning: could not read partial file ({e}) — starting fresh")
    return []

def is_done(results, model, condition, seed, experiment, ratio=None):
    """Return True if this exact run is already in results."""
    for r in results:
        match = (r.get('Model')      == model      and
                 r.get('Condition')  == condition  and
                 r.get('Seed')       == seed       and
                 r.get('Experiment') == experiment)
        if ratio is not None:
            match = match and (r.get('Ratio') == ratio)
        if match:
            return True
    return False

def save_result(result_dict, partial_file):
    """Append single result to partial CSV immediately after completion."""
    row_df = pd.DataFrame([result_dict])
    if os.path.exists(partial_file):
        row_df.to_csv(partial_file, mode='a', header=False, index=False)
    else:
        row_df.to_csv(partial_file, mode='w', header=True, index=False)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def evaluate(y_true, y_pred, y_prob):
    tp  = int(((y_pred==1)&(y_true==1)).sum())
    fp  = int(((y_pred==1)&(y_true==0)).sum())
    tn  = int(((y_pred==0)&(y_true==0)).sum())
    fn  = int(((y_pred==0)&(y_true==1)).sum())
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

def print_summary(label, seed_results):
    if not seed_results:
        return
    f1s  = [m['F1']      for m in seed_results]
    aprs = [m['AUC_PR']  for m in seed_results]
    mccs = [m['MCC']     for m in seed_results]
    rocs = [m['AUC_ROC'] for m in seed_results]
    print(f"\n=== {label} SUMMARY (mean ± std, {len(f1s)} seeds) ===")
    print(f"  F1      : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"  AUC-PR  : {np.mean(aprs)*100:.2f}% ± {np.std(aprs)*100:.2f}%")
    print(f"  AUC-ROC : {np.mean(rocs)*100:.2f}% ± {np.std(rocs)*100:.2f}%")
    print(f"  MCC     : {np.mean(mccs)*100:.2f}% ± {np.std(mccs)*100:.2f}%")

def make_master(results, groupby_cols):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    agg = df.groupby(groupby_cols).agg(
        F1_mean=('F1','mean'),         F1_std=('F1','std'),
        AUC_PR_mean=('AUC_PR','mean'), AUC_PR_std=('AUC_PR','std'),
        MCC_mean=('MCC','mean'),       MCC_std=('MCC','std'),
        AUC_ROC_mean=('AUC_ROC','mean'),
        Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),
        FPR_mean=('FPR','mean'),
        Seeds_completed=('Seed','count'),
    ).reset_index()
    return agg.sort_values('F1_mean', ascending=False).reset_index(drop=True)

def apply_imbalance(X_tr, y_tr, cond_name, seed):
    if cond_name == 'No Handling':
        return X_tr.copy(), y_tr.copy()
    elif cond_name == 'SMOTE':
        print(f"  Before SMOTE  : {Counter(y_tr)}")
        X_r, y_r = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"  After SMOTE   : {Counter(y_r)}")
        return X_r, y_r
    elif cond_name == 'Random Undersampling':
        print(f"  Before RUS    : {Counter(y_tr)}")
        X_r, y_r = RandomUnderSampler(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"  After RUS     : {Counter(y_r)}")
        return X_r, y_r
    elif cond_name == 'Cost-Sensitive':
        return X_tr.copy(), y_tr.copy()
    elif cond_name == 'ADASYN':
        print(f"  Before ADASYN : {Counter(y_tr)}")
        X_r, y_r = ADASYN(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"  After ADASYN  : {Counter(y_r)}")
        return X_r, y_r
    return X_tr.copy(), y_tr.copy()

# Model definitions
class VAE(nn.Module):
    def __init__(self, input_dim=31, latent_dim=8):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 16), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.fc_mean   = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterise(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(self.reparameterise(mean, logvar)), mean, logvar

    def reconstruction_error(self, x):
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            return ((x - x_recon) ** 2).mean(dim=1)


def vae_loss(x, x_recon, mean, logvar):
    recon = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl    = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (recon + kl) / x.size(0)


class TabularResNet(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.relu   = nn.ReLU()
        self.output = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.input_proj(x)
        h = self.relu(self.res_block1(h) + h)
        h = self.relu(self.res_block2(h) + h)
        return self.output(h).squeeze(1)


class TabularResNetLogits(TabularResNet):
    """Without sigmoid — for BCEWithLogitsLoss (cost-sensitive)."""
    def forward(self, x):
        h = self.input_proj(x)
        h = self.relu(self.res_block1(h) + h)
        h = self.relu(self.res_block2(h) + h)
        return self.output[0](h).squeeze(1)


#Load data
print("EXPERIMENT 2 & 3 — DEEP LEARNING (VAE | TabNet | ResNet)")

print("\nLOADING DATA")


for path in [SCALER_FILE, TEST_SET_FILE, REAL_TRAIN_FILE,
             SYNTHETIC_FILE, FRAUD_POOL_FILE]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)

test_df  = pd.read_csv(TEST_SET_FILE)
X_test   = test_df[FEATURE_COLS].values
y_test   = test_df[TARGET_COL].values
assert len(y_test) == EXPECTED_TEST_ROWS,   "SANITY FAIL: test row count"
assert y_test.sum() == EXPECTED_TEST_FRAUD, "SANITY FAIL: test fraud count"
print(f"Test set       : {len(y_test):,} rows ({y_test.sum()} fraud)")

train_df = pd.read_csv(REAL_TRAIN_FILE)
X_train  = train_df[FEATURE_COLS].values
y_train  = train_df[TARGET_COL].values
n_legit  = int((y_train==0).sum())
n_fraud  = int((y_train==1).sum())
print(f"Real train     : {len(y_train):,} rows ({n_fraud} fraud, {n_legit:,} legit)")

syn_df  = pd.read_csv(SYNTHETIC_FILE)
y_syn   = syn_df[TARGET_COL].values
print(f"CTGAN synthetic: {len(y_syn):,} rows ({y_syn.sum()} fraud)")

pool_df = pd.read_csv(FRAUD_POOL_FILE)
y_pool  = pool_df[TARGET_COL].values
print(f"Fraud pool     : {len(y_pool):,} synthetic fraud cases")

with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

X_syn   = scaler.transform(syn_df[FEATURE_COLS].values)
X_pool  = scaler.transform(pool_df[FEATURE_COLS].values)
print("Scaler applied ")

for name, arr in [('X_syn',X_syn),('X_pool',X_pool),
                  ('X_test',X_test),('X_train',X_train)]:
    assert not np.any(np.isnan(arr)), f"NaN in {name}"
print("All NaN checks passed")

# Pre-build augmented datasets for Experiment 3
aug_datasets = {}
for ratio in AUG_RATIOS:
    label        = f"{int(ratio*100)}%"
    n_syn_needed = max(0, int((ratio * n_legit) / (1 - ratio) - n_fraud))
    n_syn_needed = min(n_syn_needed, len(y_pool))
    rng          = np.random.default_rng(42)
    idx          = rng.choice(len(y_pool), size=n_syn_needed, replace=False)
    X_aug        = np.vstack([X_train, X_pool[idx]])
    y_aug        = np.concatenate([y_train, y_pool[idx]])
    aug_datasets[label] = (X_aug, y_aug)
    print(f"  Aug {label:>4}: {len(y_aug):,} rows  "
          f"({(y_aug==1).sum():,} fraud = {(y_aug==1).mean()*100:.2f}%)")

supervised_conditions = [
    'No Handling', 'SMOTE', 'Random Undersampling',
    'Cost-Sensitive', 'ADASYN'
]



print("EXPERIMENT 2 — SYNTHETIC-ONLY TRAINING (Deep Learning)")
print("Train: CTGAN synthetic | Test: Fixed real held-out test set")


exp2_results = load_partial(EXP2_PARTIAL)




vae_exp2_new = []
for seed in SEEDS:
    if is_done(exp2_results, 'VAE', 'VAE (Unsupervised)', seed, 'Exp2'):
        print(f"\n--- VAE Exp2 Seed {seed} --- SKIPPED (already completed)")
        continue

    print(f"\n--- VAE Exp2 Seed {seed} ---")
    set_seed(seed)

    X_syn_legit = X_syn[y_syn == 0]
    print(f"  Training on {len(X_syn_legit):,} synthetic legitimate rows")

    loader    = DataLoader(
        TensorDataset(torch.tensor(X_syn_legit, dtype=torch.float32).to(device)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    model     = VAE(INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(VAE_EPOCHS):
        ep_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_r, mu, lv = model(batch)
            loss = vae_loss(batch, x_r, mu, lv)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{VAE_EPOCHS}  "
                  f"Loss: {ep_loss/len(loader):.6f}")

    model.eval()
    recon_train = model.reconstruction_error(
        torch.tensor(X_syn, dtype=torch.float32).to(device)
    ).cpu().numpy()

    best_f1, best_thresh = 0, 0
    for pct in np.linspace(90, 99.9, 100):
        t  = np.percentile(recon_train, pct)
        f1 = f1_score(y_syn, (recon_train >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"  Best threshold: {best_thresh:.6f} (train F1={best_f1*100:.2f}%)")

    recon_test = model.reconstruction_error(
        torch.tensor(X_test, dtype=torch.float32).to(device)
    ).cpu().numpy()
    y_pred = (recon_test >= best_thresh).astype(int)
    y_prob = np.clip(
        (recon_test - recon_test.min()) /
        (recon_test.max() - recon_test.min() + 1e-8), 0, 1
    )

    m      = evaluate(y_test, y_pred, y_prob)
    print_metrics(f"VAE Exp2 (seed={seed})", m)
    result = {'Model':'VAE','Condition':'VAE (Unsupervised)',
              'Experiment':'Exp2','Seed':seed,**m}
    save_result(result, EXP2_PARTIAL)
    exp2_results.append(result)
    vae_exp2_new.append(m)

if vae_exp2_new:
    print_summary("VAE Exp2", vae_exp2_new)






for cond_name in supervised_conditions:

    print(f"TabNet Exp2 — {cond_name}")

    new_seed_results = []

    for seed in SEEDS:
        if is_done(exp2_results, 'TabNet', cond_name, seed, 'Exp2'):
            print(f"\n--- Seed {seed} --- SKIPPED (already completed)")
            continue

        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        X_tr, y_tr = apply_imbalance(X_syn, y_syn, cond_name, seed)
        assert not np.any(np.isnan(X_tr))

        n_fraud_tr = (y_tr==1).sum()
        n_legit_tr = (y_tr==0).sum()
        spw        = n_legit_tr / n_fraud_tr if n_fraud_tr > 0 else 1

        clf    = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3, gamma=1.3,
            n_independent=2, n_shared=2,
            momentum=0.02, epsilon=1e-15, seed=seed,
            device_name='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0
        )
        fit_kw = dict(max_epochs=TABNET_EPOCHS, patience=20,
                      batch_size=BATCH_SIZE, virtual_batch_size=128)
        if cond_name == 'Cost-Sensitive':
            clf.fit(X_tr.astype(np.float32), y_tr.astype(int),
                    weights={0:1.0, 1:float(spw)}, **fit_kw)
        else:
            clf.fit(X_tr.astype(np.float32), y_tr.astype(int), **fit_kw)

        y_pred = clf.predict(X_test.astype(np.float32))
        y_prob = clf.predict_proba(X_test.astype(np.float32))[:, 1]
        m      = evaluate(y_test, y_pred, y_prob)
        print_metrics(f"TabNet Exp2 + {cond_name} (seed={seed})", m)

        result = {'Model':'TabNet','Condition':cond_name,
                  'Experiment':'Exp2','Seed':seed,**m}
        save_result(result, EXP2_PARTIAL)
        exp2_results.append(result)
        new_seed_results.append(m)

    if new_seed_results:
        print_summary(f"TabNet Exp2 + {cond_name}", new_seed_results)

# Tabular ResNet 

for cond_name in supervised_conditions:

    print(f"ResNet Exp2 — {cond_name}")

    new_seed_results = []

    for seed in SEEDS:
        if is_done(exp2_results, 'ResNet', cond_name, seed, 'Exp2'):
            print(f"\n--- Seed {seed} --- SKIPPED (already completed)")
            continue

        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        X_tr, y_tr = apply_imbalance(X_syn, y_syn, cond_name, seed)
        assert not np.any(np.isnan(X_tr))

        n_fraud_tr = (y_tr==1).sum()
        n_legit_tr = (y_tr==0).sum()
        spw        = n_legit_tr / n_fraud_tr if n_fraud_tr > 0 else 1
        use_logits = (cond_name == 'Cost-Sensitive')

        if use_logits:
            model     = TabularResNetLogits(INPUT_DIM).to(device)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([spw], dtype=torch.float32).to(device)
            )
        else:
            model     = TabularResNet(INPUT_DIM).to(device)
            criterion = nn.BCELoss()

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        loader    = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32).to(device),
                torch.tensor(y_tr, dtype=torch.float32).to(device)
            ), batch_size=BATCH_SIZE, shuffle=True
        )

        model.train()
        for epoch in range(RESNET_EPOCHS):
            ep_loss = 0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{RESNET_EPOCHS}  "
                      f"Loss: {ep_loss/len(loader):.6f}")

        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        y_prob = (torch.sigmoid(raw) if use_logits else raw).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        m = evaluate(y_test, y_pred, y_prob)
        print_metrics(f"ResNet Exp2 + {cond_name} (seed={seed})", m)

        result = {'Model':'ResNet','Condition':cond_name,
                  'Experiment':'Exp2','Seed':seed,**m}
        save_result(result, EXP2_PARTIAL)
        exp2_results.append(result)
        new_seed_results.append(m)

    if new_seed_results:
        print_summary(f"ResNet Exp2 + {cond_name}", new_seed_results)


exp2_master = make_master(exp2_results, ['Model','Condition'])
if not exp2_master.empty:
    exp2_master['Model_Label'] = (exp2_master['Model'] + ' ('
                                  + exp2_master['Condition'] + ')')

    print("EXPERIMENT 2 MASTER RESULTS — DEEP LEARNING")

    
    for _, row in exp2_master.iterrows():
        done = f"({int(row['Seeds_completed'])}/5 seeds)"
        print(f"  {row['Model_Label']:<48} "
              f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
              f"AUC-PR={row['AUC_PR_mean']*100:.2f}%  "
              f"MCC={row['MCC_mean']*100:.2f}%  {done}")
    exp2_master.to_csv(os.path.join(RES_DIR,'master_results_dl_exp2.csv'),
                       index=False)
    pd.DataFrame(exp2_results).to_csv(
        os.path.join(RAW_DIR,'dl_exp2_raw.csv'), index=False)
    print(f"\n  Saved: master_results_dl_exp2.csv")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — AUGMENTED TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print("EXPERIMENT 3 — AUGMENTED TRAINING (Deep Learning)")
print(f"Ratios: {[f'{int(r*100)}%' for r in AUG_RATIOS]}")
print("Train: Real + CTGAN synthetic fraud | Test: Fixed real held-out test set")


exp3_results = load_partial(EXP3_PARTIAL)


print("EXP 3 — VAE (Real legitimate rows — same across all ratios)")


vae_exp3_new  = []
first_ratio   = list(aug_datasets.keys())[0]
for seed in SEEDS:
    if is_done(exp3_results, 'VAE', 'VAE (Unsupervised)', seed,
               'Exp3', ratio=first_ratio):
        print(f"\n--- VAE Exp3 Seed {seed} --- SKIPPED (already completed)")
        continue

    print(f"\n--- VAE Exp3 Seed {seed} ---")
    set_seed(seed)

    X_real_legit = X_train[y_train == 0]
    print(f"  Training on {len(X_real_legit):,} real legitimate rows")

    loader    = DataLoader(
        TensorDataset(torch.tensor(X_real_legit, dtype=torch.float32).to(device)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    model     = VAE(INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(VAE_EPOCHS):
        ep_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_r, mu, lv = model(batch)
            loss = vae_loss(batch, x_r, mu, lv)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{VAE_EPOCHS}  "
                  f"Loss: {ep_loss/len(loader):.6f}")

    model.eval()
    recon_train = model.reconstruction_error(
        torch.tensor(X_train, dtype=torch.float32).to(device)
    ).cpu().numpy()

    best_f1, best_thresh = 0, 0
    for pct in np.linspace(90, 99.9, 100):
        t  = np.percentile(recon_train, pct)
        f1 = f1_score(y_train, (recon_train >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"  Best threshold: {best_thresh:.6f} (train F1={best_f1*100:.2f}%)")

    recon_test = model.reconstruction_error(
        torch.tensor(X_test, dtype=torch.float32).to(device)
    ).cpu().numpy()
    y_pred = (recon_test >= best_thresh).astype(int)
    y_prob = np.clip(
        (recon_test - recon_test.min()) /
        (recon_test.max() - recon_test.min() + 1e-8), 0, 1
    )

    m = evaluate(y_test, y_pred, y_prob)
    print_metrics(f"VAE Exp3 (seed={seed})", m)
    vae_exp3_new.append(m)

    # Record same result for all ratios (VAE unaffected by augmentation)
    for ratio_label in aug_datasets:
        result = {'Model':'VAE','Condition':'VAE (Unsupervised)',
                  'Ratio':ratio_label,'Experiment':'Exp3','Seed':seed,**m}
        save_result(result, EXP3_PARTIAL)
        exp3_results.append(result)

if vae_exp3_new:
    print_summary("VAE Exp3", vae_exp3_new)



print("EXP 3 — TABNET (5 ratios × 5 seeds)")


for ratio_label, (X_aug, y_aug) in aug_datasets.items():
    cond_name = f"Synth Aug ({ratio_label})"

    print(f"TabNet Exp3 — ratio {ratio_label}  "
          f"({len(y_aug):,} rows, "
          f"{(y_aug==1).sum():,} fraud = {(y_aug==1).mean()*100:.2f}%)")

    new_seed_results = []

    for seed in SEEDS:
        if is_done(exp3_results, 'TabNet', cond_name, seed,
                   'Exp3', ratio=ratio_label):
            print(f"\n--- Seed {seed} --- SKIPPED (already completed)")
            continue

        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        clf = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3, gamma=1.3,
            n_independent=2, n_shared=2,
            momentum=0.02, epsilon=1e-15, seed=seed,
            device_name='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0
        )
        clf.fit(X_aug.astype(np.float32), y_aug.astype(int),
                max_epochs=TABNET_EPOCHS, patience=20,
                batch_size=BATCH_SIZE, virtual_batch_size=128)

        y_pred = clf.predict(X_test.astype(np.float32))
        y_prob = clf.predict_proba(X_test.astype(np.float32))[:, 1]
        m      = evaluate(y_test, y_pred, y_prob)
        print_metrics(f"TabNet Exp3 {ratio_label} (seed={seed})", m)

        result = {'Model':'TabNet','Condition':cond_name,
                  'Ratio':ratio_label,'Experiment':'Exp3','Seed':seed,**m}
        save_result(result, EXP3_PARTIAL)
        exp3_results.append(result)
        new_seed_results.append(m)

    if new_seed_results:
        print_summary(f"TabNet Exp3 ratio={ratio_label}", new_seed_results)

# ── Exp3 Tabular ResNet ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("EXP 3 — TABULAR RESNET (5 ratios × 5 seeds)")
print(f"{'─'*60}")

for ratio_label, (X_aug, y_aug) in aug_datasets.items():
    cond_name = f"Synth Aug ({ratio_label})"

    print(f"ResNet Exp3 — ratio {ratio_label}  "
          f"({len(y_aug):,} rows, "
          f"{(y_aug==1).sum():,} fraud = {(y_aug==1).mean()*100:.2f}%)")

    new_seed_results = []

    for seed in SEEDS:
        if is_done(exp3_results, 'ResNet', cond_name, seed,
                   'Exp3', ratio=ratio_label):
            print(f"\n--- Seed {seed} --- SKIPPED (already completed)")
            continue

        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_aug, dtype=torch.float32).to(device),
                torch.tensor(y_aug, dtype=torch.float32).to(device)
            ), batch_size=BATCH_SIZE, shuffle=True
        )
        model     = TabularResNet(INPUT_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        model.train()
        for epoch in range(RESNET_EPOCHS):
            ep_loss = 0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = nn.BCELoss()(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{RESNET_EPOCHS}  "
                      f"Loss: {ep_loss/len(loader):.6f}")

        model.eval()
        with torch.no_grad():
            y_prob = model(
                torch.tensor(X_test, dtype=torch.float32).to(device)
            ).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        m = evaluate(y_test, y_pred, y_prob)
        print_metrics(f"ResNet Exp3 {ratio_label} (seed={seed})", m)

        result = {'Model':'ResNet','Condition':cond_name,
                  'Ratio':ratio_label,'Experiment':'Exp3','Seed':seed,**m}
        save_result(result, EXP3_PARTIAL)
        exp3_results.append(result)
        new_seed_results.append(m)

    if new_seed_results:
        print_summary(f"ResNet Exp3 ratio={ratio_label}", new_seed_results)


exp3_master = make_master(exp3_results, ['Model','Condition','Ratio'])
if not exp3_master.empty:
    exp3_master['Model_Label'] = (exp3_master['Model'] + ' ('
                                  + exp3_master['Condition'] + ')')

    print("EXPERIMENT 3 MASTER RESULTS — DEEP LEARNING")


    for _, row in exp3_master.iterrows():
        done = f"({int(row['Seeds_completed'])}/5 seeds)"
        print(f"  {row['Model_Label']:<52} "
              f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
              f"AUC-PR={row['AUC_PR_mean']*100:.2f}%  "
              f"MCC={row['MCC_mean']*100:.2f}%  {done}")
    exp3_master.to_csv(os.path.join(RES_DIR,'master_results_dl_exp3.csv'),
                       index=False)
    pd.DataFrame(exp3_results).to_csv(
        os.path.join(RAW_DIR,'dl_exp3_raw.csv'), index=False)
    print("\n  Saved: master_results_dl_exp3.csv")

