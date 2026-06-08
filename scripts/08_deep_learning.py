

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Working directory: {os.getcwd()}")

import sys, json, time, warnings, pickle
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

#Check required libraries
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("ERROR: pytorch-tabnet not installed.")
    print("Run: pip install pytorch-tabnet")
    sys.exit(1)

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("ERROR: tabpfn not installed.")
    print("Run: pip install tabpfn")
    sys.exit(1)

# Configuration
SCALER_FILE     = os.path.join('outputs', 'models',   'standard_scaler.pkl')
TEST_SET_FILE   = os.path.join('outputs', 'results',  'test_set.csv')
REAL_TRAIN_FILE = os.path.join('outputs', 'results',  'train_set.csv')

# Expected test set values — update if Experiment 1 is re-run
EXPECTED_TEST_ROWS  = 56746
EXPECTED_TEST_FRAUD = 95

SEEDS        = [42, 123, 456, 789, 1234]
RANDOM_STATE = 42
INPUT_DIM    = 31   # 28 PCA features + log_Amount + hour_sin + hour_cos

# TabPFN row limit
TABPFN_MAX_TRAIN_ROWS = 1000

# Training settings
VAE_EPOCHS     = 50
TABNET_EPOCHS  = 100
RESNET_EPOCHS  = 50
BATCH_SIZE     = 512
LEARNING_RATE  = 1e-3

OUT_DIR = 'outputs'
RES_DIR = os.path.join(OUT_DIR, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
RAW_DIR = os.path.join(RES_DIR, 'raw_seeds')
MOD_DIR = os.path.join(OUT_DIR, 'models')

for d in [RES_DIR, FIG_DIR, RAW_DIR, MOD_DIR]:
    os.makedirs(d, exist_ok=True)

PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURES + ['log_Amount', 'hour_sin', 'hour_cos']
TARGET_COL   = 'Class'


# Device setup 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Helper functions
def evaluate(y_true, y_pred, y_prob):
    """Binary F1 (fraud class), AUC-PR, MCC, Precision, Recall, AUC-ROC, FPR."""
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
    print(f"    MCC                      : {m['MCC']*100:.2f}%")
    print(f"    Precision                : {m['Precision']*100:.2f}%")
    print(f"    Recall                   : {m['Recall']*100:.2f}%")
    print(f"    FPR                      : {m['FPR']*100:.2f}%  ({m['FP']} false alarms)")
    print(f"    TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Load data

print("DEEP LEARNING MODELS")

print("\nLOADING DATA")


for path in [SCALER_FILE, TEST_SET_FILE, REAL_TRAIN_FILE]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        print("Run 02_experiment1_real_data.py first.")
        sys.exit(1)

# Load test set
test_df = pd.read_csv(TEST_SET_FILE)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

print(f"Test set loaded : {len(y_test):,} transactions")
print(f"  Fraud         : {y_test.sum()} ({y_test.mean()*100:.4f}%)")

assert len(y_test) == EXPECTED_TEST_ROWS, \
    f"Test set has {len(y_test)} rows, expected {EXPECTED_TEST_ROWS}"
assert y_test.sum() == EXPECTED_TEST_FRAUD, \
    f"Test set has {y_test.sum()} fraud, expected {EXPECTED_TEST_FRAUD}"
print(f"  matches Experiment 1 ({EXPECTED_TEST_ROWS} rows, {EXPECTED_TEST_FRAUD} fraud)")

# Load training set
train_df = pd.read_csv(REAL_TRAIN_FILE)
X_train  = train_df[FEATURE_COLS].values
y_train  = train_df[TARGET_COL].values
n_legit  = (y_train==0).sum()
n_fraud  = (y_train==1).sum()

print(f"\nTraining set loaded : {len(y_train):,} transactions")
print(f"  Fraud : {n_fraud} ({n_fraud/len(y_train)*100:.4f}%)")
print(f"  Legitimate : {n_legit:,}")

assert not np.any(np.isnan(X_train)), "NaN in training data"
assert not np.any(np.isnan(X_test)),  "NaN in test data"
print(" no NaN/Inf in data")

# MODEL DEFINITIONS

class VAE(nn.Module):
    
    def __init__(self, input_dim=31, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder — outputs mean and log variance of latent distribution
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x):
        h      = self.encoder_shared(x)
        mean   = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterise(self, mean, logvar):
        """Reparameterisation trick: z = mean + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z            = self.reparameterise(mean, logvar)
        x_recon      = self.decode(z)
        return x_recon, mean, logvar

    def reconstruction_error(self, x):
        """Mean squared reconstruction error per sample (no gradient)."""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            return ((x - x_recon) ** 2).mean(dim=1)


def vae_loss(x, x_recon, mean, logvar):
  
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)


class TabularResNet(nn.Module):

    def __init__(self, input_dim=31, hidden_dim=64):
        super().__init__()

        # Project input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.relu  = nn.ReLU()
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project input
        h = self.input_proj(x)

        # Residual block 1: output = relu(F(h) + h)
        h = self.relu(self.res_block1(h) + h)

        # Residual block 2: output = relu(F(h) + h)
        h = self.relu(self.res_block2(h) + h)

        return self.output(h).squeeze(1)


# Imbalance conditions
supervised_conditions = {
    '4B_NoHandling':    'No Handling',
    '4C_SMOTE':         'SMOTE',
    '4D_RUS':           'Random Undersampling',
    '4E_CostSensitive': 'Cost-Sensitive',
    '4F_ADASYN':        'ADASYN',
}

def apply_imbalance(X_tr, y_tr, cond_name, seed):
    """Apply imbalance handling technique and return resampled data."""
    if cond_name == 'No Handling':
        return X_tr.copy(), y_tr.copy()
    elif cond_name == 'SMOTE':
        print(f"Before SMOTE : {Counter(y_tr)}")
        X_r, y_r = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"After SMOTE  : {Counter(y_r)}")
        return X_r, y_r
    elif cond_name == 'Random Undersampling':
        print(f"Before RUS : {Counter(y_tr)}")
        X_r, y_r = RandomUnderSampler(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"After RUS  : {Counter(y_r)}")
        return X_r, y_r
    elif cond_name == 'Cost-Sensitive':
        return X_tr.copy(), y_tr.copy()
    elif cond_name == 'ADASYN':
        print(f"Before ADASYN : {Counter(y_tr)}")
        X_r, y_r = ADASYN(random_state=seed).fit_resample(X_tr, y_tr)
        print(f"After ADASYN  : {Counter(y_r)}")
        return X_r, y_r
    return X_tr.copy(), y_tr.copy()

# VAE 

print("MODEL 1: VAE (Variational Autoencoder, Unsupervised)")


vae_all_results  = []
vae_thresh_list  = []

for seed in SEEDS:
    print(f"\n--- VAE seed {seed} ---")
    set_seed(seed)

    # Train ONLY on legitimate transactions
    X_legit        = X_train[y_train == 0]
    print(f"  Training on {len(X_legit):,} legitimate transactions only")

    X_legit_tensor = torch.tensor(X_legit, dtype=torch.float32).to(device)
    dataset        = TensorDataset(X_legit_tensor)
    loader         = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = VAE(INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(VAE_EPOCHS):
        epoch_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_recon, mean, logvar = model(batch)
            loss = vae_loss(batch, x_recon, mean, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{VAE_EPOCHS}  "
                  f"Loss: {epoch_loss/len(loader):.6f}")

    # Find threshold using reconstruction error on full training set
    model.eval()
    X_train_tensor     = torch.tensor(X_train, dtype=torch.float32).to(device)
    recon_errors_train = model.reconstruction_error(X_train_tensor).cpu().numpy()

    best_f1_thresh = 0
    best_threshold = 0
    for pct in np.linspace(90, 99.9, 100):
        thresh = np.percentile(recon_errors_train, pct)
        preds  = (recon_errors_train >= thresh).astype(int)
        f1     = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1_thresh:
            best_f1_thresh = f1
            best_threshold = thresh

    print(f"  Best threshold: {best_threshold:.6f} "
          f"(F1={best_f1_thresh*100:.2f}% on train)")
    vae_thresh_list.append(best_threshold)

    # Evaluate on real held-out test set
    X_test_tensor     = torch.tensor(X_test, dtype=torch.float32).to(device)
    recon_errors_test = model.reconstruction_error(X_test_tensor).cpu().numpy()

    y_pred = (recon_errors_test >= best_threshold).astype(int)
    # Normalise reconstruction error to [0,1] as probability proxy
    y_prob = (recon_errors_test - recon_errors_test.min()) / \
             (recon_errors_test.max() - recon_errors_test.min() + 1e-8)
    y_prob = np.clip(y_prob, 0, 1)

    m = evaluate(y_test, y_pred, y_prob)
    print_metrics(f"VAE (seed={seed})", m)

    vae_all_results.append({
        'Model': 'VAE', 'Condition': '4A_VAE',
        'Condition_Name': 'VAE (Unsupervised)',
        'Classifier': 'VAE', 'Seed': seed,
        'Threshold': best_threshold, **m
    })

# VAE summary
vae_df = pd.DataFrame(vae_all_results)
f1s    = vae_df['F1'].values
aprs   = vae_df['AUC_PR'].values
mccs   = vae_df['MCC'].values
print(f"\n=== VAE SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")
print(f"  F1 (binary, fraud class) : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
print(f"  AUC-PR : {np.mean(aprs)*100:.2f}% ± {np.std(aprs)*100:.2f}%")
print(f"  MCC: {np.mean(mccs)*100:.2f}% ± {np.std(mccs)*100:.2f}%")
print(f"  Mean threshold : {np.mean(vae_thresh_list):.6f}")

#  TabNet 

print("SECTION 3 — MODEL 2: TABNET (Supervised)")


tabnet_all_results = []

for cond_key, cond_name in supervised_conditions.items():
    print(f"TabNet — {cond_name}")


    seed_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        X_tr, y_tr = apply_imbalance(X_train, y_train, cond_name, seed)

        assert not np.any(np.isnan(X_tr)), \
            f"NaN in training data after {cond_name}"

        # Compute class weights for cost-sensitive
        n_fraud_tr = (y_tr==1).sum()
        n_legit_tr = (y_tr==0).sum()
        spw = n_legit_tr / n_fraud_tr if n_fraud_tr > 0 else 1

        # TabNet requires float32 arrays
        X_tr_32  = X_tr.astype(np.float32)
        X_te_32  = X_test.astype(np.float32)
        y_tr_int = y_tr.astype(int)

        # Build TabNet classifier
        clf = TabNetClassifier(
            n_d=16, n_a=16,           # Width of decision step output
            n_steps=3,                # Number of sequential attention steps
            gamma=1.3,                # Coefficient for feature reusage
            n_independent=2,          # Number of independent GLU layers
            n_shared=2,               # Number of shared GLU layers
            momentum=0.02,
            epsilon=1e-15,
            seed=seed,
            device_name='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0,
        )

        if cond_name == 'Cost-Sensitive':
            # Pass class weights via fit
            weights = {0: 1.0, 1: float(spw)}
            clf.fit(
                X_tr_32, y_tr_int,
                weights=weights,
                max_epochs=TABNET_EPOCHS,
                patience=20,
                batch_size=BATCH_SIZE,
                virtual_batch_size=128,
            )
        else:
            clf.fit(
                X_tr_32, y_tr_int,
                max_epochs=TABNET_EPOCHS,
                patience=20,
                batch_size=BATCH_SIZE,
                virtual_batch_size=128,
            )

        y_pred        = clf.predict(X_te_32)
        y_prob_2d     = clf.predict_proba(X_te_32)
        y_prob        = y_prob_2d[:, 1]

        assert np.all(y_prob >= 0) and np.all(y_prob <= 1), \
            "TabNet output probabilities outside [0,1]"

        m = evaluate(y_test, y_pred, y_prob)
        seed_results.append(m)
        print_metrics(f"TabNet + {cond_name} (seed={seed})", m)

        tabnet_all_results.append({
            'Model': 'TabNet', 'Condition': cond_key,
            'Condition_Name': cond_name, 'Classifier': 'TabNet',
            'Seed': seed, **m
        })

    f1s  = [m['F1']     for m in seed_results]
    aprs = [m['AUC_PR'] for m in seed_results]
    mccs = [m['MCC']    for m in seed_results]
    print(f"\n=== TabNet + {cond_name} SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")

    print(f"  F1 (binary, fraud class) : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"  AUC-PR                   : {np.mean(aprs)*100:.2f}% ± {np.std(aprs)*100:.2f}%")
    print(f"  MCC                      : {np.mean(mccs)*100:.2f}% ± {np.std(mccs)*100:.2f}%")

# Tabular ResNet
print("MODEL 3: TABULAR RESNET (Supervised)")


resnet_all_results = []

for cond_key, cond_name in supervised_conditions.items():

    print(f"Tabular ResNet — {cond_name}")


    seed_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        X_tr, y_tr = apply_imbalance(X_train, y_train, cond_name, seed)

        assert not np.any(np.isnan(X_tr)), \
            f"SANITY FAIL: NaN in training data after {cond_name}"

        n_fraud_tr = (y_tr==1).sum()
        n_legit_tr = (y_tr==0).sum()
        spw        = n_legit_tr / n_fraud_tr if n_fraud_tr > 0 else 1

        # Prepare tensors
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
        dataset     = TensorDataset(X_tr_tensor, y_tr_tensor)
        loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Build model
        model     = TabularResNet(INPUT_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Loss — weighted BCE for cost-sensitive
        if cond_name == 'Cost-Sensitive':
            pos_weight = torch.tensor([spw], dtype=torch.float32).to(device)
            criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Use raw output for BCEWithLogitsLoss — remove sigmoid from model output
            use_logits = True
        else:
            criterion  = nn.BCELoss()
            use_logits = False

        # Rebuild model without sigmoid for cost-sensitive (BCEWithLogitsLoss)
        if use_logits:
            class TabularResNetLogits(TabularResNet):
                def forward(self, x):
                    h = self.input_proj(x)
                    h = self.relu(self.res_block1(h) + h)
                    h = self.relu(self.res_block2(h) + h)
                    # Return raw logits (no sigmoid) — BCEWithLogitsLoss applies it
                    return self.output[0](h).squeeze(1)
            model     = TabularResNetLogits(INPUT_DIM).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training loop
        model.train()
        for epoch in range(RESNET_EPOCHS):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{RESNET_EPOCHS}  "
                      f"Loss: {epoch_loss/len(loader):.6f}")

        # Evaluate
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_out = model(X_test_tensor)

        if use_logits:
            y_prob = torch.sigmoid(raw_out).cpu().numpy()
        else:
            y_prob = raw_out.cpu().numpy()

        y_pred = (y_prob >= 0.5).astype(int)

        assert np.all(y_prob >= 0) and np.all(y_prob <= 1), \
            "SANITY FAIL: ResNet output probabilities outside [0, 1]"

        m = evaluate(y_test, y_pred, y_prob)
        seed_results.append(m)
        print_metrics(f"ResNet + {cond_name} (seed={seed})", m)

        resnet_all_results.append({
            'Model': 'TabularResNet', 'Condition': cond_key,
            'Condition_Name': cond_name, 'Classifier': 'ResNet',
            'Seed': seed, **m
        })

    f1s  = [m['F1']     for m in seed_results]
    aprs = [m['AUC_PR'] for m in seed_results]
    mccs = [m['MCC']    for m in seed_results]
    print(f"\n=== ResNet + {cond_name} SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")
    print(f"  F1 (binary, fraud class) : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"  AUC-PR                   : {np.mean(aprs)*100:.2f}% ± {np.std(aprs)*100:.2f}%")
    print(f"  MCC                      : {np.mean(mccs)*100:.2f}% ± {np.std(mccs)*100:.2f}%")

#TabPFN 

print("MODEL 4: TABPFN (Supervised, Pre-trained Transformer)")



tabpfn_all_results = []

for seed in SEEDS:
    print(f"\n--- TabPFN seed {seed} ---")
    set_seed(seed)

    # Stratified subsample to TABPFN_MAX_TRAIN_ROWS rows
    rng         = np.random.default_rng(seed)
    fraud_idx   = np.where(y_train == 1)[0]
    legit_idx   = np.where(y_train == 0)[0]

    # Preserve fraud rate in subsample
    fraud_rate   = len(fraud_idx) / len(y_train)
    n_fraud_sub  = max(1, int(TABPFN_MAX_TRAIN_ROWS * fraud_rate))
    n_legit_sub  = TABPFN_MAX_TRAIN_ROWS - n_fraud_sub

    fraud_sample = rng.choice(fraud_idx, size=min(n_fraud_sub, len(fraud_idx)),
                              replace=False)
    legit_sample = rng.choice(legit_idx, size=min(n_legit_sub, len(legit_idx)),
                              replace=False)

    sub_idx = np.concatenate([fraud_sample, legit_sample])
    rng.shuffle(sub_idx)

    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    print(f"Subsampled training set: {len(y_sub):,} rows")
    print(f"Fraud : {(y_sub==1).sum()} ({(y_sub==1).mean()*100:.4f}%)")
    print(f"Legit : {(y_sub==0).sum():,}")


    # TabPFN — no training needed, just fit (loads prior weights)
    clf = TabPFNClassifier(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    clf.fit(X_sub, y_sub)

    y_prob_2d = clf.predict_proba(X_test)
    y_prob    = y_prob_2d[:, 1]
    y_pred    = (y_prob >= 0.5).astype(int)

    assert np.all(y_prob >= 0) and np.all(y_prob <= 1), \
        "TabPFN output probabilities outside [0,1]"

    m = evaluate(y_test, y_pred, y_prob)
    print_metrics(f"TabPFN (seed={seed})", m)

    tabpfn_all_results.append({
        'Model': 'TabPFN', 'Condition': '4G_TabPFN',
        'Condition_Name': 'TabPFN (Pre-trained)',
        'Classifier': 'TabPFN', 'Seed': seed,
        'Train_rows_used': len(y_sub), **m
    })

# TabPFN summary
tabpfn_df = pd.DataFrame(tabpfn_all_results)
f1s  = tabpfn_df['F1'].values
aprs = tabpfn_df['AUC_PR'].values
mccs = tabpfn_df['MCC'].values
print(f"\n=== TabPFN SUMMARY (mean ± std across {len(SEEDS)} seeds) ===")

print(f"  NOTE: Trained on {TABPFN_MAX_TRAIN_ROWS:,}-row stratified subsample per seed")
print(f"  F1 (binary, fraud class) : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
print(f"  AUC-PR                   : {np.mean(aprs)*100:.2f}% ± {np.std(aprs)*100:.2f}%")
print(f"  MCC                      : {np.mean(mccs)*100:.2f}% ± {np.std(mccs)*100:.2f}%")

#Master results 

print("MASTER RESULTS TABLE")


all_results_df = pd.DataFrame(
    vae_all_results + tabnet_all_results + resnet_all_results + tabpfn_all_results
)

master = all_results_df.groupby(['Condition_Name', 'Classifier']).agg(
    F1_mean=('F1','mean'),       F1_std=('F1','std'),
    AUC_PR_mean=('AUC_PR','mean'), AUC_PR_std=('AUC_PR','std'),
    MCC_mean=('MCC','mean'),     MCC_std=('MCC','std'),
    Precision_mean=('Precision','mean'),
    Recall_mean=('Recall','mean'),
    AUC_ROC_mean=('AUC_ROC','mean'),
    FPR_mean=('FPR','mean'),
).reset_index()
master['Model_Label'] = master['Classifier'] + ' (' + master['Condition_Name'] + ')'
master = master.sort_values('F1_mean', ascending=False).reset_index(drop=True)

print("\n=== MASTER RESULTS TABLE — EXPERIMENT 4 DEEP LEARNING ===")
print("  NOTE: F1 is binary (positive class = fraud). Not micro or macro averaged.")
for _, row in master.iterrows():
    print(f"  {row['Model_Label']:<50} "
          f"F1={row['F1_mean']*100:.2f}% ± {row['F1_std']*100:.2f}%  "
          f"AUC-PR={row['AUC_PR_mean']*100:.2f}%  "
          f"MCC={row['MCC_mean']*100:.2f}%")

master_path = os.path.join(RES_DIR, 'master_results_exp4_deep_learning.csv')
master.to_csv(master_path, index=False)
print(f"\nSaved: {master_path}")


print("COMPARISON WITH EXPERIMENT 1 (Classical Baseline)")


exp1_path = os.path.join(RES_DIR, 'master_results_real_data.csv')
if os.path.exists(exp1_path):
    exp1     = pd.read_csv(exp1_path)
    exp1_best = exp1.sort_values('F1_mean', ascending=False).iloc[0]

    print(f"\n  Exp 1 best (classical) : {exp1_best['Model']}")
    print(f"    F1     = {exp1_best['F1_mean']*100:.2f}% ± {exp1_best['F1_std']*100:.2f}%")
    print(f"    AUC-PR = {exp1_best['AUC_PR_mean']*100:.2f}%")

    dl_best = master.iloc[0]
    print(f"\n  Exp 4 best (deep learning) : {dl_best['Model_Label']}")
    print(f"    F1     = {dl_best['F1_mean']*100:.2f}% ± {dl_best['F1_std']*100:.2f}%")
    print(f"    AUC-PR = {dl_best['AUC_PR_mean']*100:.2f}%")

    gap = (exp1_best['F1_mean'] - dl_best['F1_mean']) * 100
    print(f"\n  Gap (Exp 1 best vs DL best): {gap:+.2f}pp")
    if gap > 0:
        print(f"  → Classical models outperform deep learning by {gap:.2f}pp on F1")
    else:
        print(f"  → Deep learning outperforms classical models by {abs(gap):.2f}pp on F1")

#Save raw results ──────────────────────────────────────────────
raw_path = os.path.join(RAW_DIR, 'exp4_deep_learning_raw.csv')
all_results_df.to_csv(raw_path, index=False)
print(f"\nRaw seed results saved: {raw_path}")


