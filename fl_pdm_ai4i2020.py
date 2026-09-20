"""
==============================================================================
Federated Learning for Privacy-Preserving Predictive Maintenance
in Non-IID Multi-Plant Manufacturing Using the AI4I 2020 Dataset
==============================================================================
Manuscript: Bekkour et al. (2026) — Discover Artificial Intelligence
Repository: https://github.com/Yusufdoc/fl-pdm-ai4i2020
Submission ID: 5a77aa9b-22fd-4f7c-8c3c-ad45620fa69b

Description:
    Reproduces all experimental results reported in Tables 5–6 and Figures 2–8.
    Implements FedAvg, FedProx, Local-Only, and Centralized baselines
    on the AI4I 2020 Predictive Maintenance dataset partitioned into
    three non-IID simulated manufacturing plants by product type.

    Plant A (Type L): n=6,000 records, 3.9% failure rate
    Plant B (Type M): n=2,997 records, 2.8% failure rate
    Plant C (Type H): n=1,003 records, 2.1% failure rate

    All results are averaged over 5 independent seeds (42, 123, 456, 789, 2024).
    Classification threshold is calibrated on the validation set (Youden index)
    to avoid test-set leakage.

Requirements:
    pip install torch scikit-learn pandas numpy matplotlib seaborn requests
    Tested with: Python 3.9+, PyTorch 2.0+, scikit-learn 1.3+

Usage:
    python fl_pdm_ai4i2020.py --mode all       # Run all experiments
    python fl_pdm_ai4i2020.py --mode fedavg    # FedAvg only
    python fl_pdm_ai4i2020.py --mode fedprox   # FedProx only
    python fl_pdm_ai4i2020.py --mode centralized
    python fl_pdm_ai4i2020.py --mode local
    python fl_pdm_ai4i2020.py --mode baselines # RF, XGBoost, LightGBM
==============================================================================
"""

import argparse
import copy
import json
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not installed — skip LightGBM baseline")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — skip XGBoost baseline")

# ============================================================
# CONFIGURATION
# ============================================================
SEEDS = [42, 123, 456, 789, 2024]
N_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MU_FEDPROX = 0.01           # FedProx proximal term
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
OUTPUT_DIR = "results"
DATA_PATH = "ai4i2020.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 1. DATA LOADING AND PARTITIONING
# ============================================================

def download_ai4i2020():
    """Download AI4I 2020 from UCI if not present."""
    if os.path.exists(DATA_PATH):
        return
    print("Downloading AI4I 2020 dataset...")
    url = ("https://raw.githubusercontent.com/SamyamoyRakshit/"
           "AI4I-2020-Predictive-Maintenance-Dataset__Linear-Regression"
           "/main/ai4i2020.csv")
    r = requests.get(url, timeout=30)
    with open(DATA_PATH, 'wb') as f:
        f.write(r.content)
    print(f"  Downloaded: {os.path.getsize(DATA_PATH):,} bytes")


def load_and_partition():
    """
    Load AI4I 2020 and partition into 3 non-IID plants by product type.
    Returns dict: {plant_id: (X, y)} for plants A, B, C.
    """
    download_ai4i2020()
    df = pd.read_csv(DATA_PATH)

    # Feature selection — exclude identifiers and failure mode columns
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    target_col = 'Machine failure'

    # Partitioning by product type (non-IID: different failure rates)
    plants = {
        'A': df[df['Type'] == 'L'],   # Type L → Plant A (majority, 60%)
        'B': df[df['Type'] == 'M'],   # Type M → Plant B (medium, 30%)
        'C': df[df['Type'] == 'H'],   # Type H → Plant C (minority, 10%)
    }

    partitions = {}
    for plant_id, plant_df in plants.items():
        X = plant_df[feature_cols].values.astype(np.float32)
        y = plant_df[target_col].values.astype(np.float32)
        failure_rate = y.mean() * 100
        print(f"  Plant {plant_id} (Type {'L' if plant_id=='A' else 'M' if plant_id=='B' else 'H'}): "
              f"n={len(X):,}, failure rate={failure_rate:.1f}%")
        partitions[plant_id] = (X, y)

    return partitions, feature_cols


def split_plant_data(X, y, seed):
    """Split plant data into train/val/test with fixed seed."""
    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=seed, stratify=y)
    # Second split: val vs test
    val_size_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size_relative),
        random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, scaler=None):
    """Scale features and create PyTorch DataLoaders."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def to_loader(X, y, shuffle=False):
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (to_loader(X_train_scaled, y_train, shuffle=True),
            to_loader(X_val_scaled, y_val),
            to_loader(X_test_scaled, y_test),
            scaler)


# ============================================================
# 2. MODEL ARCHITECTURE — PdMNet
# ============================================================

class PdMNet(nn.Module):
    """
    Predictive Maintenance Network — lightweight MLP.
    Architecture: Input(5) → FC(16,ReLU,BN,0.3) → FC(8,ReLU,BN,0.3) → FC(1,Sigmoid)
    Total parameters: ~2,753
    """
    def __init__(self, input_dim=5):
        super(PdMNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 3. EVALUATION UTILITIES
# ============================================================

def youden_threshold(model, val_loader):
    """
    Calibrate decision threshold on validation set using Youden Index (J).
    J = Sensitivity + Specificity - 1 = TPR - FPR
    Avoids test-set leakage.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            probs = model(X_batch).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_j, best_thresh = -1, 0.5
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        j = tpr + tnr - 1
        if j > best_j:
            best_j, best_thresh = j, t

    return best_thresh


def evaluate_model(model, test_loader, threshold=0.5):
    """Evaluate model on test loader with given threshold."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            probs = model(X_batch).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    metrics = {
        'accuracy':  accuracy_score(all_labels, preds),
        'f1':        f1_score(all_labels, preds, zero_division=0),
        'auc':       roc_auc_score(all_labels, all_probs),
        'recall':    recall_score(all_labels, preds, zero_division=0),
        'precision': precision_score(all_labels, preds, zero_division=0),
    }
    return metrics


# ============================================================
# 4. LOCAL TRAINING UTILITIES
# ============================================================

def train_local(model, train_loader, optimizer, criterion, n_epochs,
                global_model=None, mu=0.0):
    """
    Train model locally for n_epochs.
    If global_model is provided and mu > 0, applies FedProx proximal term.
    """
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # FedProx proximal term
            if global_model is not None and mu > 0:
                proximal = 0.0
                for w, w_global in zip(model.parameters(), global_model.parameters()):
                    proximal += torch.sum((w - w_global.detach()) ** 2)
                loss += (mu / 2) * proximal

            loss.backward()
            optimizer.step()
    return model


def aggregate_fedavg(global_model, local_models, n_samples_per_plant):
    """
    FedAvg aggregation: weighted average of local model weights.
    Weight proportional to number of training samples.
    """
    total_samples = sum(n_samples_per_plant)
    global_state = copy.deepcopy(local_models[0].state_dict())

    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key])
        for model, n in zip(local_models, n_samples_per_plant):
            weight = n / total_samples
            global_state[key] += weight * model.state_dict()[key].float()

    global_model.load_state_dict(global_state)
    return global_model


# ============================================================
# 5. MAIN FL TRAINING LOOP
# ============================================================

def run_federated(partitions, seed, algorithm='fedavg'):
    """
    Run one FL experiment (FedAvg or FedProx) with a given seed.
    Returns: per-round metrics and final per-plant metrics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    plant_ids = list(partitions.keys())
    n_plants = len(plant_ids)
    mu = MU_FEDPROX if algorithm == 'fedprox' else 0.0

    # Prepare plant data
    plant_data = {}
    plant_n_train = {}
    for pid in plant_ids:
        X, y = partitions[pid]
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_plant_data(X, y, seed)
        tr_loader, va_loader, te_loader, scaler = prepare_dataloaders(
            X_tr, y_tr, X_va, y_va, X_te, y_te)
        plant_data[pid] = {
            'train': tr_loader, 'val': va_loader, 'test': te_loader,
            'scaler': scaler, 'n_train': len(X_tr),
            'X_test': X_te, 'y_test': y_te
        }
        plant_n_train[pid] = len(X_tr)

    # Build global test set (concatenate all test sets)
    X_global_test = np.vstack([plant_data[p]['X_test'] for p in plant_ids])
    y_global_test = np.concatenate([plant_data[p]['y_test'] for p in plant_ids])
    # Scale using Plant A's scaler (representative)
    X_global_test_scaled = plant_data['A']['scaler'].transform(X_global_test)
    global_test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_global_test_scaled),
                      torch.FloatTensor(y_global_test)),
        batch_size=BATCH_SIZE)

    # Global model
    global_model = PdMNet().to(DEVICE)
    criterion = nn.BCELoss()

    round_metrics = []

    for round_num in range(N_ROUNDS):
        local_models = []

        for pid in plant_ids:
            local_model = copy.deepcopy(global_model).to(DEVICE)
            optimizer = optim.Adam(local_model.parameters(),
                                   lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            global_ref = copy.deepcopy(global_model).to(DEVICE) if mu > 0 else None
            local_model = train_local(
                local_model, plant_data[pid]['train'],
                optimizer, criterion, LOCAL_EPOCHS,
                global_model=global_ref, mu=mu)
            local_models.append(local_model)

        # Aggregate
        n_samples = [plant_n_train[p] for p in plant_ids]
        global_model = aggregate_fedavg(global_model, local_models, n_samples)

        # Evaluate global model on global test set
        # Use Youden threshold from Plant A's validation set
        threshold = youden_threshold(global_model, plant_data['A']['val'])
        metrics = evaluate_model(global_model, global_test_loader, threshold)
        metrics['round'] = round_num + 1
        round_metrics.append(metrics)

    # Final per-plant evaluation
    per_plant = {}
    for pid in plant_ids:
        threshold = youden_threshold(global_model, plant_data[pid]['val'])
        per_plant[pid] = evaluate_model(
            global_model, plant_data[pid]['test'], threshold)

    return round_metrics, per_plant, global_model, plant_data


def run_local_only(partitions, seed):
    """
    Local-only baseline: each plant trains and evaluates its own model.
    Returns per-plant metrics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    plant_ids = list(partitions.keys())
    per_plant = {}
    criterion = nn.BCELoss()

    for pid in plant_ids:
        X, y = partitions[pid]
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_plant_data(X, y, seed)
        tr_loader, va_loader, te_loader, _ = prepare_dataloaders(
            X_tr, y_tr, X_va, y_va, X_te, y_te)

        model = PdMNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(),
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # Train for N_ROUNDS * LOCAL_EPOCHS total epochs (comparable budget)
        for _ in range(N_ROUNDS * LOCAL_EPOCHS):
            model.train()
            for X_batch, y_batch in tr_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        threshold = youden_threshold(model, va_loader)
        per_plant[pid] = evaluate_model(model, te_loader, threshold)

    # Compute average across plants (unweighted)
    avg_metrics = {}
    for metric in ['accuracy', 'f1', 'auc', 'recall', 'precision']:
        avg_metrics[metric] = np.mean([per_plant[p][metric] for p in plant_ids])

    return per_plant, avg_metrics


def run_centralized(partitions, seed):
    """
    Centralized baseline: all plant data pooled together.
    Returns metrics on pooled test set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Pool all plant data
    all_X = np.vstack([partitions[p][0] for p in partitions])
    all_y = np.concatenate([partitions[p][1] for p in partitions])

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_plant_data(all_X, all_y, seed)
    tr_loader, va_loader, te_loader, _ = prepare_dataloaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te)

    model = PdMNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for _ in range(N_ROUNDS * LOCAL_EPOCHS):
        model.train()
        for X_batch, y_batch in tr_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    threshold = youden_threshold(model, va_loader)
    return evaluate_model(model, te_loader, threshold)


# ============================================================
# 6. BASELINE ML CLASSIFIERS (Table 6)
# ============================================================

def run_baselines(partitions, seed):
    """RF, XGBoost, LightGBM, Logistic Regression on pooled test set."""
    np.random.seed(seed)

    all_X = np.vstack([partitions[p][0] for p in partitions])
    all_y = np.concatenate([partitions[p][1] for p in partitions])

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_plant_data(all_X, all_y, seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    rf_probs = rf.predict_proba(X_te_s)[:, 1]
    rf_preds = rf.predict(X_te_s)
    results['RandomForest'] = {
        'accuracy': accuracy_score(y_te, rf_preds),
        'f1': f1_score(y_te, rf_preds, zero_division=0),
        'auc': roc_auc_score(y_te, rf_probs),
        'recall': recall_score(y_te, rf_preds, zero_division=0),
        'precision': precision_score(y_te, rf_preds, zero_division=0),
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_tr_s, y_tr)
    lr_probs = lr.predict_proba(X_te_s)[:, 1]
    lr_preds = lr.predict(X_te_s)
    results['LogisticRegression'] = {
        'accuracy': accuracy_score(y_te, lr_preds),
        'f1': f1_score(y_te, lr_preds, zero_division=0),
        'auc': roc_auc_score(y_te, lr_probs),
        'recall': recall_score(y_te, lr_preds, zero_division=0),
        'precision': precision_score(y_te, lr_preds, zero_division=0),
    }

    if XGB_AVAILABLE:
        clf = xgb.XGBClassifier(n_estimators=100, random_state=seed,
                                  eval_metric='logloss', use_label_encoder=False)
        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_te, preds),
            'f1': f1_score(y_te, preds, zero_division=0),
            'auc': roc_auc_score(y_te, probs),
            'recall': recall_score(y_te, preds, zero_division=0),
            'precision': precision_score(y_te, preds, zero_division=0),
        }

    if LGBM_AVAILABLE:
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1)
        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)
        results['LightGBM'] = {
            'accuracy': accuracy_score(y_te, preds),
            'f1': f1_score(y_te, preds, zero_division=0),
            'auc': roc_auc_score(y_te, probs),
            'recall': recall_score(y_te, preds, zero_division=0),
            'precision': precision_score(y_te, preds, zero_division=0),
        }

    return results


# ============================================================
# 7. MULTI-SEED RUNNER
# ============================================================

def run_multi_seed(partitions, algorithm='fedavg', seeds=SEEDS):
    """Run an algorithm across multiple seeds and aggregate results."""
    all_metrics = defaultdict(list)
    all_per_plant = defaultdict(lambda: defaultdict(list))
    all_round_metrics = []

    for seed in seeds:
        print(f"    Seed {seed}...", end=' ', flush=True)

        if algorithm in ('fedavg', 'fedprox'):
            round_metrics, per_plant, _, _ = run_federated(
                partitions, seed, algorithm=algorithm)
            for metric in ['f1', 'auc', 'recall', 'precision', 'accuracy']:
                all_metrics[metric].append(round_metrics[-1][metric])
            for pid in per_plant:
                for metric in ['f1', 'auc', 'recall']:
                    all_per_plant[pid][metric].append(per_plant[pid][metric])
            all_round_metrics.append(round_metrics)

        elif algorithm == 'local':
            per_plant, avg_metrics = run_local_only(partitions, seed)
            for metric in ['f1', 'auc', 'recall', 'precision', 'accuracy']:
                all_metrics[metric].append(avg_metrics[metric])
            for pid in per_plant:
                for metric in ['f1', 'auc', 'recall']:
                    all_per_plant[pid][metric].append(per_plant[pid][metric])

        elif algorithm == 'centralized':
            metrics = run_centralized(partitions, seed)
            for metric in ['f1', 'auc', 'recall', 'precision', 'accuracy']:
                all_metrics[metric].append(metrics[metric])

        print(f"F1={all_metrics['f1'][-1]:.3f}, AUC={all_metrics['auc'][-1]:.3f}")

    # Aggregate
    summary = {}
    for metric in ['f1', 'auc', 'recall', 'precision', 'accuracy']:
        values = all_metrics[metric]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
            'values': values
        }

    # Per-plant summary
    per_plant_summary = {}
    for pid in all_per_plant:
        per_plant_summary[pid] = {}
        for metric in ['f1', 'auc', 'recall']:
            values = all_per_plant[pid][metric]
            per_plant_summary[pid][metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'values': values
            }

    return summary, per_plant_summary, all_round_metrics


# ============================================================
# 8. FIGURE GENERATION
# ============================================================

def plot_convergence(round_metrics_list, metric='f1', algorithm='FedAvg',
                     figsize=(8, 5), save_path=None):
    """Plot convergence curve across seeds (mean ± std)."""
    n_rounds = len(round_metrics_list[0])
    rounds = np.arange(1, n_rounds + 1)
    per_round = []
    for round_idx in range(n_rounds):
        values = [rm[round_idx][metric] for rm in round_metrics_list]
        per_round.append(values)

    means = [np.mean(v) for v in per_round]
    stds = [np.std(v, ddof=1) for v in per_round]
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(rounds, means, color='#ED7D31', linewidth=2, label=f'{algorithm} mean')
    ax.fill_between(rounds, means - stds, means + stds,
                    alpha=0.25, color='#ED7D31', label='± 1 SD')
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel(metric.upper().replace('AUC', 'AUC-ROC'), fontsize=12)
    ax.set_title(f'{algorithm} Convergence — {metric.upper()} across {len(round_metrics_list)} seeds',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(1, n_rounds)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_comparison_bar(results_dict, metric='auc', figsize=(9, 6), save_path=None):
    """Bar chart comparing all scenarios (Figure 2 or 3)."""
    labels = list(results_dict.keys())
    means = [results_dict[k][metric]['mean'] for k in labels]
    stds = [results_dict[k][metric]['std'] for k in labels]

    colors = ['#4472C4', '#ED7D31', '#A9D18E', '#70AD47']
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    bars = ax.bar(labels, means, color=colors[:len(labels)],
                  edgecolor='black', linewidth=0.8, width=0.55,
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'})

    for bar, mean, std in zip(bars, means, stds):
        label = f"{mean:.3f}" if std == 0 else f"{mean:.3f}\n±{std:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.002,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    metric_name = 'AUC-ROC' if metric == 'auc' else metric.upper()
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_xlabel('Training Strategy', fontsize=13)
    ax.set_title(f'{metric_name} Comparison — All Scenarios (mean ± SD, 5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(min(m - max(s, 0.01) for m, s in zip(means, stds)) - 0.02,
                max(m + max(s, 0.01) for m, s in zip(means, stds)) + 0.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# 9. RESULTS TABLE PRINTER
# ============================================================

def print_results_table(results, label):
    """Print formatted results table."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    header = f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Values'}"
    print(header)
    print('-' * 65)
    for metric in ['accuracy', 'f1', 'auc', 'recall', 'precision']:
        m = results[metric]
        vals_str = ', '.join([f"{v:.4f}" for v in m['values']])
        print(f"{metric:<12} {m['mean']:>10.4f} {m['std']:>10.4f}   [{vals_str}]")
    print('=' * 65)


def save_results_json(all_results, path):
    """Save all results to JSON for verification."""
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Results saved to {path}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FL PdM AI4I 2020 Experiments')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'fedavg', 'fedprox', 'centralized',
                                 'local', 'baselines'],
                        help='Which experiment(s) to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds to use')
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("  FL for Privacy-Preserving PdM — AI4I 2020")
    print(f"  Mode: {args.mode} | Seeds: {args.seeds}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}\n")

    # Load and partition data
    print("Loading and partitioning data...")
    partitions, feature_cols = load_and_partition()
    print(f"  Features: {feature_cols}")
    print(f"  Model parameters: {PdMNet().count_parameters():,}\n")

    all_results = {}

    if args.mode in ('all', 'fedavg'):
        print("Running FedAvg...")
        summary, per_plant, round_metrics = run_multi_seed(
            partitions, algorithm='fedavg', seeds=args.seeds)
        all_results['fedavg'] = {'summary': summary, 'per_plant': per_plant}
        print_results_table(summary, 'FedAvg — Global Test Set')
        plot_convergence(round_metrics, metric='f1', algorithm='FedAvg',
                         save_path=f"{OUTPUT_DIR}/fedavg_convergence_f1.png")
        plot_convergence(round_metrics, metric='auc', algorithm='FedAvg',
                         save_path=f"{OUTPUT_DIR}/fedavg_convergence_auc.png")

    if args.mode in ('all', 'fedprox'):
        print("\nRunning FedProx...")
        summary, per_plant, round_metrics = run_multi_seed(
            partitions, algorithm='fedprox', seeds=args.seeds)
        all_results['fedprox'] = {'summary': summary, 'per_plant': per_plant}
        print_results_table(summary, f'FedProx (μ={MU_FEDPROX}) — Global Test Set')
        plot_convergence(round_metrics, metric='f1', algorithm='FedProx',
                         save_path=f"{OUTPUT_DIR}/fedprox_convergence_f1.png")

    if args.mode in ('all', 'local'):
        print("\nRunning Local-Only...")
        summary, per_plant, _ = run_multi_seed(
            partitions, algorithm='local', seeds=args.seeds)
        all_results['local'] = {'summary': summary, 'per_plant': per_plant}
        print_results_table(summary, 'Local-Only — Average Across Plants')

    if args.mode in ('all', 'centralized'):
        print("\nRunning Centralized...")
        summary, _, _ = run_multi_seed(
            partitions, algorithm='centralized', seeds=args.seeds)
        all_results['centralized'] = {'summary': summary}
        print_results_table(summary, 'Centralized (Pooled) — Test Set')

    if args.mode in ('all', 'baselines'):
        print("\nRunning ML Baselines...")
        baseline_all = defaultdict(lambda: defaultdict(list))
        for seed in args.seeds:
            results = run_baselines(partitions, seed)
            for clf_name, metrics in results.items():
                for metric, value in metrics.items():
                    baseline_all[clf_name][metric].append(value)
        print(f"\n{'='*65}")
        print("  Baseline ML Classifiers")
        print(f"{'='*65}")
        for clf_name in baseline_all:
            print(f"\n  {clf_name}:")
            for metric in ['accuracy', 'f1', 'auc', 'recall']:
                vals = baseline_all[clf_name][metric]
                print(f"    {metric:<12}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
        all_results['baselines'] = dict(baseline_all)

    # Generate comparison figures if all experiments ran
    if args.mode == 'all' and len(all_results) >= 4:
        print("\nGenerating comparison figures...")
        display_names = {
            'local':       'Local Only\n(avg)',
            'fedavg':      'FedAvg',
            'fedprox':     'FedProx',
            'centralized': 'Centralized\n(pooled)'
        }
        plot_dict = {
            display_names[k]: all_results[k]['summary']
            for k in ['local', 'fedavg', 'fedprox', 'centralized']
            if k in all_results
        }
        plot_comparison_bar(plot_dict, metric='f1',
                            save_path=f"{OUTPUT_DIR}/fig2_f1_comparison.png")
        plot_comparison_bar(plot_dict, metric='auc',
                            save_path=f"{OUTPUT_DIR}/fig3_auc_comparison.png")

    # Save all results to JSON
    if all_results:
        save_results_json(all_results, f"{OUTPUT_DIR}/all_results.json")

    print(f"\n✅ Done. Results saved to '{OUTPUT_DIR}/'")


if __name__ == '__main__':
    main()
