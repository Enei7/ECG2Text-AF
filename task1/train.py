#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from task1.dataset import ECGDataset
from task1.model import MSCNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Default to a repo-relative training2017 folder, allow override by env var
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.environ.get('ECG_DATA_DIR', os.path.join(BASE_DIR, 'training2017'))
CSV_PATH = os.path.join(DATA_DIR, "REFERENCE.csv")
# Results directory can be overridden by ECG_RESULTS_DIR, default inside repo
RESULTS_DIR = os.environ.get('ECG_RESULTS_DIR', os.path.join(BASE_DIR, 'task1', 'results'))

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_LEN = 2400  # 2400 samples (~8s @ 300Hz)
BATCH_SIZE = 64
EPOCHS = 50  # default epochs
LEARNING_RATE = 5e-5
TEST_SPLIT_SIZE = 0.15  # 15% test set
VAL_SPLIT_SIZE = 0.1  # validation fraction relative to remaining data
RANDOM_STATE = 42

OVERSAMPLE_FACTOR = 2


DEFAULT_AUG_MODE = 'all'  # 'none', 'noise', 'scale', 'shift', 'all'
DEFAULT_AUG_PROB = 0.3   

def load_and_split_data(csv_path):

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Stage 1.1: keep only N and A classes
    df_filtered = df[df["class"].isin(["N", "A"])].copy()
    
    # 将 'N' 映射为 0, 'A' 映射为 1
    df_filtered["label"] = df_filtered["class"].map({"N": 0, "A": 1})
    
    file_ids = df_filtered["data_source"].values
    labels = df_filtered["label"].values
    
    print(f"Total samples (N+A): {len(labels)}")
    print(f"A class (AF) count: {np.sum(labels)}")
    
    # First split: train+val vs test
    ids_train_val, ids_test, lbl_train_val, lbl_test = train_test_split(
        file_ids, labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels  # preserve N/A ratio in test set
    )
    
    # Second split: train vs validation (relative to remaining data)
    # Note: relative_val_size is the fraction of the remaining (1 - TEST_SPLIT_SIZE)
    relative_val_size = VAL_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
    
    ids_train, ids_val, lbl_train, lbl_val = train_test_split(
        ids_train_val, lbl_train_val,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=lbl_train_val  # preserve N/A ratio in validation set
    )
    
    print(f"Train set size: {len(ids_train)} (A class: {np.sum(lbl_train)})")
    print(f"Validation set size: {len(ids_val)} (A class: {np.sum(lbl_val)})")
    print(f"Test set size: {len(ids_test)} (A class: {np.sum(lbl_test)})")
    
    return (ids_train, lbl_train), (ids_val, lbl_val), (ids_test, lbl_test)

def create_dataloaders(train_data, val_data, test_data, aug_mode='all', aug_prob=0.3):
    """Create DataLoaders and apply simple oversampling (train only)."""
    (ids_train, lbl_train) = train_data
    (ids_val, lbl_val) = val_data
    (ids_test, lbl_test) = test_data

    # --- Simple oversampling for the training set ---
    print(f"Original training A (AF) count: {np.sum(lbl_train)}")
    
    # find indices of A class (minority)
    af_indices = np.where(lbl_train == 1)[0]
    
    # duplicate A-class file_ids and labels (simple oversampling)
    ids_train_oversampled = list(ids_train)
    lbl_train_oversampled = list(lbl_train)
    
    for _ in range(OVERSAMPLE_FACTOR):
        ids_train_oversampled.extend(ids_train[af_indices])
        lbl_train_oversampled.extend(lbl_train[af_indices])
        
    print(f"Oversampled training set size: {len(ids_train_oversampled)}")
    print(f"Oversampled A (AF) count: {np.sum(lbl_train_oversampled)}")
    
    # compute pos_weight for BCEWithLogitsLoss
    n_pos = np.sum(lbl_train_oversampled)
    n_neg = len(lbl_train_oversampled) - n_pos
    pos_weight_value = n_neg / n_pos if n_pos > 0 else 1.0
    # --- oversampling end ---

    # create Datasets (enable augmentation for training)
    train_dataset = ECGDataset(DATA_DIR, ids_train_oversampled, lbl_train_oversampled, 
                               target_len=TARGET_LEN, is_train=True, 
                               aug_mode=aug_mode, aug_prob=aug_prob)
    val_dataset = ECGDataset(DATA_DIR, ids_val, lbl_val, target_len=TARGET_LEN, is_train=False)
    test_dataset = ECGDataset(DATA_DIR, ids_test, lbl_test, target_len=TARGET_LEN, is_train=False)

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, pos_weight_value

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        
        # filter out failed loads (label == -1)
        valid_indices = (y != -1)
        if not valid_indices.any():
            continue
        X, y = X[valid_indices], y[valid_indices]
        
        if X.shape[0] <= 1:
            print("\nWarning: batch size <= 1, skipping this batch to avoid BatchNorm errors.")
            continue
        
        optimizer.zero_grad()
        logits = model(X) # [B, 1]

        loss = criterion(logits.squeeze(), y.float()) 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, find_best_threshold=False):
    """Evaluate the model on validation or test set.

    Returns: avg_loss, accuracy, f1, auc, best_threshold
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            
            valid_indices = (y != -1)
            if not valid_indices.any():
                continue
            X, y = X[valid_indices], y[valid_indices]

            logits = model(X) # [B, 1]
            loss = criterion(logits.squeeze(), y.float())
            total_loss += loss.item()
            
            # compute probabilities
            probs = torch.sigmoid(logits).squeeze() # [B]
            
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # find best threshold (based on F1)
    best_threshold = 0.5
    if find_best_threshold and len(np.unique(all_labels)) > 1:
        thresholds = np.arange(0.05, 0.95, 0.05)
        best_f1 = 0
        for thresh in thresholds:
            preds_temp = (np.array(all_probs) > thresh).astype(int)
            f1_temp = f1_score(all_labels, preds_temp, pos_label=1, zero_division=0)
            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = thresh
        print(f"  Best threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    
    # apply threshold to get predictions
    all_preds = (np.array(all_probs) > best_threshold).astype(int)

    # compute metrics
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0) 
    
    # ensure A class present in evaluation set for AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5
        
    return avg_loss, acc, f1, auc, best_threshold

def main(args):
    # set random seed for reproducibility
    import sys
    config = getattr(sys, '_training_config', {
        'seed': RANDOM_STATE, 
        'aug_mode': DEFAULT_AUG_MODE, 
        'aug_prob': DEFAULT_AUG_PROB,
        'epochs': EPOCHS,
        'lr': LEARNING_RATE
    })
    set_seed(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {config['seed']}")

    # 1. load and split data
    train_data, val_data, test_data = load_and_split_data(CSV_PATH)
    
    # 2. create DataLoaders (with oversampling)
    train_loader, val_loader, test_loader, pos_weight_value = create_dataloaders(
        train_data, val_data, test_data, 
        aug_mode=config['aug_mode'], 
        aug_prob=config['aug_prob']
    )

    # 3. initialize model (single-stream or dual-stream)
    print("\n--- Experiment configuration ---")
    if args.single_stream:
        print("Model: single-stream (Baseline)")
        model = MSCNN(in_channels=1, out_channels=1, use_single_stream=True).to(device)
    else:
        print(f"Model: dual-stream (MS-CNN)")
        print(f"Stream 1 kernel: 3")
        print(f"Stream 2 kernel: {args.k_size_stream2}")
        model = MSCNN(in_channels=1, out_channels=1, use_single_stream=False, k_size_stream2=args.k_size_stream2).to(device)
    print("--------------------")
    
    # 4. define loss and optimizer
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"\nUsing pos_weight for class balancing: {pos_weight_value:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # 自动处理 sigmoid + BCELoss
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # LR scheduler: reduce LR when validation F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                       patience=10, verbose=True, min_lr=1e-7)

    # 5. training loop
    best_val_f1 = 0
    best_model_path = os.path.join(RESULTS_DIR, "best.pth")
    
    total_epochs = config['epochs']

    for epoch in range(total_epochs):
        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_auc, _ = evaluate(model, val_loader, criterion, device, find_best_threshold=True)
        
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1 (A): {val_f1:.4f}, Val AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            print(f"  Validation F1 improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model...")
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    # 6. final evaluation on test set
    print("\n--- Training finished, evaluating best model on test set ---")
    # load best model
    model.load_state_dict(torch.load(best_model_path))
    
    test_loss, test_acc, test_f1, test_auc, best_test_threshold = evaluate(model, test_loader, criterion, device, find_best_threshold=True)
    
    print("\n--- Final test results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score (A): {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Best test threshold: {best_test_threshold:.3f}")

    # compute confusion matrix
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            valid_indices = (y != -1)
            if not valid_indices.any(): continue
            X, y = X[valid_indices], y[valid_indices]
            
            probs = torch.sigmoid(model(X)).squeeze()
            preds = (probs > best_test_threshold).float()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nTest set Confusion Matrix:")
    print(cm)
    print("        (Pred: N) (Pred: A)")
    print(f"True N: {cm[0][0]:<8} {cm[0][1]:<8}")
    print(f"True A: {cm[1][0]:<8} {cm[1][1]:<8}")

    # plot ROC curve
    model.eval()
    all_labels_roc = []
    all_probs_roc = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            valid_indices = (y != -1)
            if not valid_indices.any(): continue
            X, y = X[valid_indices], y[valid_indices]
            
            probs = torch.sigmoid(model(X)).squeeze()
            all_labels_roc.extend(y.cpu().numpy())
            all_probs_roc.extend(probs.cpu().numpy())
    
    fpr, tpr, thresholds = roc_curve(all_labels_roc, all_probs_roc)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # save ROC curve
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "single_stream" if args.single_stream else f"dual_stream_k{args.k_size_stream2}"
    roc_path = os.path.join(RESULTS_DIR, f"roc_curve_{model_type}_{timestamp}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {roc_path}")
    plt.close()
    
    # save test results to file
    results_txt_path = os.path.join(RESULTS_DIR, f"test_results_{model_type}_{timestamp}.txt")
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Test set evaluation results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Experiment configuration:\n")
        f.write(f"  Model type: {'single-stream (Baseline)' if args.single_stream else 'dual-stream (MS-CNN)'}\n")
        if not args.single_stream:
            f.write(f"  Stream 2 kernel size: {args.k_size_stream2}\n")
        f.write(f"  Epochs: {epoch+1}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Learning rate: {config['lr']}\n")
        f.write(f"  Oversample factor: {OVERSAMPLE_FACTOR}\n")
        f.write(f"  Augmentation: {config['aug_mode']} (prob={config['aug_prob']})\n\n")

        f.write(f"Dataset splits:\n")
        f.write(f"  Train size: {len(train_data[0])}\n")
        f.write(f"  Validation size: {len(val_data[0])}\n")
        f.write(f"  Test size: {len(test_data[0])}\n\n")

        f.write(f"Test set metrics:\n")
        f.write(f"  Loss: {test_loss:.4f}\n")
        f.write(f"  Accuracy: {test_acc:.4f}\n")
        f.write(f"  F1-Score (A): {test_f1:.4f}\n")
        f.write(f"  AUC: {test_auc:.4f}\n")
        f.write(f"  Best threshold: {best_test_threshold:.3f}\n\n")

        f.write(f"Confusion matrix:\n")
        f.write(f"           Pred N    Pred A\n")
        f.write(f"  True N:   {cm[0][0]:<8} {cm[0][1]:<8}\n")
        f.write(f"  True A:   {cm[1][0]:<8} {cm[1][1]:<8}\n\n")

        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        f.write(f"Detailed metrics:\n")
        f.write(f"  Sensitivity (Recall): {sensitivity:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"\nModel path: {best_model_path}\n")
        f.write(f"ROC path: {roc_path}\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Test results saved to: {results_txt_path}")
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: MS-CNN training script")

    # Model params
    parser.add_argument('--single_stream', action='store_true', help='Use single-stream model (baseline)')
    parser.add_argument('--k_size_stream2', type=int, default=7, choices=[5, 7, 9], help='Kernel size for Stream 2 (default: 7)')

    # Augmentation params
    parser.add_argument('--aug_mode', type=str, default=DEFAULT_AUG_MODE, 
                       choices=['none', 'noise', 'scale', 'shift', 'all'],
                       help='Augmentation mode (default: all)')
    parser.add_argument('--aug_prob', type=float, default=DEFAULT_AUG_PROB,
                       help='Augmentation probability (default: 0.3)')

    # Training params
    parser.add_argument('--seed', type=int, default=RANDOM_STATE, help='Random seed (default: 42)')
    parser.add_argument('--no_aug', action='store_true', help='Disable augmentation')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE})')
    
    args = parser.parse_args()
    
    # update training configuration from CLI
    random_state = args.seed
    epochs = args.epochs
    learning_rate = args.lr
    if args.no_aug:
        aug_mode = 'none'
        aug_prob = 0.0
    else:
        aug_mode = args.aug_mode
        aug_prob = args.aug_prob
    
    print(f"\n{'='*60}")
    print(f"Training configuration:")
    print(f"  Random seed: {random_state}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Augmentation mode: {aug_mode}")
    print(f"  Augmentation probability: {aug_prob}")
    print(f"{'='*60}\n")
    
    import sys
    sys._training_config = {
        'seed': random_state,
        'aug_mode': aug_mode,
        'aug_prob': aug_prob,
        'epochs': epochs,
        'lr': learning_rate
    }
    
    main(args)