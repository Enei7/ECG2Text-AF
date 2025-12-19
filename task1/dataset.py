#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class ECGDataset(Dataset):
    """PyTorch Dataset for single-lead ECG records.

    This Dataset reads `.mat` files containing a 1-D ECG waveform (`val` variable),
    applies Z-score normalization, length normalization (pad/crop) and optional
    augmentations (time-scale and additive noise) for training.

    Args:
        data_dir (str): Directory containing .mat files.
        file_ids (list): List/array of record ids (filenames without '.mat').
        labels (list): Corresponding labels (0/1 or -1 for invalid).
        target_len (int): Target length in samples (default 2400).
        is_train (bool): Whether dataset is used for training (enables random crop).
        aug_mode (str): 'none'|'noise'|'scale'|'all' to control augmentation.
        aug_prob (float): Probability of applying augmentation to eligible samples.
        noise_factor (float): Standard deviation for additive Gaussian noise.
        max_shift_pct (float): (unused) reserved for potential shift augmentation.
        scale_range (tuple): (min, max) scaling factors for time-scaling.
    """

    def __init__(self, data_dir, file_ids, labels, 
                 target_len=2400, 
                 is_train=True, 
                 aug_mode='none', 
                 aug_prob=0.3,
                 noise_factor=0.05, 
                 max_shift_pct=0.1, 
                 scale_range=(0.9, 1.1)):

        self.data_dir = data_dir
        self.file_ids = file_ids
        self.labels = labels
        self.target_len = target_len
        self.is_train = is_train

        self.aug_mode = aug_mode
        self.aug_prob = aug_prob
        self.noise_factor = noise_factor
        self.max_shift_pct = max_shift_pct
        self.scale_range = scale_range

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        # 1. Get record id and label
        record_id = self.file_ids[index]
        label = self.labels[index]

        # 2. Load the .mat file
        mat_path = os.path.join(self.data_dir, record_id + ".mat")
        try:
            mat = loadmat(mat_path)
            sig = mat["val"][0].astype(np.float32)
        except Exception as e:
            print(f"Error: failed to load {mat_path}: {e}")
            sig = np.zeros(self.target_len, dtype=np.float32)
            label = -1

        # 3. Preprocessing and augmentation
        # Augmentation applies only during training and typically only to the
        # minority/positive class (label == 1) by default.
        do_augment = self.is_train and label == 1 and random.random() < self.aug_prob

        # 3a: Time-scaling (apply before cropping)
        if do_augment and (self.aug_mode == 'scale' or self.aug_mode == 'all'):
            sig = self._apply_scale(sig)

        # 3b: Z-score normalization
        sig = self._normalize(sig)

        # 3c: Fix length (random crop during training provides shift augmentation)
        allow_random_crop = self.is_train
        sig = self._pad_crop(sig, random_crop=allow_random_crop)

        # 3d: Additive noise (after normalization and length adjustment)
        if do_augment and (self.aug_mode == 'noise' or self.aug_mode == 'all'):
            sig = self._apply_noise(sig)
            
        # 4. Transform to Tensor
        sig_tensor = torch.from_numpy(sig.copy()).unsqueeze(0) # [1, target_len]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return sig_tensor, label_tensor

    def _normalize(self, sig):
        std = sig.std()
        if std > 1e-6:
            sig = (sig - sig.mean()) / std
        else:
            sig = sig - sig.mean()
        return sig

    def _pad_crop(self, sig, random_crop=False):
        current_len = len(sig)
        if current_len == self.target_len:
            return sig
        
        if current_len > self.target_len:
            if random_crop:
                # Training: random crop (shift augmentation)
                start = random.randint(0, current_len - self.target_len)
            else:
                # Validation/Test: center crop
                start = (current_len - self.target_len) // 2
            sig = sig[start : start + self.target_len]
        else:
            # zero fill
            padding = np.zeros(self.target_len - current_len, dtype=np.float32)
            sig = np.concatenate([sig, padding])
        return sig

    def _apply_noise(self, sig):
        noise = np.random.randn(len(sig)) * self.noise_factor
        return (sig + noise).astype(np.float32)

    def _apply_scale(self, sig):
        factor = random.uniform(self.scale_range[0], self.scale_range[1])
        old_len = len(sig)
        new_len = int(old_len * factor)
        
        if new_len == old_len:
            return sig
            
        old_x = np.linspace(0, 1, old_len)
        new_x = np.linspace(0, 1, new_len)
        
        sig = np.interp(new_x, old_x, sig).astype(np.float32)
        return sig