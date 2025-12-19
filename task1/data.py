#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get('ECG_DATA_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training2017')))
CSV_PATH = os.path.join(DATA_DIR, "REFERENCE.csv")

# Directory to save visualization outputs
SAVE_DIR = "./task1_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 300          # Original sampling rate (Hz)
TARGET_LEN = 2400 # Target length (samples) ~8s at 300 Hz

NUM_N_TO_PLOT = 1
NUM_A_TO_PLOT = 1

def load_ecg(record_id, data_dir=DATA_DIR):
    """Load ECG signal from a MATLAB `.mat` file.

    Args:
        record_id (str): Base filename (without extension) of the record.
        data_dir (str): Directory where `.mat` files are located.

    Returns:
        np.ndarray or None: 1-D float32 ECG signal, or None on failure.
    """
    mat_path = os.path.join(data_dir, record_id + ".mat")
    try:
        mat = loadmat(mat_path)
        return mat["val"][0].astype(np.float32)
    except FileNotFoundError:
        print(f"Warning: file not found: {mat_path}")
        return None
    except Exception as e:
        print(f"Error loading {record_id}: {e}")
        return None

def preprocess(sig, target_len=TARGET_LEN):
    """Simple preprocessing for visualization.

    Steps:
    1. Z-score normalization
    2. Center-crop if longer than `target_len` or zero-pad if shorter

    Returns None if input `sig` is None.
    """
    if sig is None:
        return None

    # 1. Z-score normalization
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)

    # 2. Fix length by center-cropping or zero-padding
    current_len = len(sig)
    if current_len >= target_len:
        start = (current_len - target_len) // 2
        sig = sig[start : start + target_len]
    else:
        padding = np.zeros(target_len - current_len, dtype=np.float32)
        sig = np.concatenate([sig, padding])
    return sig


def save_waveform(sig, fs, save_path):
    """Save a time-domain waveform plot to `save_path`.

    The plot uses seconds on the x-axis and normalized amplitude on the y-axis.
    """
    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, sig, lw=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (Normalized)")
    plt.title(os.path.basename(save_path).replace(".png",""))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved waveform: {save_path}")


def save_spectrogram(sig, fs, save_path):
    """Save a spectrogram (time-frequency) image to `save_path`.

    The function restricts the displayed frequency range to 0-80 Hz.
    """
    f, tt, Sxx = signal.spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(tt, f[f <= 80], Sxx_db[f <= 80], shading="gouraud")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(os.path.basename(save_path).replace(".png",""))
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved spectrogram: {save_path}")

def main():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: REFERENCE.csv not found at {CSV_PATH}. Please check the DATA_DIR setting.")
        return

    df_N = df[df["class"] == "N"]
    df_A = df[df["class"] == "A"]

    print(f"Found Normal (N) records: {len(df_N)}")
    print(f"Found AF (A) records:     {len(df_A)}")

    # random selection
    N_ids = df_N["data_source"].sample(min(NUM_N_TO_PLOT, len(df_N)), random_state=42)
    A_ids = df_A["data_source"].sample(min(NUM_A_TO_PLOT, len(df_A)), random_state=42)

    # Normal (N) class
    print("\n--- Processing N (normal) samples ---")
    for rid in N_ids:
        sig_raw = load_ecg(rid)
        sig_processed = preprocess(sig_raw)
        if sig_processed is not None:
            save_waveform(sig_processed, FS, os.path.join(SAVE_DIR, f"N_{rid}_waveform.png"))
            save_spectrogram(sig_processed, FS, os.path.join(SAVE_DIR, f"N_{rid}_spectrogram.png"))

    # Atrial Fibrillation (A) class
    print("\n--- Processing A (AF) samples ---")
    for rid in A_ids:
        sig_raw = load_ecg(rid)
        sig_processed = preprocess(sig_raw)
        if sig_processed is not None:
            save_waveform(sig_processed, FS, os.path.join(SAVE_DIR, f"A_{rid}_waveform.png"))
            save_spectrogram(sig_processed, FS, os.path.join(SAVE_DIR, f"A_{rid}_spectrogram.png"))

    print(f"\nVisualization complete. Images saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()