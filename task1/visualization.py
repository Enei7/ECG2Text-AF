import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from dataset import ECGDataset

# Use an environment variable `ECG_DATA_DIR` if present; otherwise assume a
# `training2017/` folder one level above this package directory.
DEFAULT_DATA_DIR = os.environ.get(
    "ECG_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training2017")),
)
DATA_DIR = DEFAULT_DATA_DIR
CSV_PATH = os.path.join(DATA_DIR, "REFERENCE.csv")

# Output directory (can be overridden via environment variable)
SAVE_DIR = os.environ.get(
    "ECG_VIS_SAVE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "task1", "task1_visualizations_augmented")),
)
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_LEN = 2400  # samples per signal
FS = 300  # sampling frequency (Hz)

def get_a_sample_id():
    """Randomly pick an A-class (AF) record id from REFERENCE.csv."""
    df = pd.read_csv(CSV_PATH)
    df_A = df[df["class"] == "A"]
    return df_A["data_source"].sample(1).iloc[0]

def plot_signal_and_spectrum(signal, title, ax_time, ax_freq, color='C0'):
    """Plot time-domain waveform and its magnitude spectrum (FFT).

    Frequency plot is limited to 0-40 Hz for clarity.
    """
    samples = np.arange(len(signal))

    # Time domain
    ax_time.plot(samples, signal, lw=0.8, color=color)
    ax_time.set_title(f"{title} - Time Domain")
    ax_time.set_xlabel("Sample Index (N)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, linestyle='--', alpha=0.5)

    # Frequency domain (FFT)
    N = len(signal)
    signal_fft = fft(signal)
    freqs = fftfreq(N, 1 / FS)
    signal_fft = np.abs(signal_fft)[: N // 2]
    freqs = freqs[: N // 2]

    ax_freq.plot(freqs, signal_fft, lw=0.8, color=color)
    ax_freq.set_title(f"{title} - Frequency Domain")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.grid(True, linestyle='--', alpha=0.5)

    ax_freq.set_xlim(0, 40)

def main():
    # 1. Randomly select an A-class sample
    record_id = get_a_sample_id()
    print(f"Visualizing AF (A-class) sample: {record_id}")

    # 2. Prepare 4 Dataset instances representing different augmentation modes
    common_params = {
        "data_dir": DATA_DIR,
        "file_ids": [record_id],
        "labels": [1],  # A-class label
        "target_len": TARGET_LEN,
        "is_train": True,  # training mode (enables random crop)
        "aug_prob": 1.0   # 100% probability to trigger augmentation
    }

    # augmentation modes
    dataset_orig = ECGDataset(**common_params, aug_mode='none')
    dataset_noise = ECGDataset(**common_params, aug_mode='noise')
    dataset_scale = ECGDataset(**common_params, aug_mode='scale')
    dataset_all = ECGDataset(**common_params, aug_mode='all')

    # 3. Get a sample from each Dataset (index 0)
    sig_orig, _ = dataset_orig[0]
    sig_orig = sig_orig.squeeze(0).numpy()
    
    sig_noise, _ = dataset_noise[0]
    sig_noise = sig_noise.squeeze(0).numpy()
    
    sig_scale, _ = dataset_scale[0]
    sig_scale = sig_scale.squeeze(0).numpy()
    
    sig_all, _ = dataset_all[0]
    sig_all = sig_all.squeeze(0).numpy()

    # 4. Plot time-domain and frequency-domain comparisons for each augmentation
    # Figure 1: Original sample
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_signal_and_spectrum(sig_orig, "Original Sample", axs[0], axs[1], color='C0')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"A_{record_id}_original.png"), dpi=150)
    plt.close()

    # Figure 2: Noise augmentation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_signal_and_spectrum(sig_noise, "Augmentation: Noise", axs[0], axs[1], color='C1')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"A_{record_id}_noise_augmentation.png"), dpi=150)
    plt.close()

    # Figure 3: Scale/shift augmentation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_signal_and_spectrum(sig_scale, "Augmentation: Scale", axs[0], axs[1], color='C2')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"A_{record_id}_scale_augmentation.png"), dpi=150)
    plt.close()

    # Figure 4: Noise + Scale augmentation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_signal_and_spectrum(sig_all, "Augmentation: Noise + Scale", axs[0], axs[1], color='C3')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"A_{record_id}_noise_scale_augmentation.png"), dpi=150)
    plt.close()

    print(f"Visualizations saved in {SAVE_DIR}") 

if __name__ == "__main__":
    main()
