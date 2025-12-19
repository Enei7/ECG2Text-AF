
---

# 🫀 ECG Atrial Fibrillation Detection with MS-CNN (Task 1)

This project investigates **atrial fibrillation (AF) detection from ECG signals** using a **Multi-Scale Convolutional Neural Network (MS-CNN)**.
The goal is to classify ECG recordings into **Normal rhythm (N)** or **Atrial Fibrillation (AF / A)** and systematically analyze the effects of **multi-scale convolution, hyperparameters, and data augmentation**.

This project is developed as **Task 1** of the *Modern Signal Processing* course at **SUSTech**.

---

## 📌 Project Highlights

* Binary ECG classification: **Normal (N) vs. AF (A)**
* **Single-stream vs. multi-stream (MS-CNN)** comparison
* Quantitative analysis of **convolution kernel sizes**
* Hyperparameter study (**learning rate, epochs**)
* Controlled **data augmentation** for minority class (AF)
* Comprehensive evaluation with **F1-score, AUC, ROC**

---

## 📂 Project Structure

```text
task1/
├── data.py                     # ECG preprocessing & basic visualization
├── visualization.py            # Time & frequency-domain visualization of augmentations
├── dataset.py                  # PyTorch Dataset (with augmentation support)
├── model.py                    # MS-CNN model definition
├── train.py                    # Training & evaluation script
├── README.md                   # Project documentation
├── task1_figs/                 # (ignored) generated figures
├── task1_visualizations_augmented/  # (ignored) augmentation visualizations
└── results/                    # (ignored) checkpoints, ROC curves, logs
```

> **Note**:
>
> * `results/` and visualization outputs are intentionally excluded from version control.
> * The ECG dataset is **not uploaded** due to size and license considerations.

---

## ⚙️ Environment Setup

### Dependencies

```bash
pip install torch torchvision numpy pandas scipy matplotlib scikit-learn tqdm
```

---

## 📊 Dataset

This project uses the **PhysioNet / CinC Challenge 2017 ECG dataset**.

Expected directory structure:

```text
training2017/
├── REFERENCE.csv
├── A00001.mat
├── A00002.mat
└── ...
```

Only two classes are used:

* **N** → Normal rhythm
* **A** → Atrial fibrillation

Other classes (O, ~) are excluded in Task 1.

---

## 🔄 Data Processing Pipeline

Each ECG signal undergoes the following steps:

1. **Loading** from `.mat` files
2. **Z-score normalization**
3. **Length unification (2400 samples)**

   * Training: random cropping (implicit temporal shift)
   * Validation/Test: center cropping
   * Short signals: zero padding
4. **Optional data augmentation (training only, AF class only)**

   * Gaussian noise
   * Temporal scaling (stretch/compress)
   * Random cropping (shift effect)

---

## 🧠 Model Architecture

### MS-CNN (Multi-Scale CNN)

The MS-CNN consists of **two parallel convolutional streams**:

* **Stream 1**: small receptive field (kernel size = 3)
* **Stream 2**: large receptive field (kernel size = 5 / 7 / 9)

Features from both streams are concatenated and passed to an MLP classifier.

```
Input [B, 1, 2400]
├── Stream 1 (k=3)
│   └── Conv Blocks + Pooling
├── Stream 2 (k=5/7/9)
│   └── Conv Blocks + Pooling
└── Feature Concatenation
    → Global Average Pooling
    → MLP (1024 → 1024 → 256 → 1)
```

A **single-stream CNN** (k=3 only) is also implemented as a baseline.

---

## 🧪 Experimental Design

### Stage 1: Effect of Multi-Scale Kernels

**Objective**: Evaluate the impact of different kernel sizes in Stream 2.

Models:

* Single-stream baseline
* MS-CNN (3,5), MS-CNN (3,7), MS-CNN (3,9)

```bash
python train.py --single_stream --no_aug --lr 1e-2 --epochs 50
python train.py --k_size_stream2 5 --no_aug --lr 1e-2 --epochs 50
python train.py --k_size_stream2 7 --no_aug --lr 1e-2 --epochs 50
python train.py --k_size_stream2 9 --no_aug --lr 1e-2 --epochs 50
```

---

### Stage 2: Hyperparameter Study

**Objective**: Analyze training stability under different learning rates.

```bash
python train.py --k_size_stream2 5 --no_aug --lr 1e-2 --epochs 50
python train.py --k_size_stream2 5 --no_aug --lr 1e-3 --epochs 50
python train.py --k_size_stream2 5 --no_aug --lr 1e-4 --epochs 50
```

---

### Stage 3: Data Augmentation Study

**Objective**: Evaluate generalization improvement via augmentation.

Augmentation is applied **only to AF samples** with a given probability.

```bash
python train.py --k_size_stream2 7 --aug_mode noise --aug_prob 0.3
python train.py --k_size_stream2 7 --aug_mode scale --aug_prob 0.3
python train.py --k_size_stream2 7 --aug_mode all   --aug_prob 0.3
```

---

## 📈 Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**
* **F1-score** *(primary metric for AF detection)*
* **ROC curve & AUC**

---

## 🚀 Quick Start

```bash
# Recommended baseline
python train.py --k_size_stream2 7 --no_aug --lr 1e-2 --epochs 50

# With data augmentation
python train.py --k_size_stream2 7 --aug_mode all --aug_prob 0.2 --lr 1e-2 --epochs 50
```

---

## 📄 License

MIT License

---
