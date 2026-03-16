# FetalDiag: UniMatch Implementation for ISBS2026 Fetal Ultrasound Analysis Challenge

This repository contains a solution using semi-supervised learning (UniMatch) for the **ISBS2026 Fetal Ultrasound Analysis Challenge**. It is designed for the segmentation and classification of fetal ultrasound images, supporting multi-view analysis including 4CH, LVOT, RVOT, and 3VT views.

## 📋 Project Structure

The pipeline is organized into four sequential steps located in the `FetalDiag` directory:

| Script | Description |
| :--- | :--- |
| `step_0_split_train_valid_fold.py` | Generates train/validation splits (JSON) using stratified sampling based on views and multi-labels. |
| `step_1_unimatch_train.py` | Main training script using the UniMatch semi-supervised learning framework. |
| `step_2_inference.py` | Runs inference on a dataset using a trained checkpoint. |
| `step_3_evaluate.py` | Evaluates predictions against Ground Truth (GT) using Dice, NSD, and F1 scores. |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Wenying-Li/FetalDiag.git
   cd FetalDiag/FetalDiag
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (Python 3.8+).
   ```bash
   pip install -r requirements.txt
   ```
   *Key dependencies include: `torch`, `monai`, `numpy`, `h5py`, `scikit-learn`, `tensorboard`.*

## 🚀 Usage

### 1. Data Preparation (Step 0)
Organize your `.h5` data into `images` and `labels` directories under a root folder (e.g., `data/`). Then generate the split JSON files.

```bash
python step_0_split_train_valid_fold.py \
  --root data \
  --valid_ratio 0.2 \
  --seed 2026
```
*Outputs: `train_labeled.json`, `train_unlabeled.json`, `valid.json`*

### 2. Training (Step 1)
Train the model using the generated JSON files. You can choose between `unet` or `echocare` architectures.

```bash
python step_1_unimatch_train.py \
  --model echocare \
  --batch-size 8 \
  --train-epochs 300 \
  --save-path ./checkpoints_echocare \
  --gpu 0
```
**Key Arguments:**
- `--model`: `unet` or `echocare`
- `--cls_only`: Train only the classification head (freezes encoder/seg).
- `--seg-only-epochs`: Number of initial epochs to train only segmentation.
- `--small-sample`: Use a subset of unlabeled data for quick debugging.

### 3. Inference (Step 2)
Generate predictions for a dataset (e.g., validation set).

```bash
python step_2_inference.py \
  --data-json ./data/valid.json \
  --ckpt ./checkpoints_echocare/best.pth \
  --out-dir ./output \
  --mask-mode oracle
```
**Key Arguments:**
- `--mask-mode`: `oracle` (uses known view ID to mask logits) or `none`.
- `--cls-thr`: Global classification threshold (default: 0.5).
- `--smooth-ckpt-thr`: Smooths checkpoint thresholds to prevent overfitting.

### 4. Evaluation (Step 3)
Compute metrics (Dice, NSD, Macro-F1) by comparing predictions with ground truth.

```bash
python step_3_evaluate.py \
  --valid-json ./data/valid.json \
  --pred-dir ./output \
  --save-dir ./eval_results
```

## 📊 Metrics
The evaluation script reports:
- **Segmentation**: Mean Dice, Mean NSD (Normalized Surface Dice).
- **Classification**: Macro-F1 (Masked), per-class Precision/Recall.
- **Confusion Matrices**: Per-class confusion matrices are logged during training and evaluation.

## ⚙️ Configuration
The system uses default mappings for anatomical views and classes:
- **Views**: 4CH (0), LVOT (1), RVOT (2), 3VT (3).
- **Allowed Maps**: JSON mappings define which classes are valid for which views (see `DEFAULT_SEG_ALLOWED` in scripts).

