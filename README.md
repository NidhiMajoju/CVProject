# Seeing Through the Storm
### Synthetic Weather Augmentation for Robust Road Scene Segmentation

Rishna Renikunta · Nidhi Majoju  
CS Project — University of Texas at Dallas

---

## Overview

Autonomous driving models trained on daytime data degrade significantly under adverse conditions such as night or rain. This project investigates whether **CycleGAN-based image translation** can generate synthetic adverse-weather driving images from clear daytime scenes, and whether augmenting training data with these synthetic images improves semantic segmentation robustness under real domain shift.

The pipeline has four stages:

1. **Data preparation** — filter BDD100K by weather/time-of-day
2. **Image translation** — train CycleGAN to translate day → night and day → rain
3. **Segmentation training** — train a baseline model (daytime only) and an augmented model (daytime + synthetic)
4. **Evaluation** — compare both models on real adverse-weather test images using mIoU, pixel accuracy, and per-class IoU

---

## Project Structure

```
weather_seg/
├── venv/                          # virtual environment (not tracked in git)
├── data/
│   ├── bdd100k/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   │   ├── day/
│   │   │   │   ├── night/
│   │   │   │   └── rain/
│   │   │   └── val/
│   │   │       ├── day/
│   │   │       ├── night/
│   │   │       └── rain/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── synthetic/
│       ├── night/                 # CycleGAN-generated synthetic night images
│       └── rain/                  # CycleGAN-generated synthetic rain images
├── cyclegan/
│   ├── __init__.py
│   ├── datasets.py                # unpaired image dataset loader
│   ├── models.py                  # Generator (ResNet-based) + PatchDiscriminator
│   ├── losses.py                  # GAN loss, cycle loss, identity loss
│   ├── train_cyclegan.py          # CycleGAN training loop
│   └── generate_synthetic.py      # inference: translate daytime → synthetic weather
├── segmentation/
│   ├── __init__.py
│   ├── datasets.py                # BDD100K segmentation dataset + transforms
│   ├── models.py                  # ResNet-50 encoder U-Net
│   ├── train_seg.py               # segmentation training loop
│   └── evaluate.py                # mIoU, pixel accuracy, per-class IoU
├── utils/
│   ├── __init__.py
│   ├── filter_bdd.py              # filter BDD100K images by weather/time-of-day
│   ├── metrics.py                 # shared metric utilities
│   └── transforms.py              # shared albumentations transforms
├── checkpoints/
│   ├── cyclegan/                  # saved CycleGAN generator weights (.pth)
│   └── segmentation/              # saved segmentation model weights (.pth)
├── outputs/
│   ├── logs/                      # TensorBoard logs
│   └── plots/                     # evaluation plots and qualitative results
├── main.py                        # end-to-end pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- ~50GB disk space for BDD100K + synthetic images + checkpoints

---

## Environment Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd weather_seg
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU availability

```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # your GPU name
```

---

## Dataset Setup

### 1. Download BDD100K

Create a free account at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) and download the following three archives:

| Archive | Size | Contents |
|---|---|---|
| `bdd100k_images_100k.zip` | ~6.5 GB | All driving images |
| `bdd100k_sem_seg_labels_trainval.zip` | ~400 MB | Pixel-level segmentation masks |
| `bdd100k_labels_release.zip` | ~100 MB | JSON metadata (weather, time-of-day) |

Unzip all three into `data/bdd100k/`.

### 2. Filter images by condition

```bash
python utils/filter_bdd.py
```

This reads the JSON metadata and copies images into the appropriate subfolders (`day/`, `night/`, `rain/`) under `data/bdd100k/images/train/` and `data/bdd100k/images/val/`. Segmentation masks are matched automatically by filename.

**Expected counts after filtering (approximate):**

| Split | Daytime | Night | Rain |
|---|---|---|---|
| Train | ~36,000 | ~27,000 | ~8,000 |
| Val | ~4,500 | ~3,400 | ~1,000 |

---

## Running the Pipeline

All stages can be run end-to-end from `main.py`:

```bash
python main.py
```

Or run each stage individually:

### Stage 1 — Train CycleGAN

```bash
python cyclegan/train_cyclegan.py
```

Trains two generators: `G_day2night` (day → night) and `G_night2day` (night → day). Checkpoints are saved every 10 epochs to `checkpoints/cyclegan/`. Training takes approximately **12–20 hours** on a single GPU.

### Stage 2 — Generate synthetic images

```bash
python cyclegan/generate_synthetic.py
```

Runs the trained `G_day2night` generator over all daytime training images and saves outputs to `data/synthetic/night/`. Original segmentation masks are reused without modification since CycleGAN preserves scene geometry.

### Stage 3 — Train baseline segmentation model

```bash
python segmentation/train_seg.py --run baseline
```

Trains on daytime images only. Best checkpoint saved to `checkpoints/segmentation/baseline_best.pth`.

### Stage 4 — Train augmented segmentation model

```bash
python segmentation/train_seg.py --run augmented
```

Trains on daytime + synthetic night images. Best checkpoint saved to `checkpoints/segmentation/augmented_best.pth`.

### Stage 5 — Evaluate and compare

```bash
python segmentation/evaluate.py
```

Evaluates both models on real night/rain test images and prints a side-by-side per-class IoU comparison.

---

## Monitoring Training

TensorBoard logs are written to `outputs/logs/`. To monitor training in real time:

```bash
tensorboard --logdir outputs/logs/
```

Then open `http://localhost:6006` in your browser.

---

## Hyperparameters

Defined as dictionaries at the top of `main.py`:

```python
CYCLEGAN_CONFIG = {
    'lr': 2e-4,
    'epochs': 200,
    'lambda_cycle': 10.0,
    'lambda_identity': 0.5
}

SEG_CONFIG = {
    'lr': 1e-4,
    'epochs': 80,
    'batch_size': 4
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection over Union across all 19 classes — primary metric |
| **Pixel accuracy** | Fraction of correctly classified pixels |
| **Per-class IoU** | IoU broken down by class (road, sky, car, person, etc.) |

The project is considered successful if the augmented model achieves higher mIoU on real adverse-weather images than the baseline.

---

## Expected Results

A well-trained baseline model typically drops **8–15 mIoU points** when evaluated on night images compared to daytime. Synthetic augmentation is expected to recover **3–7 of those points**. Classes that tend to benefit most include road, vegetation, and sky.

---

## Estimated Compute Time

| Stage | Estimated Time (RTX 3090) |
|---|---|
| Data download + filtering | 2–3 hrs |
| CycleGAN training (200 epochs) | 12–20 hrs |
| Synthetic image generation | 1–2 hrs |
| Baseline segmentation (80 epochs) | 4–6 hrs |
| Augmented segmentation (80 epochs) | 5–7 hrs |
| Evaluation + plots | ~1 hr |

**Total: approximately 2–3 days of compute.** Start CycleGAN training as early as possible — it is the longest step and can run overnight.

---

## .gitignore

```
venv/
data/
checkpoints/
outputs/
__pycache__/
*.pth
*.pyc
.DS_Store
```

---

## References

- Long et al., *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
- Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020