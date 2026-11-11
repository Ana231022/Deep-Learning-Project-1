# Deep-Learning-Project-1
# Efficient ResNet on CIFAR-10 (≤ 5M parameters)

**Deep Learning Project 1, Fall 2026, NYU Tandon** · Kaggle team **NovaVerse** · Ananya Bhatnagar (ab13093)

A pre-activation Wide ResNet trained from scratch on the provided CIFAR-10 training data. We also tried knowledge distillation from a larger teacher trained from scratch; the final entry is the student trained without it.

| | |
|---|---|
| **Final model** | WRN-22-4w (pre-activation Wide ResNet), run v2-D, trained on all 50,000 images |
| **Trainable parameters** | **4,851,866** (limit 5,000,000) |
| **Kaggle test accuracy** | **89.70%** (public leaderboard) |
| **Validation accuracy** | 95.90% (same architecture and recipe, 100 epochs on 45,000 images, 5,000 held out) |

## Results

| Run | Model | Parameters | Training images | Epochs | CutMix | Teacher | Validation accuracy | Kaggle public |
|---|---|---|---|---|---|---|---|---|
| v1 | WRN-22-4 | 4,298,970 | 45,000 | 100 | No | – | 96.38% | 87.30% |
| v2-A | WRN-22-4w | 4,851,866 | 45,000 | 100 | Yes | – | 95.90% | [fill in] |
| **v2-D (final)** | WRN-22-4w | 4,851,866 | 50,000 | 200 | Yes | – | – | **89.70%** |
| v2-C | WRN-22-4w | 4,851,866 | 50,000 | 200 | Yes | WRN-28-10 | – | [fill in] |
| v2-B (teacher, **over the limit, not our entry**) | WRN-28-10 | 36,479,194 | 50,000 | 200 | Yes | – | – | 89.70% |

Validation accuracy uses flip test-time augmentation on a fixed 5,000-image split (seed 42). Runs on all 50,000 images have no validation split. Per-run configs, epoch logs and summaries are in [`results/`](results/); training curves in `results/curves_all.png`.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/cifar_resnet_v2.ipynb` | **Main notebook, with outputs:** final model (v2-D), teacher, distillation, results table, plots, parameter count |
| `notebooks/cifar_resnet_v1.ipynb` | First submission (WRN-22-4), with outputs |
| `cifar_lib.py` | Models, data loading, augmentation, training loop (CutMix, EMA, distillation), prediction |
| `run_v2.py` | Command-line version of the v2 notebook (stages A–D) |
| `predict.py` | Re-creates `submission.csv` from a saved checkpoint, prints the parameter count |
| `results/` | Config, log, summary and submission for every run; checkpoints of the students |
| `tools/collect_results.py` | Copies those files from a training folder into `results/` |

## Reproduce

```bash
pip install -r requirements.txt     # PyTorch with CUDA: use the command from pytorch.org (RTX 50-series needs CUDA 12.8+)
# put the competition files in data/ (see data/README.md)

# 1. re-create the submitted predictions from the saved checkpoint (seconds)
python predict.py --ckpt results/v2_D_student_cutmix_full/best.pt --data_dir data --out submission.csv

# 2. retrain everything from scratch (about 5-6 h on one RTX 5080)
python run_v2.py                      # stages A, B (teacher), C (final student), D
python run_v2.py --stages D           # only the final student (no teacher needed)
```

Outputs go to `runs/<run_name>/`. Runs resume automatically if interrupted. On Windows, if the data loader fails, add `--workers 0`.

The notebooks do the same thing interactively. They expect the data in `C:\cifar\data` (the paths used for the reported runs); change `data_dir`, `save_root` and `cache_dir` in the config cell for another location.

## Method

**Architecture.** Pre-activation residual blocks (BN → ReLU → 3×3 conv → BN → ReLU → 3×3 conv, plus a skip connection that is the identity, or a 1×1 stride-2 convolution where the shape changes). A 3×3 stem with 16 channels, three stages of 3 blocks each with 68 / 136 / 272 channels at 32×32 / 16×16 / 8×8, then BN, ReLU, 8×8 global average pooling and a linear layer.

**Training.** SGD with Nesterov momentum 0.9; learning rate 0.1 (batch 128) with 5 epochs of warm-up and cosine decay; weight decay 5e-4 (not on BN/bias); label smoothing 0.1; random crop + flip + TrivialAugmentWide + random erasing (p = 0.25); CutMix (α = 1, on half the batches); EMA of weights (0.999); bf16 mixed precision; flip test-time augmentation.

**Distillation (tried, not in the final entry).** The WRN-28-10 teacher is trained from scratch on the same 50,000 images with the same recipe. The distilled student (v2-C) minimises `(1 − α)·CE + α·T²·KL(teacher ‖ student)` with α = 0.7 and T = 4. It did not beat the student trained alone, so the final entry is v2-D. The teacher (89.70%, 36.5M parameters) is never part of the submitted model.

**Data rules.** Only the provided training images are used. No pretrained weights or external data.

## References

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 · *Identity Mappings in Deep Residual Networks*, ECCV 2016
- Zagoruyko & Komodakis, *Wide Residual Networks*, BMVC 2016
- Müller & Hutter, *TrivialAugment*, ICCV 2021
- Zhong et al., *Random Erasing Data Augmentation*, AAAI 2020 · DeVries & Taylor, *Cutout*, 2017
- Yun et al., *CutMix*, ICCV 2019
- Szegedy et al., *Rethinking the Inception Architecture* (label smoothing), CVPR 2016
- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*, 2015
- Loshchilov & Hutter, *SGDR*, ICLR 2017
- Recht et al., *Do CIFAR-10 Classifiers Generalize to CIFAR-10?*, 2018
