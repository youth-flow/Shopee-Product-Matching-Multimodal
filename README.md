# Shopee Product Matching Multimodal

This repository contains a course project for multimodal product matching on the Shopee Price Match Guarantee dataset. The project compares image-only retrieval, CLIP image/text retrieval, and ArcFace-based multimodal fine-tuning.

The repository intentionally excludes raw images, cached embeddings, pretrained weights, and trained checkpoints. Those files are large and can be downloaded or regenerated from public sources. Keeping them out of Git avoids oversized history and makes the repository easy to clone.

## What is included

- `scripts/`: training, ablation, evaluation, and visualization scripts.
- `outputs/*/*_result.json`: lightweight experiment summaries.
- `outputs/visualizations/*.png`: summary figures generated from the result JSON files.
- `retrieval_visualization.png` and `retrieval_visualization.pdf`: retrieval examples.
- Course presentation file in the repository root.

## What is not included

- `data/train.csv`
- `data/train_images/`
- `models/clip-vit-base-patch32/`
- `models/resnet50/resnet50_imagenet1k_v2.pth`
- `outputs/**/checkpoints/`
- `outputs/**/*.npy`
- `outputs/**/*predictions*.json`

These files are ignored by `.gitignore`.

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your local CUDA version before running the training scripts.

## Data

Download the Shopee Price Match Guarantee dataset from Kaggle:

https://www.kaggle.com/c/shopee-product-matching/data

Place the files like this:

```text
data/
  train.csv
  train_images/
    0000a68812bc7e98c42888dfb1c07da0.jpg
    ...
```

The scripts use `data/train.csv` and `data/train_images` by default.

## Model Weights

Download CLIP ViT-B/32 from Hugging Face:

```powershell
huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/clip-vit-base-patch32
```

The CLIP scripts use `local_files_only=True`, so the model must exist under `models/clip-vit-base-patch32`.

Download the ResNet50 ImageNet V2 weights used by the baseline:

```powershell
mkdir models\resnet50
python -c "from torchvision.models import ResNet50_Weights; import os, torch; os.makedirs('models/resnet50', exist_ok=True); sd = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True); torch.save(sd, 'models/resnet50/resnet50_imagenet1k_v2.pth')"
```

## Run Experiments

Baseline ResNet50 image retrieval:

```powershell
python scripts\baseline.py
```

CLIP image/text ablation:

```powershell
python scripts\ablation.py
```

ArcFace multimodal training with frozen CLIP encoders:

```powershell
python scripts\enhanced_frozen.py
```

Full CLIP fine-tuning variant:

```powershell
python scripts\enhanced.py
```

Generate summary figures from result JSON files:

```powershell
python scripts\visualize.py
```

Generate retrieval examples:

```powershell
python scripts\vis_results.py
```

## Results

All results below use the default grouped validation split (`seed=42`, `val_ratio=0.2`) and mAP@50.

| Method | mAP@50 |
| --- | ---: |
| ResNet50 baseline | 0.65918 |
| CLIP image-only | 0.64558 |
| CLIP text-only | 0.67401 |
| CLIP fusion (`alpha=0.4`) | 0.81675 |
| ArcFace multimodal fine-tuning | 0.86530 |
| ArcFace with frozen CLIP encoders | 0.89610 |

The best recorded run is the frozen-CLIP ArcFace model at epoch 8 with mAP@50 = 0.89610.

## Large Artifacts

If trained checkpoints or prediction files need to be published later, use GitHub Releases, Kaggle Datasets, or Hugging Face Hub instead of committing them to Git. The local ignored files currently include checkpoints around 630 MB and pretrained CLIP weights around 577 MB.
