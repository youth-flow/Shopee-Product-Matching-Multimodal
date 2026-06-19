# Models Directory

This directory is intentionally kept out of Git except for this note.

Expected local layout:

```text
models/
  clip-vit-base-patch32/
  resnet50/
    resnet50_imagenet1k_v2.pth
```

Download CLIP:

```powershell
huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/clip-vit-base-patch32
```

Download ResNet50 ImageNet V2 weights:

```powershell
mkdir models\resnet50
python -c "from torchvision.models import ResNet50_Weights; import os, torch; os.makedirs('models/resnet50', exist_ok=True); sd = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True); torch.save(sd, 'models/resnet50/resnet50_imagenet1k_v2.pth')"
```

Model weights are large and are ignored by `.gitignore`.
