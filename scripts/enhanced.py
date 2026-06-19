"""
改进版微调: 双分支独立ArcFace Loss
解决问题：原版本图像分支塌陷（img-only mAP仅0.26）

改进点：
1. 图像分支和文本分支分别计算ArcFace Loss
2. 融合分支额外计算Loss
3. 三个Loss加权组合，防止某分支塌陷
4. 评估时使用消融实验得到的最优α
"""
import os
import re
import json
import random
import argparse
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# 兼容不同PyTorch版本
try:
    from torch.amp import autocast, GradScaler
    AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW_API = False
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import CLIPModel, CLIPProcessor

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ============ 工具函数 ============
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_title(title: str, max_len: int = 77) -> str:
    if pd.isna(title):
        return ""
    title = str(title)
    title = re.sub(r'[^\w\s\u4e00-\u9fff\u0e00-\u0e7f]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:max_len]


# ============ ArcFace Loss ============
class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss"""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(features, weight)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


# ============ 改进版多模态模型 ============
class ImprovedMultiModalModel(nn.Module):
    """
    改进版：三分支独立ArcFace
    - 图像分支: img_feat -> img_embed -> arc_head_img
    - 文本分支: txt_feat -> txt_embed -> arc_head_txt
    - 融合分支: concat -> fused_embed -> arc_head_fused
    """

    def __init__(self, clip_dir: str, num_classes: int, embed_dim: int = 512):
        super().__init__()

        # 加载 CLIP
        self.clip = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)

        clip_dim = self.clip.config.projection_dim  # 512

        # 图像分支投影层
        self.img_proj = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # 文本分支投影层
        self.txt_proj = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # 三个独立的ArcFace头
        self.arc_head_img = ArcMarginProduct(embed_dim, num_classes, s=30.0, m=0.50)
        self.arc_head_txt = ArcMarginProduct(embed_dim, num_classes, s=30.0, m=0.50)
        self.arc_head_fused = ArcMarginProduct(embed_dim, num_classes, s=30.0, m=0.50)

        self.embed_dim = embed_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip.get_image_features(pixel_values=images)

    def encode_text(self, titles: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.processor(
            text=titles, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        return self.clip.get_text_features(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )

    def forward(self, images: torch.Tensor, titles: List[str], labels: Optional[torch.Tensor] = None):
        device = images.device

        # 原始CLIP特征
        img_feat = self.encode_image(images)  # (B, 512)
        txt_feat = self.encode_text(titles, device)  # (B, 512)

        img_feat = F.normalize(img_feat, p=2, dim=1)
        txt_feat = F.normalize(txt_feat, p=2, dim=1)

        # 各分支embedding
        img_embed = F.normalize(self.img_proj(img_feat), p=2, dim=1)
        txt_embed = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)
        fused_embed = F.normalize(self.fusion(torch.cat([img_feat, txt_feat], dim=1)), p=2, dim=1)

        if labels is not None:
            # 训练模式：返回三个分支的logits
            logits_img = self.arc_head_img(img_embed, labels)
            logits_txt = self.arc_head_txt(txt_embed, labels)
            logits_fused = self.arc_head_fused(fused_embed, labels)
            return logits_img, logits_txt, logits_fused, img_embed, txt_embed, fused_embed

        # 推理模式
        return img_embed, txt_embed, fused_embed


# ============ 数据集 ============
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, processor: CLIPProcessor):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        title = clean_title(row["title"])
        label = row["label"]
        return pixel_values, title, label


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, processor: CLIPProcessor):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        title = clean_title(row["title"])
        return str(row["posting_id"]), pixel_values, title, row["label_group"]


def train_collate(batch):
    imgs, titles, labels = zip(*batch)
    return torch.stack(imgs), list(titles), torch.tensor(labels, dtype=torch.long)


def eval_collate(batch):
    pids, imgs, titles, lgs = zip(*batch)
    return list(pids), torch.stack(imgs), list(titles), list(lgs)


# ============ 评估 ============
def ap_at_k(pred: List[str], true_set: set, k: int) -> float:
    pred = pred[:k]
    if len(true_set) == 0:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, pid in enumerate(pred, start=1):
        if pid in seen:
            continue
        seen.add(pid)
        if pid in true_set:
            hits += 1
            score += hits / i
    return score / min(len(true_set), k)


def map_at_k(preds: Dict[str, List[str]], true_sets: Dict[str, set], k: int) -> float:
    scores = [ap_at_k(preds[q], true_sets[q], k) for q in preds if q in true_sets]
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, true_sets: Dict[str, set],
             topk: int, device: str, alpha: float = 0.4) -> Tuple[float, Dict]:
    """
    评估模型
    alpha: 图像权重（基于消融实验，最优α=0.4）
    """
    model.eval()

    all_ids, all_img, all_txt, all_fused = [], [], [], []

    for pids, imgs, titles, _ in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        img_embed, txt_embed, fused_embed = model(imgs, titles)

        all_ids.extend(pids)
        all_img.append(img_embed.cpu())
        all_txt.append(txt_embed.cpu())
        all_fused.append(fused_embed.cpu())

    img_emb = torch.cat(all_img).numpy()
    txt_emb = torch.cat(all_txt).numpy()
    fused_emb = torch.cat(all_fused).numpy()

    n = len(all_ids)

    # 1. 用融合embedding检索
    sim_fused = fused_emb @ fused_emb.T
    np.fill_diagonal(sim_fused, -np.inf)
    preds_fused = {}
    for i, qid in enumerate(all_ids):
        top_idx = np.argsort(-sim_fused[i])[:topk]
        preds_fused[qid] = [all_ids[j] for j in top_idx]
    map_fused = map_at_k(preds_fused, true_sets, topk)

    # 2. 用加权融合检索（基于消融实验的α）
    sim_img = img_emb @ img_emb.T
    sim_txt = txt_emb @ txt_emb.T
    sim_weighted = alpha * sim_img + (1 - alpha) * sim_txt
    np.fill_diagonal(sim_weighted, -np.inf)
    preds_weighted = {}
    for i, qid in enumerate(all_ids):
        top_idx = np.argsort(-sim_weighted[i])[:topk]
        preds_weighted[qid] = [all_ids[j] for j in top_idx]
    map_weighted = map_at_k(preds_weighted, true_sets, topk)

    # 3. 纯图像检索
    np.fill_diagonal(sim_img, -np.inf)
    preds_img = {}
    for i, qid in enumerate(all_ids):
        top_idx = np.argsort(-sim_img[i])[:topk]
        preds_img[qid] = [all_ids[j] for j in top_idx]
    map_img = map_at_k(preds_img, true_sets, topk)

    # 4. 纯文本检索
    np.fill_diagonal(sim_txt, -np.inf)
    preds_txt = {}
    for i, qid in enumerate(all_ids):
        top_idx = np.argsort(-sim_txt[i])[:topk]
        preds_txt[qid] = [all_ids[j] for j in top_idx]
    map_txt = map_at_k(preds_txt, true_sets, topk)

    # 返回最佳结果（通常是weighted）
    best_map = max(map_fused, map_weighted)
    best_preds = preds_weighted if map_weighted >= map_fused else preds_fused

    return best_map, {
        "map_fused": map_fused,
        "map_weighted": map_weighted,
        "map_img_only": map_img,
        "map_txt_only": map_txt,
        "predictions": best_preds
    }


# ============ 训练 ============
def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    loss_weights=(0.3, 0.3, 0.4)):
    """
    三分支联合训练
    loss_weights: (img_weight, txt_weight, fused_weight)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    w_img, w_txt, w_fused = loss_weights

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for imgs, titles, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda') if AMP_NEW_API else autocast():
            logits_img, logits_txt, logits_fused, _, _, _ = model(imgs, titles, labels)

            # 三分支损失
            loss_img = criterion(logits_img, labels)
            loss_txt = criterion(logits_txt, labels)
            loss_fused = criterion(logits_fused, labels)

            # 加权组合
            loss = w_img * loss_img + w_txt * loss_txt + w_fused * loss_fused

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "img": f"{loss_img.item():.2f}",
            "txt": f"{loss_txt.item():.2f}",
            "fused": f"{loss_fused.item():.2f}"
        })

    return total_loss / len(loader)


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/train.csv")
    parser.add_argument("--image_dir", default="data/train_images")
    parser.add_argument("--clip_dir", default="models/clip-vit-base-patch32")
    parser.add_argument("--output_dir", default="outputs/enhanced")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # 损失权重
    parser.add_argument("--w_img", type=float, default=0.3, help="图像分支损失权重")
    parser.add_argument("--w_txt", type=float, default=0.3, help="文本分支损失权重")
    parser.add_argument("--w_fused", type=float, default=0.4, help="融合分支损失权重")

    # 评估时的融合权重（基于消融实验）
    parser.add_argument("--eval_alpha", type=float, default=0.4, help="评估时图像权重")

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 加载数据
    print("[1/6] Loading data...")
    df = pd.read_csv(args.csv_path)
    df["posting_id"] = df["posting_id"].astype(str)

    # 划分
    print("[2/6] Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss.split(df, groups=df["label_group"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    # 训练集 label 编码
    le = LabelEncoder()
    df_train["label"] = le.fit_transform(df_train["label_group"])
    num_classes = df_train["label"].nunique()

    print(f"  Train: {len(df_train)} samples, {num_classes} classes")
    print(f"  Val: {len(df_val)} samples")

    # Ground truth
    val_group = df_val.groupby("label_group")["posting_id"].apply(list).to_dict()
    true_sets = {}
    for pid, lg in zip(df_val["posting_id"], df_val["label_group"]):
        same = set(val_group[lg])
        same.discard(pid)
        true_sets[pid] = same

    # 模型
    print("[3/6] Building model...")
    model = ImprovedMultiModalModel(args.clip_dir, num_classes, embed_dim=512).to(args.device)

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    # DataLoader
    processor = model.processor
    train_ds = TrainDataset(df_train, args.image_dir, processor)
    val_ds = EvalDataset(df_val, args.image_dir, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=eval_collate)

    # 评估模式
    if args.eval_only:
        print("\n[EVAL ONLY]")
        best_map, info = evaluate(model, val_loader, true_sets, args.topk, args.device, args.eval_alpha)
        print(f"mAP@{args.topk} (best): {best_map:.5f}")
        print(f"  - fused: {info['map_fused']:.5f}")
        print(f"  - weighted (α={args.eval_alpha}): {info['map_weighted']:.5f}")
        print(f"  - img-only: {info['map_img_only']:.5f}")
        print(f"  - txt-only: {info['map_txt_only']:.5f}")
        return

    # 训练
    print("[4/6] Training...")
    print(f"  Loss weights: img={args.w_img}, txt={args.w_txt}, fused={args.w_fused}")
    print(f"  Eval alpha: {args.eval_alpha}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda') if AMP_NEW_API else GradScaler()

    best_map, best_epoch = 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, args.device, epoch,
            loss_weights=(args.w_img, args.w_txt, args.w_fused)
        )
        scheduler.step()

        # 评估
        map_score, info = evaluate(model, val_loader, true_sets, args.topk, args.device, args.eval_alpha)

        print(f"[Epoch {epoch}] loss={train_loss:.4f}")
        print(f"  mAP@{args.topk}: best={map_score:.5f} | fused={info['map_fused']:.5f} | "
              f"weighted={info['map_weighted']:.5f} | img={info['map_img_only']:.5f} | txt={info['map_txt_only']:.5f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            f"mAP@{args.topk}": map_score,
            "mAP_fused": info['map_fused'],
            "mAP_weighted": info['map_weighted'],
            "mAP_img_only": info['map_img_only'],
            "mAP_txt_only": info['map_txt_only']
        })

        # 保存最佳
        if map_score > best_map:
            best_map = map_score
            best_epoch = epoch
            torch.save(model.state_dict(),
                      os.path.join(args.output_dir, "checkpoints", "best_model.pth"))
            save_json(os.path.join(args.output_dir, "best_predictions.json"), info['predictions'])
            print(f"  ★ New best!")

    # 最终结果
    print("\n" + "="*70)
    print(f"[RESULT] Best mAP@{args.topk}: {best_map:.5f} at epoch {best_epoch}")
    print("="*70)

    # 保存
    summary = {
        "best_epoch": best_epoch,
        f"best_mAP@{args.topk}": best_map,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "loss_weights": {"img": args.w_img, "txt": args.w_txt, "fused": args.w_fused},
        "eval_alpha": args.eval_alpha,
        "history": history
    }
    save_json(os.path.join(args.output_dir, "training_result.json"), summary)
    print(f"\n[SAVED] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
