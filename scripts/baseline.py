"""
Baseline: ResNet50 纯图像特征检索
- 不使用文本
- 预训练权重，不微调
"""
import os
import json
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit

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


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


# ============ 数据集 ============
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        return str(row["posting_id"]), self.transform(img)


# ============ 模型 ============
def build_resnet50(weights_path: str, device: str) -> nn.Module:
    model = models.resnet50(weights=None)

    # 加载权重
    sd = torch.load(weights_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    elif "model" in sd:
        sd = sd["model"]

    # 去除 module. 前缀
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v

    model.load_state_dict(new_sd, strict=False)
    model.fc = nn.Identity()  # 输出 2048 维特征
    model.eval().to(device)
    return model


# ============ 特征提取 ============
@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    image_dir: str,
    model: nn.Module,
    device: str,
    batch_size: int = 32
) -> Tuple[List[str], np.ndarray]:

    ds = ImageDataset(df, image_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ids, embs = [], []
    for pids, imgs in tqdm(dl, desc="Extracting ResNet features"):
        imgs = imgs.to(device)
        feat = model(imgs)
        feat = l2_normalize(feat)
        ids.extend(list(pids))
        embs.append(feat.cpu())

    return ids, torch.cat(embs, dim=0).numpy().astype(np.float32)


# ============ KNN 检索 ============
def knn_search(query_emb: np.ndarray, db_emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (similarities, indices)"""
    if HAS_FAISS:
        d = db_emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(db_emb.astype(np.float32))
        sims, idxs = index.search(query_emb.astype(np.float32), topk)
        return sims, idxs
    else:
        # sklearn fallback
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(db_emb)
        dist, idxs = nn.kneighbors(query_emb, n_neighbors=topk)
        return 1 - dist, idxs


# ============ 评估指标 ============
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


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/train.csv")
    parser.add_argument("--image_dir", default="data/train_images")
    parser.add_argument("--resnet_weights", default="models/resnet50/resnet50_imagenet1k_v2.pth")
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print("[1/5] Loading data...")
    df = pd.read_csv(args.csv_path)
    df["posting_id"] = df["posting_id"].astype(str)

    # 划分数据集
    print("[2/5] Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss.split(df, groups=df["label_group"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}")

    # 构建 ground truth
    val_group = df_val.groupby("label_group")["posting_id"].apply(list).to_dict()
    true_sets = {}
    for pid, lg in zip(df_val["posting_id"], df_val["label_group"]):
        same = set(val_group[lg])
        same.discard(pid)  # 去掉自己
        true_sets[pid] = same

    # 加载模型
    print("[3/5] Loading ResNet50...")
    model = build_resnet50(args.resnet_weights, args.device)

    # 提取特征
    print("[4/5] Extracting embeddings...")
    val_ids, val_emb = extract_embeddings(df_val, args.image_dir, model, args.device, args.batch_size)

    # KNN 检索
    print("[5/5] KNN search and evaluation...")
    sims, idxs = knn_search(val_emb, val_emb, args.topk + 1)  # +1 去掉自己

    # 构建预测结果
    preds = {}
    for i, qid in enumerate(val_ids):
        # 去掉自己
        candidates = [val_ids[j] for j in idxs[i] if val_ids[j] != qid][:args.topk]
        preds[qid] = candidates

    # 计算指标
    map_score = map_at_k(preds, true_sets, args.topk)

    print("\n" + "="*50)
    print(f"[RESULT] Baseline ResNet50 mAP@{args.topk}: {map_score:.5f}")
    print("="*50)

    # 保存结果
    result = {
        "method": "baseline_resnet50",
        f"mAP@{args.topk}": map_score,
        "val_samples": len(df_val),
        "val_groups": df_val["label_group"].nunique()
    }
    save_json(os.path.join(args.output_dir, "baseline_result.json"), result)
    save_json(os.path.join(args.output_dir, "baseline_predictions.json"), preds)

    # 保存 embedding 供后续使用
    np.save(os.path.join(args.output_dir, "val_ids.npy"), np.array(val_ids, dtype=object))
    np.save(os.path.join(args.output_dir, "val_resnet_emb.npy"), val_emb)

    print(f"\n[SAVED] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
