"""
消融实验: CLIP 模型对比
- CLIP image-only (α=1.0)
- CLIP text-only (α=0.0)
- CLIP image+text (α扫描)

目的：分解多模态各部分的贡献
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
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


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


# ============ 数据集 ============
class CLIPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        return str(row["posting_id"]), img, title


def collate_fn(batch):
    pids, imgs, titles = zip(*batch)
    return list(pids), list(imgs), list(titles)


# ============ 特征提取 ============
@torch.no_grad()
def extract_clip_embeddings(
    df: pd.DataFrame,
    image_dir: str,
    clip_dir: str,
    device: str,
    batch_size: int = 16
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """提取 CLIP 图像和文本特征"""

    processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_dir, local_files_only=True).eval().to(device)

    ds = CLIPDataset(df, image_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, collate_fn=collate_fn)

    ids, img_embs, txt_embs = [], [], []

    for pids, images, titles in tqdm(dl, desc="Extracting CLIP features"):
        inputs = processor(
            text=titles,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        img_feat = l2_normalize(img_feat)
        txt_feat = l2_normalize(txt_feat)

        ids.extend(list(pids))
        img_embs.append(img_feat.cpu())
        txt_embs.append(txt_feat.cpu())

    img_emb = torch.cat(img_embs, dim=0).numpy().astype(np.float32)
    txt_emb = torch.cat(txt_embs, dim=0).numpy().astype(np.float32)

    return ids, img_emb, txt_emb


# ============ KNN 检索 ============
def knn_search(query_emb: np.ndarray, db_emb: np.ndarray, topk: int):
    if HAS_FAISS:
        d = db_emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(db_emb.astype(np.float32))
        sims, idxs = index.search(query_emb.astype(np.float32), topk)
        return sims, idxs
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(db_emb)
        dist, idxs = nn.kneighbors(query_emb, n_neighbors=topk)
        return 1 - dist, idxs


# ============ 融合检索 ============
def retrieve_with_alpha(
    ids: List[str],
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    alpha: float,
    topk: int
) -> Dict[str, List[str]]:
    """
    融合检索: score = alpha * img_sim + (1-alpha) * txt_sim

    alpha=1.0: 纯图像
    alpha=0.0: 纯文本
    """
    n = len(ids)
    id2idx = {pid: i for i, pid in enumerate(ids)}

    # 计算相似度矩阵
    img_sim = img_emb @ img_emb.T  # (n, n)
    txt_sim = txt_emb @ txt_emb.T  # (n, n)

    fused_sim = alpha * img_sim + (1 - alpha) * txt_sim

    # 对角线设为负无穷（去掉自己）
    np.fill_diagonal(fused_sim, -np.inf)

    # 取 topk
    preds = {}
    for i, qid in enumerate(ids):
        top_idx = np.argsort(-fused_sim[i])[:topk]
        preds[qid] = [ids[j] for j in top_idx]

    return preds


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
    parser.add_argument("--clip_dir", default="models/clip-vit-base-patch32")
    parser.add_argument("--output_dir", default="outputs/ablation")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # alpha 列表：包含纯图像(1.0)、纯文本(0.0)、以及多种融合比例
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print("[1/4] Loading data...")
    df = pd.read_csv(args.csv_path)
    df["posting_id"] = df["posting_id"].astype(str)

    # 划分数据集
    print("[2/4] Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss.split(df, groups=df["label_group"]))
    df_val = df.iloc[val_idx].reset_index(drop=True)
    print(f"  Val: {len(df_val)} samples, {df_val['label_group'].nunique()} groups")

    # 构建 ground truth
    val_group = df_val.groupby("label_group")["posting_id"].apply(list).to_dict()
    true_sets = {}
    for pid, lg in zip(df_val["posting_id"], df_val["label_group"]):
        same = set(val_group[lg])
        same.discard(pid)
        true_sets[pid] = same

    # 提取 CLIP 特征
    print("[3/4] Extracting CLIP embeddings...")
    val_ids, img_emb, txt_emb = extract_clip_embeddings(
        df_val, args.image_dir, args.clip_dir, args.device, args.batch_size
    )

    # 消融实验
    print("\n[4/4] Ablation experiments...")
    print("="*60)

    results = []
    best_alpha, best_map = None, -1.0

    for alpha in args.alphas:
        preds = retrieve_with_alpha(val_ids, img_emb, txt_emb, alpha, args.topk)
        map_score = map_at_k(preds, true_sets, args.topk)

        # 标注模式
        if alpha == 1.0:
            mode = "image-only"
        elif alpha == 0.0:
            mode = "text-only"
        else:
            mode = f"fusion"

        print(f"  α={alpha:.1f} ({mode:12s}): mAP@{args.topk} = {map_score:.5f}")

        results.append({
            "alpha": alpha,
            "mode": mode,
            f"mAP@{args.topk}": map_score
        })

        if map_score > best_map:
            best_map = map_score
            best_alpha = alpha

    print("="*60)
    print(f"\n[BEST] α={best_alpha:.1f}, mAP@{args.topk} = {best_map:.5f}")

    # 分解贡献分析
    img_only = next(r for r in results if r["alpha"] == 1.0)[f"mAP@{args.topk}"]
    txt_only = next(r for r in results if r["alpha"] == 0.0)[f"mAP@{args.topk}"]

    print("\n[ANALYSIS] 贡献分解:")
    print(f"  CLIP image-only:  {img_only:.5f}")
    print(f"  CLIP text-only:   {txt_only:.5f}")
    print(f"  Best fusion:      {best_map:.5f} (α={best_alpha:.1f})")
    print(f"  文本增益: {best_map - img_only:+.5f} (相比纯图像)")

    # 保存结果
    summary = {
        "ablation_results": results,
        "best_alpha": best_alpha,
        "best_mAP": best_map,
        "analysis": {
            "clip_image_only": img_only,
            "clip_text_only": txt_only,
            "text_contribution": best_map - img_only
        }
    }
    save_json(os.path.join(args.output_dir, "ablation_result.json"), summary)

    # 保存 embedding
    np.save(os.path.join(args.output_dir, "val_ids.npy"), np.array(val_ids, dtype=object))
    np.save(os.path.join(args.output_dir, "val_clip_img.npy"), img_emb)
    np.save(os.path.join(args.output_dir, "val_clip_txt.npy"), txt_emb)

    print(f"\n[SAVED] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
