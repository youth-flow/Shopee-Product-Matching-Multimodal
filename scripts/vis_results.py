"""
检索结果可视化展示
读取 best_predictions.json，展示3个示例的 Top-3 检索结果
"""
import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

# ============ 配置 ============
PREDICTIONS_PATH = "outputs/enhanced_frozen/best_predictions.json"
CSV_PATH = "data/train.csv"
IMAGE_DIR = "data/train_images"
NUM_EXAMPLES = 3
TOP_K = 3

# ============ 加载数据 ============
print("Loading data...")

with open(PREDICTIONS_PATH, "r") as f:
    predictions = json.load(f)

df = pd.read_csv(CSV_PATH)
df["posting_id"] = df["posting_id"].astype(str)

id_to_info = {}
for _, row in df.iterrows():
    id_to_info[str(row["posting_id"])] = {
        "title": row["title"],
        "image": row["image"],
        "label_group": row["label_group"]
    }

print(f"Done: {len(predictions)} predictions, {len(df)} products")

# ============ 随机选择示例 ============
query_ids = list(predictions.keys())
sample_ids = random.sample(query_ids, min(NUM_EXAMPLES, len(query_ids)))

# ============ 可视化 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 3行4列布局
fig, axes = plt.subplots(NUM_EXAMPLES, 4, figsize=(14, 12))

for row, qid in enumerate(sample_ids):
    query_info = id_to_info.get(qid, {})
    query_label = query_info.get("label_group", "Unknown")
    query_title = str(query_info.get('title', ''))

    # ========== Query 图片 ==========
    ax = axes[row, 0]
    query_img_path = os.path.join(IMAGE_DIR, query_info.get("image", ""))
    if os.path.exists(query_img_path):
        img = Image.open(query_img_path)
        ax.imshow(img)

    for spine in ax.spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(3)
    ax.set_xticks([])
    ax.set_yticks([])

    # 标题在下方
    wrapped = textwrap.fill(query_title, width=20)[:60]
    ax.set_xlabel(f"Query\n{wrapped}", fontsize=8, color='blue', fontweight='bold')

    # ========== Top-K 结果 ==========
    top_preds = predictions[qid][:TOP_K]
    for j, pred_id in enumerate(top_preds):
        pred_info = id_to_info.get(pred_id, {})
        pred_label = pred_info.get("label_group", "Unknown")
        pred_title = str(pred_info.get('title', ''))

        is_match = (pred_label == query_label)
        color = "green" if is_match else "red"
        tag = "Match" if is_match else "Miss"

        ax = axes[row, j + 1]
        pred_img_path = os.path.join(IMAGE_DIR, pred_info.get("image", ""))
        if os.path.exists(pred_img_path):
            img = Image.open(pred_img_path)
            ax.imshow(img)

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

        wrapped = textwrap.fill(pred_title, width=20)[:60]
        ax.set_xlabel(f"Top-{j+1} [{tag}]\n{wrapped}", fontsize=8, color=color)

# 列标题
cols = ["Query", "Top-1", "Top-2", "Top-3"]
for j, col in enumerate(cols):
    axes[0, j].set_title(col, fontsize=12, fontweight='bold', pad=10)

# 底部图例
fig.text(0.5, 0.02,
         "Blue = Query  |  Green = Match  |  Red = Miss",
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))

plt.suptitle("Multi-Modal Product Retrieval Results", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.15)

plt.savefig("retrieval_visualization.png", dpi=150, bbox_inches="tight", facecolor='white')
plt.savefig("retrieval_visualization.pdf", bbox_inches="tight", facecolor='white')
print("Saved: retrieval_visualization.png / .pdf")
plt.show()