"""
可视化脚本 - 自动读取实验结果JSON文件
支持对比 enhanced 和 enhanced_frozen 两个版本

运行方式：
    python visualize.py

或指定路径：
    python visualize.py \
        --baseline_json outputs/baseline/baseline_result.json \
        --ablation_json outputs/ablation/ablation_result.json \
        --enhanced_json outputs/enhanced/training_result.json \
        --frozen_json outputs/enhanced_frozen/training_result.json \
        --output_dir outputs/visualizations
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ============ 中文字体设置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_json(path):
    """加载JSON文件，不存在返回None"""
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def parse_results(baseline_json, ablation_json, enhanced_json, frozen_json=None):
    """解析所有实验结果"""

    # 读取baseline
    baseline_data = load_json(baseline_json)
    if not baseline_data:
        raise FileNotFoundError(f"文件不存在: {baseline_json}")
    baseline_map = baseline_data.get('mAP@50', baseline_data.get('mAP', 0))

    # 读取消融实验
    ablation_data = load_json(ablation_json)
    if not ablation_data:
        raise FileNotFoundError(f"文件不存在: {ablation_json}")
    ablation_results = ablation_data.get('ablation_results', [])
    best_alpha = ablation_data.get('best_alpha', 0.4)
    best_fusion_map = ablation_data.get('best_mAP', 0)

    # 解析消融结果
    clip_img = 0
    clip_txt = 0
    for r in ablation_results:
        if r['alpha'] == 1.0:
            clip_img = r['mAP@50']
        elif r['alpha'] == 0.0:
            clip_txt = r['mAP@50']

    # 读取微调结果(原版)
    enhanced_data = load_json(enhanced_json)
    if not enhanced_data:
        raise FileNotFoundError(f"文件不存在: {enhanced_json}")
    arcface_map = enhanced_data.get('best_mAP@50', enhanced_data.get('best_mAP', 0))
    training_history = enhanced_data.get('history', [])

    # 读取微调结果(冻结版) - 可选
    frozen_data = load_json(frozen_json)
    frozen_map = 0
    frozen_history = []
    if frozen_data:
        frozen_map = frozen_data.get('best_mAP@50', frozen_data.get('best_mAP', 0))
        frozen_history = frozen_data.get('history', [])

    results = {
        'baseline': baseline_map,
        'clip_img': clip_img,
        'clip_txt': clip_txt,
        'clip_fusion': best_fusion_map,
        'best_alpha': best_alpha,
        'arcface': arcface_map,
        'arcface_frozen': frozen_map
    }

    return results, ablation_results, training_history, frozen_history


# ============ 图1: 主要结果对比柱状图 ============
def plot_main_results(results: dict, output_path: str):
    """各方法mAP@50对比"""

    best_alpha = results.get('best_alpha', 0.4)
    has_frozen = results.get('arcface_frozen', 0) > 0

    if has_frozen:
        methods = [
            'Baseline\n(ResNet50)',
            'CLIP\nimage-only',
            'CLIP\ntext-only',
            f'CLIP fusion\n(α={best_alpha})',
            'ArcFace\n(不冻结)',
            'ArcFace\n(冻结CLIP)'
        ]
        scores = [
            results['baseline'],
            results['clip_img'],
            results['clip_txt'],
            results['clip_fusion'],
            results['arcface'],
            results['arcface_frozen']
        ]
        colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    else:
        methods = [
            'Baseline\n(ResNet50)',
            'CLIP\nimage-only',
            'CLIP\ntext-only',
            f'CLIP fusion\n(α={best_alpha})',
            'ArcFace\n微调'
        ]
        scores = [
            results['baseline'],
            results['clip_img'],
            results['clip_txt'],
            results['clip_fusion'],
            results['arcface']
        ]
        colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(14 if has_frozen else 12, 7))
    bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    # 添加数值标签
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{score:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Baseline参考线
    baseline = results['baseline']
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(methods) - 0.4, baseline + 0.008, 'Baseline', fontsize=10, color='gray')

    ax.set_ylabel('mAP@50', fontsize=14)
    ax.set_title('各方法性能对比', fontsize=18, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)

    # 添加提升百分比
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if i > 0:
            gain = (score - baseline) / baseline * 100
            sign = '+' if gain > 0 else ''
            color = '#27ae60' if gain > 0 else '#c0392b'
            ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                    f'{sign}{gain:.1f}%', ha='center', fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[已保存] {output_path}")


# ============ 图2: 消融实验曲线 ============
def plot_alpha_curve(ablation_data: list, output_path: str):
    """消融实验α曲线"""

    if not ablation_data:
        print("[跳过] 无消融数据")
        return

    # 按alpha排序
    ablation_data = sorted(ablation_data, key=lambda x: x['alpha'])
    alphas = [d['alpha'] for d in ablation_data]
    maps = [d['mAP@50'] for d in ablation_data]

    fig, ax = plt.subplots(figsize=(11, 7))

    # 主曲线
    ax.plot(alphas, maps, 'o-', color='#3498db', linewidth=2.5, markersize=12,
            markerfacecolor='white', markeredgewidth=2.5, label='mAP@50')

    # 填充曲线下方区域
    ax.fill_between(alphas, maps, alpha=0.15, color='#3498db')

    # 标注最佳点
    best_idx = np.argmax(maps)
    best_alpha, best_map = alphas[best_idx], maps[best_idx]
    ax.scatter([best_alpha], [best_map], color='#e74c3c', s=300, zorder=5,
               marker='*', label=f'Best: α={best_alpha}')

    # 最优点标注
    ax.annotate(f'最优: {best_map:.4f}', xy=(best_alpha, best_map),
                xytext=(best_alpha + 0.1, best_map + 0.012),
                fontsize=11, fontweight='bold', color='#c0392b')

    # 端点标注
    ax.text(0.0, maps[0] - 0.022, '纯文本', fontsize=10, color='#7f8c8d', ha='center')
    ax.text(1.0, maps[-1] - 0.022, '纯图像', fontsize=10, color='#7f8c8d', ha='center')

    ax.set_xlabel('α (图像权重)', fontsize=14)
    ax.set_ylabel('mAP@50', fontsize=14)
    ax.set_title('多模态融合权重消融实验', fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)

    # legend放右下角
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # 公式放左下角
    ax.text(0.02, 0.05, r'$score = \alpha \cdot sim_{img} + (1-\alpha) \cdot sim_{txt}$',
            transform=ax.transAxes, fontsize=12, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', alpha=0.9, edgecolor='#cccc00'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[已保存] {output_path}")


# ============ 图3: 训练曲线（通用函数） ============
def plot_training_curve(history: list, output_path: str, title_suffix: str = ""):
    """训练Loss和mAP曲线 - 显示所有分支"""

    if not history:
        print(f"[跳过] 无训练历史数据 {title_suffix}")
        return

    epochs = [h['epoch'] for h in history]
    losses = [h['train_loss'] for h in history]

    # 提取各分支mAP
    maps_best = [h.get('mAP@50', 0) for h in history]
    maps_fused = [h.get('mAP_fused', h.get('mAP@50', 0)) for h in history]
    maps_weighted = [h.get('mAP_weighted', 0) for h in history]
    maps_img = [h.get('mAP_img_only', 0) for h in history]
    maps_txt = [h.get('mAP_txt_only', 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # === 左图: Loss曲线 ===
    ax1.plot(epochs, losses, 'o-', color='#e74c3c', linewidth=2.5, markersize=9,
             markerfacecolor='white', markeredgewidth=2)

    for e, l in zip(epochs, losses):
        ax1.text(e, l + 0.5, f'{l:.1f}', ha='center', fontsize=9)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title(f'训练损失曲线{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-')
    ax1.set_axisbelow(True)
    ax1.set_xticks(epochs)

    # === 右图: mAP曲线（所有分支） ===
    has_weighted = any(m > 0 for m in maps_weighted)
    has_txt = any(m > 0 for m in maps_txt)

    if has_weighted:
        ax2.plot(epochs, maps_weighted, 'o-', color='#2ecc71', linewidth=2.5, markersize=9,
                 markerfacecolor='white', markeredgewidth=2, label='Weighted (α融合)')
        ax2.plot(epochs, maps_fused, 's-', color='#9b59b6', linewidth=2, markersize=7,
                 markerfacecolor='white', markeredgewidth=2, label='Fused (concat)')
    else:
        ax2.plot(epochs, maps_fused, 'o-', color='#2ecc71', linewidth=2.5, markersize=9,
                 markerfacecolor='white', markeredgewidth=2, label='Fused (img+txt)')

    if has_txt:
        ax2.plot(epochs, maps_txt, '^--', color='#e74c3c', linewidth=1.8, markersize=7,
                 markerfacecolor='white', markeredgewidth=1.8, label='Text-only')

    ax2.plot(epochs, maps_img, 's--', color='#3498db', linewidth=1.8, markersize=6,
             markerfacecolor='white', markeredgewidth=1.8, label='Image-only')

    # 标注最佳点
    best_idx = np.argmax(maps_best)
    best_epoch = epochs[best_idx]
    best_map = maps_best[best_idx]

    ax2.scatter([best_epoch], [best_map], color='#e74c3c', s=180, zorder=5, marker='*')
    ax2.annotate(f'Best: {best_map:.3f}', xy=(best_epoch, best_map),
                 xytext=(best_epoch - 1.5, best_map + 0.015), fontsize=10, fontweight='bold')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP@50', fontsize=12)
    ax2.set_title(f'验证集性能曲线{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-')
    ax2.set_axisbelow(True)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[已保存] {output_path}")


# ============ 图5: 结果汇总表格 ============
def plot_summary_table(results: dict, output_path: str):
    """结果汇总表格"""

    baseline = results['baseline']
    best_alpha = results.get('best_alpha', 0.4)
    has_frozen = results.get('arcface_frozen', 0) > 0

    # 找最优方法
    all_scores = {
        'clip_fusion': results['clip_fusion'],
        'arcface': results['arcface']
    }
    if has_frozen:
        all_scores['arcface_frozen'] = results['arcface_frozen']
    best_method = max(all_scores, key=all_scores.get)

    fig, ax = plt.subplots(figsize=(12, 5.5 if has_frozen else 4.5))
    ax.axis('off')

    # 表格数据
    table_data = [
        ['方法', 'mAP@50', '相对Baseline', '说明'],
        ['Baseline (ResNet50)', f"{results['baseline']:.4f}", '-', '纯图像CNN特征'],
        ['CLIP image-only', f"{results['clip_img']:.4f}",
         f"{(results['clip_img']-baseline)/baseline*100:+.1f}%", '消融实验 α=1.0'],
        ['CLIP text-only', f"{results['clip_txt']:.4f}",
         f"{(results['clip_txt']-baseline)/baseline*100:+.1f}%", '消融实验 α=0.0'],
        [f'CLIP fusion (α={best_alpha})', f"{results['clip_fusion']:.4f}",
         f"{(results['clip_fusion']-baseline)/baseline*100:+.1f}%",
         '★ 最优' if best_method == 'clip_fusion' else '多模态融合'],
        ['ArcFace (不冻结)', f"{results['arcface']:.4f}",
         f"{(results['arcface']-baseline)/baseline*100:+.1f}%",
         '★ 最优' if best_method == 'arcface' else '度量学习微调']
    ]

    if has_frozen:
        table_data.append([
            'ArcFace (冻结CLIP)', f"{results['arcface_frozen']:.4f}",
            f"{(results['arcface_frozen']-baseline)/baseline*100:+.1f}%",
            '★ 最优' if best_method == 'arcface_frozen' else '冻结CLIP微调'
        ])

    num_rows = len(table_data) - 1

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center',
                     colWidths=[0.32, 0.18, 0.22, 0.28])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # 表头样式
    for j in range(4):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 最优行高亮
    best_row_map = {
        'clip_fusion': 4,
        'arcface': 5,
        'arcface_frozen': 6
    }
    best_row = best_row_map.get(best_method, 5)
    if best_row <= num_rows:
        for j in range(4):
            table[(best_row, j)].set_facecolor('#d5f5e3')
            table[(best_row, j)].set_text_props(fontweight='bold')

    # 交替行颜色
    for i in range(1, num_rows + 1):
        if i != best_row:
            color = '#f8f9fa' if i % 2 == 0 else 'white'
            for j in range(4):
                table[(i, j)].set_facecolor(color)

    ax.set_title('实验结果汇总', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[已保存] {output_path}")


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='生成实验可视化图表')
    parser.add_argument('--baseline_json', type=str, default='outputs/baseline/baseline_result.json')
    parser.add_argument('--ablation_json', type=str, default='outputs/ablation/ablation_result.json')
    parser.add_argument('--enhanced_json', type=str, default='outputs/enhanced/training_result.json')
    parser.add_argument('--frozen_json', type=str, default='outputs/enhanced_frozen/training_result.json')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("读取实验结果并生成可视化图表")
    print("=" * 60)

    # 读取并解析结果
    print(f"\n读取文件:")
    print(f"  Baseline: {args.baseline_json}")
    print(f"  Ablation: {args.ablation_json}")
    print(f"  Enhanced: {args.enhanced_json}")
    print(f"  Frozen:   {args.frozen_json}")

    try:
        results, ablation_data, training_history, frozen_history = parse_results(
            args.baseline_json, args.ablation_json, args.enhanced_json, args.frozen_json
        )
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("请确保已运行相应实验脚本生成结果文件")
        return

    has_frozen = results.get('arcface_frozen', 0) > 0

    print(f"\n解析结果:")
    print(f"  Baseline:       {results['baseline']:.4f}")
    print(f"  CLIP img:       {results['clip_img']:.4f}")
    print(f"  CLIP txt:       {results['clip_txt']:.4f}")
    print(f"  CLIP fusion:    {results['clip_fusion']:.4f} (α={results['best_alpha']})")
    print(f"  ArcFace:        {results['arcface']:.4f}")
    if has_frozen:
        print(f"  ArcFace冻结:    {results['arcface_frozen']:.4f}")
    print(f"\n输出目录: {args.output_dir}")
    print("=" * 60)

    # 生成图表
    fig_num = 1

    print(f"\n[{fig_num}/5] 主要结果对比图...")
    plot_main_results(results, os.path.join(args.output_dir, 'fig1_main_results.png'))
    fig_num += 1

    print(f"[{fig_num}/5] 消融实验曲线...")
    plot_alpha_curve(ablation_data, os.path.join(args.output_dir, 'fig2_ablation_alpha.png'))
    fig_num += 1

    print(f"[{fig_num}/5] 训练曲线 (不冻结)...")
    plot_training_curve(training_history, os.path.join(args.output_dir, 'fig3_training_curve.png'),
                        " (不冻结CLIP)")
    fig_num += 1

    if has_frozen and frozen_history:
        print(f"[{fig_num}/5] 训练曲线 (冻结CLIP)...")
        plot_training_curve(frozen_history, os.path.join(args.output_dir, 'fig4_training_curve_frozen.png'),
                            " (冻结CLIP)")
    else:
        print(f"[{fig_num}/5] 跳过冻结版训练曲线 (无数据)")
    fig_num += 1

    print(f"[{fig_num}/5] 结果汇总表...")
    plot_summary_table(results, os.path.join(args.output_dir, 'fig5_summary_table.png'))

    print("\n" + "=" * 60)
    print(f"[完成] 所有图表已保存到: {args.output_dir}/")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - fig1_main_results.png          (主要结果对比柱状图)")
    print("  - fig2_ablation_alpha.png        (消融实验α曲线)")
    print("  - fig3_training_curve.png        (训练曲线-不冻结)")
    if has_frozen:
        print("  - fig4_training_curve_frozen.png (训练曲线-冻结CLIP)")
    print("  - fig5_summary_table.png         (结果汇总表)")


if __name__ == '__main__':
    main()