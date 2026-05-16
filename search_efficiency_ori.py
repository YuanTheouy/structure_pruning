#!/usr/bin/env python3

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches

def load_dataset(pkl_file):
    """加载pkl数据集文件"""
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"文件未找到: {pkl_file}")
    
    print(f"正在加载数据集: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 统一转换为多动作格式
    if isinstance(data, dict):
        if "root_trans_offset" in data:
            # 旧格式单动作数据
            filename = os.path.splitext(os.path.basename(pkl_file))[0]
            data = {filename: data}
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    return data

def collect_x_axis_velocities(motion_data):
    """收集所有动作序列的X轴速度数据"""
    all_x_velocities = []
    motion_labels = []
    
    print(f"\n正在从 {len(motion_data)} 个动作序列中收集数据:")
    
    for motion_key, motion in motion_data.items():
        if "base_lin_vel_local_50window" not in motion or motion["base_lin_vel_local_50window"] is None:
            print(f"  跳过 '{motion_key}': 无 base_lin_vel_local_50window 数据")
            continue
        
        base_lin_vel_local_50window = motion["base_lin_vel_local_50window"]
        n_frames = len(base_lin_vel_local_50window)
        
        print(f"  加载 '{motion_key}': {n_frames} 帧")
        
        # 只收集X轴数据（索引0）
        x_velocities = base_lin_vel_local_50window[:, 0]
        all_x_velocities.append(x_velocities)
        motion_labels.extend([motion_key] * n_frames)
    
    if not all_x_velocities:
        raise ValueError("未找到包含 base_lin_vel_local_50window 数据的动作序列")
    
    # 合并所有数据
    all_x_velocities = np.concatenate(all_x_velocities)
    motion_labels = np.array(motion_labels)
    
    print(f"\n总共收集了 {len(all_x_velocities)} 帧的数据")
    return all_x_velocities, motion_labels

def plot_x_velocity_distribution_for_paper(x_velocities, motion_labels, save_path=None, show_plot=True, reduce_range=None, reduce_ratio=0.2):
    """绘制适合论文使用的X轴速度分布图
    
    Args:
        reduce_range: 要削减的速度范围，例如 (-0.1, 0.1) 表示削减 -0.1 到 0.1 m/s 范围内的数据
        reduce_ratio: 削减比例，例如 0.2 表示删除该范围内20%的数据点
    """
    
    # 如果指定了削减范围，先处理数据
    if reduce_range is not None:
        min_vel, max_vel = reduce_range
        
        # 找到在指定范围内的数据点索引
        in_range_mask = (x_velocities >= min_vel) & (x_velocities <= max_vel)
        in_range_indices = np.where(in_range_mask)[0]
        
        print(f"在范围 [{min_vel}, {max_vel}] m/s 内找到 {len(in_range_indices)} 个数据点")
        
        # 随机选择要删除的数据点
        np.random.seed(42)  # 固定随机种子确保可重复
        n_to_remove = int(len(in_range_indices) * reduce_ratio)
        indices_to_remove = np.random.choice(in_range_indices, n_to_remove, replace=False)
        
        print(f"将删除 {n_to_remove} 个数据点 ({reduce_ratio*100:.1f}%)")
        
        # 创建保留数据的掩码
        keep_mask = np.ones(len(x_velocities), dtype=bool)
        keep_mask[indices_to_remove] = False
        
        # 更新数据
        x_velocities = x_velocities[keep_mask]
        motion_labels = motion_labels[keep_mask]
        
        print(f"削减后剩余 {len(x_velocities)} 个数据点")
    
    # 设置论文级别的样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 绘制总体分布（青色）- 显示百分比
    weights = np.ones_like(x_velocities) / len(x_velocities) * 100
    n, bins, patches = ax.hist(x_velocities, bins=50, alpha=0.7, color='#17A2B8', 
                              weights=weights, label='X-axis Velocity Distribution')
    
    # 添加统计信息
    mean_val = np.mean(x_velocities)
    
    # 只添加均值线（红色虚线）
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, alpha=0.8, 
               label=f'Mean: {mean_val:.3f} m/s')
    
    # 设置标题和标签
    ax.set_title('X-axis Velocity Distribution', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X-axis Velocity (m/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    
    # 美化网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_facecolor('white')
    
    # 设置图例 - 显示分布和均值
    legend_elements = [
        mpatches.Patch(color='#17A2B8', alpha=0.7, label='X-axis Velocity Distribution'),
        plt.Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label=f'Mean: {mean_val:.3f} m/s')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True,
             facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 保存高质量图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='png', pad_inches=0.2)
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='pdf', pad_inches=0.2)
        
        print(f"图形已保存至: {save_path}")
        print(f"PDF格式已保存至: {save_path.replace('.png', '.pdf')}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def print_x_velocity_statistics(x_velocities, motion_labels):
    """打印X轴速度的详细统计信息"""
    print("\n" + "="*80)
    print("X轴速度分布统计分析")
    print("="*80)
    
    print(f"\n[总体统计]")
    print(f"总数据点: {len(x_velocities):,}")
    print(f"动作数量: {len(np.unique(motion_labels))}")
    print(f"平均值: {np.mean(x_velocities):+.6f} m/s")
    print(f"中位数: {np.median(x_velocities):+.6f} m/s")
    print(f"标准差: {np.std(x_velocities):.6f} m/s")
    print(f"最小值: {np.min(x_velocities):+.6f} m/s")
    print(f"最大值: {np.max(x_velocities):+.6f} m/s")
    print(f"25%分位数: {np.percentile(x_velocities, 25):+.6f} m/s")
    print(f"75%分位数: {np.percentile(x_velocities, 75):+.6f} m/s")
    print(f"95%置信区间: [{np.percentile(x_velocities, 2.5):+.6f}, {np.percentile(x_velocities, 97.5):+.6f}] m/s")
    
    # 正态性检验
    try:
        _, p_value = stats.normaltest(x_velocities)
        print(f"正态性检验 p值: {p_value:.6f} ({'正态分布' if p_value > 0.05 else '非正态分布'})")
    except:
        print("正态性检验失败")
    
    # 各动作的统计
    print(f"\n[各动作统计]")
    unique_motions = np.unique(motion_labels)
    
    for motion in unique_motions:
        mask = motion_labels == motion
        motion_x_data = x_velocities[mask]
        if len(motion_x_data) == 0:
            continue
            
        print(f"\n  动作: {motion}")
        print(f"    数据点: {len(motion_x_data):,}")
        print(f"    持续时间: {len(motion_x_data)/30:.2f}s (假设30fps)")
        print(f"    平均值: {np.mean(motion_x_data):+.6f} m/s")
        print(f"    标准差: {np.std(motion_x_data):.6f} m/s")
        print(f"    范围: [{np.min(motion_x_data):+.6f}, {np.max(motion_x_data):+.6f}] m/s")

def main():
    parser = argparse.ArgumentParser(description='绘制用于论文的X轴速度分布图')
    parser.add_argument('pkl_file', help='输入pkl文件路径')
    parser.add_argument('--save', '-s', type=str, default=None, 
                       help='保存图像的文件路径（建议使用.png扩展名）')
    parser.add_argument('--no-plot', action='store_true', 
                       help='只输出统计信息，不显示图表')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图表，仅保存文件')
    parser.add_argument('--reduce-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='削减指定速度范围内的数据，格式：--reduce-range -0.1 0.1')
    parser.add_argument('--reduce-ratio', type=float, default=0.2,
                       help='削减比例 (0.0-1.0)，默认0.2 (20%%)')
    
    args = parser.parse_args()
    
    try:
        # 加载数据集
        motion_data = load_dataset(args.pkl_file)
        
        # 收集X轴速度数据
        x_velocities, motion_labels = collect_x_axis_velocities(motion_data)
        
        # 打印统计信息
        print_x_velocity_statistics(x_velocities, motion_labels)
        
        # 绘制分布图
        if not args.no_plot:
            # 默认保存路径
            if args.save is None and not args.no_show:
                save_path = os.path.join(os.path.dirname(args.pkl_file), 
                                       'x_velocity_distribution_paper.png')
            else:
                save_path = args.save
            
            plot_x_velocity_distribution_for_paper(
                x_velocities, motion_labels, 
                save_path=save_path,
                show_plot=not args.no_show,
                reduce_range=args.reduce_range,
                reduce_ratio=args.reduce_ratio
            )
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 