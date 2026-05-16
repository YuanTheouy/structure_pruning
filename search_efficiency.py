import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==============================================================================
# 1. 采用专业论文的视觉风格
# ==============================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.7,
    'lines.linewidth': 2.5,
    'lines.markersize': 10
})

# ==============================================================================
# 2. 替换为您自己的真实数据 (REPLACE WITH YOUR DATA)
# ==============================================================================

# LLaMA-V2 7B 的 Dense PPL (用于计算奖励)
ppl_dense = 5.47

# 您的FastForward Pruning (FFP) 学习曲线数据
# X轴: 累计GPU小时
ffp_gpu_hours = np.linspace(0, 8, 20) # 示例: 您的搜索总共花了8 GPU-hours
# Y轴: 对应的PPL值 (我们会自动计算为奖励)
ffp_ppl_values = 8.5 - 2 * (1 - np.exp(-ffp_gpu_hours * 0.8)) # 示例: PPL从8.5收敛到约6.5
ffp_reward = ppl_dense / ffp_ppl_values

# 您方法最终策略的下游任务准确率
ffp_final_accuracy = 59.5 # 示例: 您的最终准确率

# ==============================================================================
# 3. SOTA 及经典对手的数据点
# ==============================================================================

# DarwinLM: (GPU-Hours, ZS Average Accuracy)
darwinlm_point = (27.6, 57.2)

# AMC: 学习曲线数据 (示例: PPL很高且不下降，导致奖励很低)
amc_gpu_hours = np.linspace(0, 8, 20)
amc_ppl_values = np.random.uniform(25, 30, 20)
amc_reward = ppl_dense / amc_ppl_values


# ==============================================================================
# 4. 绘图 (Plotting)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(8, 5.5))

# 绘制您的学习曲线 (对应左Y轴)
ax1.plot(ffp_gpu_hours, ffp_reward, color=sns.color_palette("deep")[2], label='Ours (Reward)')

# 绘制AMC的学习曲线
ax1.plot(amc_gpu_hours, amc_reward, color=sns.color_palette("deep")[3], linestyle='--', label='AMC (Reward)')

ax1.set_xlabel('Search Cost (GPU-Hours)', fontweight='bold')
ax1.set_ylabel('Reward ($PPL_{dense} / PPL$)', fontweight='bold')
ax1.tick_params(axis='y')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# 创建共享X轴的第二个Y轴 (右侧)
ax2 = ax1.twinx()

# 在右Y轴上绘制DarwinLM的最终性能点
ax2.scatter(darwinlm_point[0], darwinlm_point[1], color=sns.color_palette("deep")[1], marker='*', s=300, label='DarwinLM (Final Accuracy)', zorder=5, edgecolors='black', linewidths=0.5)

# 在右Y轴上绘制您自己方法的最终性能点
our_final_point_x = ffp_gpu_hours[-1]
ax2.scatter(our_final_point_x, ffp_final_accuracy, color=sns.color_palette("deep")[0], marker='o', s=150, label='Ours (Final Accuracy)', zorder=5, edgecolors='black', linewidths=0.5)

# # # # # # # # # # # # # # # # # # # #
# THIS IS THE CORRECTED LINE
# # # # # # # # # # # # # # # # # # # #
ax2.set_ylabel(r'Zero-Shot Average Accuracy (\%)', fontweight='bold')
# # # # # # # # # # # # # # # # # # # #

ax2.tick_params(axis='y')

# 设置Y轴的范围
min_acc = min(darwinlm_point[1], ffp_final_accuracy)
ax1.set_ylim(0, 1.0) # 奖励通常在0-1范围
ax2.set_ylim(min_acc - 2, 62)

# 统一图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# 添加标题并优化布局
plt.title('Search Efficiency on LLaMA-V2 7B', fontweight='bold', fontsize=16)
fig.tight_layout()

# 保存为高质量图片
plt.savefig('search_efficiency_llama2_7b_final_style.png', dpi=300)
plt.savefig('search_efficiency_llama2_7b_final_style.pdf', bbox_inches='tight')

print("Final styled plot saved as search_efficiency_llama2_7b_final_style.png and search_efficiency_llama2_7b_final_style.pdf")