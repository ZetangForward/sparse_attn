import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# --- 1. ICML 风格配置 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',        # 公式字体与 Times 搭配最佳
    'font.size': 12,                   # 全局字号
    'axes.labelsize': 14,              # 轴标签字号
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.linewidth': 1.0,             # 边框略加粗
    'lines.linewidth': 2.0,            # 线条加粗
    'pdf.fonttype': 42,                # 保证 PDF 字体可编辑
})

# --- 2. 数据准备 ---
lengths = ['8K', '16K', '32K', '64K', '128K']
data_lengths = {
    'Full + Streaming': [1.06, 1.21, 1.45, 1.83, 2.33],
    'Full + Xattn':     [0.96, 1.10, 1.24, 1.44, 1.64],
    'PruLong':          [1.07, 1.19, 1.41, 1.75, 2.16],
    # 'DuoAttention':     [1.14, 1.24, 1.51, 1.84, 2.37],
    'Xattention':       [1.01, 1.17, 1.37, 1.71, 2.12],
    'MoBA':             [0.63, 0.64, 0.78, 1.06, 1.32],
    'InfLLM-V2':        [1.00, 0.71, 0.80, 0.86, 1.17],
    'Native Sparse Attention': [0.39, 0.44, 0.55, 0.78, 1.20]
}

tasks = [
    'Qasper', 'MFQA-En', 'HotpotQA', '2WikiMQA', 
    'GovRep', 'MultiNews', 'TREC', 'TriviaQA', 
    'SAMSum', 'Pass.Count', 'Pass.Retr', 
    'RepoBench', 'LCC'
]
data_tasks = {
    'Full + Streaming': [1.34, 1.25, 1.37, 1.37, 1.56, 1.58, 1.37, 1.45, 1.40, 1.36, 1.26, 1.82, 1.89],
    'Full + Xattn':     [1.15, 1.12, 1.15, 1.14, 1.26, 1.28, 1.11, 1.26, 1.16, 1.13, 1.14, 1.43, 1.47],
    'Prulong':          [1.23, 1.15, 1.17, 1.17, 1.29, 1.35, 1.17, 1.24, 1.17, 1.18, 1.15, 1.42, 1.45],
    'DuoAttention':     [1.23, 1.15, 1.17, 1.17, 1.29, 1.35, 1.17, 1.24, 1.17, 1.18, 1.15, 1.42, 1.45],
    'Xattention':       [1.25, 1.18, 1.22, 1.21, 1.34, 1.43, 1.15, 1.34, 1.22, 1.22, 1.21, 1.56, 1.61],
    
}

# --- 3. 样式映射配置  ---

style_map = {
    'Full + Streaming': {
        'color': 'tab:blue',
        'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 10, 'label': 'Full + Streaming (Ours)'
    },
    'Full + Xattn': {
        'color': 'tab:red',
        'marker': 's', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 9,  'label': 'Full + Xattn (Ours)'
    },
    'Xattention': {
        'color': 'tab:orange',
        'marker': '^', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 5,  'label': 'Xattention'
    },
    'PruLong': {
        'color': 'tab:green',
        'marker': 'D', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 4,  'label': 'PruLong'
    },
    'MoBA': {
        'color': 'tab:purple',
        'marker': 'v', 'linestyle': '--',  'linewidth': 1.2, 'zorder': 3,  'label': 'MoBA'
    },
    'InfLLM-V2': {
        'color': 'tab:brown',
        'marker': '*', 'linestyle': '--',  'linewidth': 1.2, 'zorder': 2,  'label': 'InfLLM-V2'
    },
    'Native Sparse Attention': {
        'color': 'tab:gray',
        'marker': 'X', 'linestyle': '--',   'linewidth': 1.2, 'zorder': 1,  'label': 'Native Sparse Attention'
    }
}


# --- 4. 绘图函数 ---
def plot_scaling_trend(x_cats, data, filename):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    x = np.arange(len(x_cats))
    
    # 按照 data 中的顺序遍历，或者你可以指定绘图顺序
    # 这里我们直接遍历 data，根据 style_map 取样式
    for label, values in data.items():
        if label not in style_map: continue # 防止报错
        
        style = style_map[label]
        
        ax.plot(x, values, 
                marker=style['marker'], 
                markersize=5,  
                color=style['color'], 
                linewidth=style['linewidth'], 
                linestyle=style['linestyle'],
                label=style.get('label', label), # 使用自定义标签（如果需要加 (Ours)）
                alpha=0.9,
                zorder=style['zorder'], # 关键：确保 Ours 画在最上层
                clip_on=False           # 关键：防止标记在边缘被截断
        )

    # 强化 Baseline (y=1.0)
    ax.axhline(y=1.0, color='#444444', linestyle='--', linewidth=1.2, zorder=0)
    # 调整文字位置，避免遮挡
    ax.text(3, 0.92, 'Full Attention Baseline', 
        fontsize=10, color='#444444', fontstyle='italic',
        ha='left', va='bottom')

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(x_cats)
    ax.set_xlabel('Context Length', fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup', fontweight='bold', labelpad=8)
    
    # 网格线优化：只留横向，且颜色淡一点
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray')
    
    # 边框美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 图例优化
    # 将图例分为两行：上面放 Ours，下面放 Baselines，或者横排
    # 这里通过 ncol=2 或 3 来控制布局
    legend = ax.legend(frameon=False, loc='upper left', ncol=1,)
    
    plt.tight_layout()
    # 建议保存为 PDF 以获得最佳质量
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

# --- 4. 绘图函数：柱状图 (针对 Tasks) ---
# 优势：解决横坐标密集问题，放大 Y 轴差异
def plot_bar_tasks(x_cats, data, filename):
    fig, ax = plt.subplots(figsize=(14, 4.5)) # 宽图
    
    x = np.arange(len(x_cats))
    width = 0.16 # 柱子稍微窄一点以留出空隙
    num_bars = len(data)
    total_width = num_bars * width
    start = -total_width/2 + width/2
    
    hatches = ['', '///', '\\\\', 'xx', '..'] # 纹理

    for i, (label, values) in enumerate(data.items()):
        offset = start + i * width
        rects = ax.bar(x + offset, values, width, 
                       label=label, 
                       color=colors[i], 
                       edgecolor='black', 
                       linewidth=0.6,
                       hatch=hatches[i],
                       alpha=0.95,
                       zorder=3)

    # 调整 Y 轴显示范围 (关键：解决 1.0 看不明显的问题)
    # 从 0.9 或 0.95 开始，而不是 0
    ax.set_ylim(0.95, 2.05) 
    
    # 明显的基准线
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, zorder=2)
    
    # 网格线放到底层
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_cats, rotation=25, ha='right', fontsize=11)
    
    ax.set_ylabel('Normalized Speedup', fontweight='bold')
    ax.set_xlabel('LongBench Subtasks', fontweight='bold')
    
    # 去除多余边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例横排放在顶部
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=5, fontsize=11, handlelength=1.5, columnspacing=1.5)

    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

# --- 5. 执行绘图 ---
plot_scaling_trend(lengths, data_lengths, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_length.pdf')
# plot_bar_tasks(tasks, data_tasks, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_tasks.pdf')