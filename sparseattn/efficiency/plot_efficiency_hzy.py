import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

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
    'Full + Streaming':         [1.06, 1.21, 1.45, 1.83, 2.33],
    'Full + Xattn':             [0.96, 1.10, 1.24, 1.44, 1.64],
    'PruLong':                  [1.07, 1.19, 1.41, 1.75, 2.16],
    'Xattention':               [1.01, 1.17, 1.37, 1.71, 2.12],
    'MoBA':                     [0.63, 0.64, 0.78, 1.06, 1.32],
    'InfLLM-V2':                [1.00, 0.71, 0.80, 0.86, 1.17],
    'Native Sparse Attention':  [0.39, 0.44, 0.55, 0.78, 1.20]
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
    
# --- [新增] 5. 数据准备：Speedup Prefill 分类统计 ---
# 注意：这里的数据是示例占位符，请替换为你真实的 Speedup 数值
category_labels = [
    'Single-Document QA', 
    'Multi-Document QA', 
    'Summarization', 
    'Few-shot Learning', 
    'Synthetic Tasks', 
    'Code'
]


# --- [新增] 6. 绘图函数：类别分组柱状图 ---
def plot_category_speedup(x_cats, data, filename):
    # 调整画布大小，宽一点以容纳分组
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 1. 计算柱状图位置
    # 筛选出在 style_map 中存在且在 data 中有数据的模型
    valid_models = [m for m in style_map.keys() if m in data]
    n_models = len(valid_models)
    x = np.arange(len(x_cats))
    
    # 设置总宽度和单个柱子的宽度
    total_width = 0.85          # 一组柱子占用的总宽度比例
    bar_width = total_width / n_models
    
    # 2. 循环绘制
    # enumerate 自动提供索引 i，用于计算偏移量
    for i, model_name in enumerate(valid_models):
        style = style_map[model_name]
        values = data[model_name]
        
        # 计算偏移量：让这组柱子整体居中
        # 公式推导：(i - 中间位置) * 宽度
        offset = (i - (n_models - 1) / 2) * bar_width
        
        ax.bar(x + offset, values, width=bar_width,
               color=style['color'],           # 核心：使用 style_map 的颜色
               label=style.get('label', model_name), # 使用 style_map 的标签
               edgecolor='black',              # 加上黑边增加对比度
               linewidth=0.6,
               alpha=0.9,                      # 轻微透明度增加质感
               zorder=3)                       # 确保图层在网格之上

    # 3. 装饰与美化
    # Y轴设置
    ax.set_ylabel('Speedup', fontweight='bold', labelpad=10)
    ax.set_ylim(0.0, 1.8)  
    
    # 基准线 (Baseline = 1.0)
    ax.axhline(y=1.0, color='#444444', linestyle='--', linewidth=1.5, zorder=2)
    
    # X轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(x_cats, rotation=15, ha='right', fontsize=11) # 稍微倾斜避免重叠
    # ax.set_xlabel('Tasks', fontweight='bold') # 横轴标签可选

    # 网格线 (仅 Y 轴)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例设置
    # 分列显示，放在顶部外侧或内部上方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=4, frameon=False, fontsize=9, 
              handlelength=1.5, columnspacing=1.2)
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()


data_scores = {
    'Full + Streaming':         [60, 59, 58, 57, 56,],      # 性能保持较好
    'Full + Xattn':             [60, 59, 58, 58, 57,],
    'PruLong':                  [58, 55, 50, 40, 20,],      # 性能严重下降示例
    'Xattention':               [59, 58, 57, 55, 52,],
    'MoBA':                     [55, 50, 45, 40, 35,],
    'InfLLM-V2':                [50, 48, 45, 42, 10,],      
    'Native Sparse Attention':  [40, 35, 30, 20, 10,]
}

# --- 3. 样式映射 (保持不变) ---
style_map = {
    'Full + Streaming': {'color': 'tab:blue',   'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 10, 'label': 'Full + Streaming (Ours)'},
    'Full + Xattn':     {'color': 'tab:red',    'marker': 's', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 9,  'label': 'Full + Xattn (Ours)'},
    'Xattention':       {'color': 'tab:orange', 'marker': '^', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 5,  'label': 'Xattention'},
    'PruLong':          {'color': 'tab:green',  'marker': 'D', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 4,  'label': 'PruLong'},
    'MoBA':             {'color': 'tab:purple', 'marker': 'v', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 3,  'label': 'MoBA'},
    'InfLLM-V2':        {'color': 'tab:brown',  'marker': '*', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 2,  'label': 'InfLLM-V2'},
    'Native Sparse Attention': {'color': 'tab:gray', 'marker': 'X', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 1, 'label': 'Native Sparse Attention'}
}

# --- 4. 核心绘图函数：气泡折线图 ---
def plot_bubble_scaling(x_cats, y_data, score_data, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(x_cats))

    # 1. 预处理分数以控制气泡大小
    # 将分数归一化并映射到合适的像素面积范围 (例如 20 到 200)
    all_scores = []
    for v in score_data.values(): all_scores.extend(v)
    min_score, max_score = min(all_scores), max(all_scores)
    
    def get_bubble_size(score):
        # 线性映射：(val - min)/(max - min)
        # 加上基数 30 保证最小的点也能看见，乘以 150 拉大差异
        norm = (score - min_score) / (max_score - min_score + 1e-6)
        return 30 + norm * 180 

    # 2. 循环绘图
    for label, speedups in y_data.items():
        if label not in style_map: continue
        style = style_map[label]
        scores = score_data.get(label, [min_score]*len(speedups)) # 防御性编程
        
        # (A) 画线 (不带 Marker，或者 Marker 很小)
        ax.plot(x, speedups, 
                color=style['color'], 
                linestyle=style['linestyle'], 
                linewidth=style['linewidth'],
                zorder=style['zorder'],
                label=style.get('label', label), # 用于图例颜色
                alpha=0.8)

        # (B) 画气泡 (Scatter) 覆盖在线上
        sizes = [get_bubble_size(s) for s in scores]
        ax.scatter(x, speedups, 
                   s=sizes, 
                   color=style['color'], 
                   marker=style['marker'], # 保持原来的形状
                   edgecolor='white',      # 加白边增加区分度
                   linewidth=0.8,
                   zorder=style['zorder'] + 0.1, # 稍微浮在上面
                   clip_on=False)

    # 3. 装饰
    ax.axhline(y=1.0, color='#444444', linestyle='--', linewidth=1.2, zorder=0)
    ax.text(0, 0.92, 'Baseline (Full Attn)', fontsize=10, color='#444444', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(x_cats)
    ax.set_xlabel('Context Length', fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup', fontweight='bold', labelpad=8)
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 4. 复杂的双图例制作 ---
    # 第一部分：模型颜色图例
    # 我们获取当前的 handle 和 label
    handles, labels = ax.get_legend_handles_labels()
    # 重新创建 Line2D 对象以确保图例里的图标大小一致，不受散点图大小影响
    leg_handles = []
    for h, l in zip(handles, labels):
        # 找到对应的 style
        key = [k for k, v in style_map.items() if v.get('label', k) == l][0]
        leg_handles.append(mlines.Line2D([], [], color=style_map[key]['color'], 
                                         marker=style_map[key]['marker'],
                                         linestyle=style_map[key]['linestyle'],
                                         markersize=6, label=l))
    
    first_legend = ax.legend(handles=leg_handles, loc='upper left', 
                             frameon=False, ncol=1, fontsize=9)
    ax.add_artist(first_legend) # 这一步很关键，因为下面要加第二个图例

    # 第二部分：气泡大小图例 (Size Legend)
    # 手动创建几个假点来解释大小含义
    score_levels = [min_score, (min_score+max_score)/2, max_score]
    size_labels = [f'Score: {int(s)}' for s in score_levels]
    size_handles = []
    for s, l in zip(score_levels, size_labels):
        size_handles.append(
            plt.scatter([], [], s=get_bubble_size(s), 
                        color='gray', edgecolors='none', alpha=0.6, label=l)
        )
    
    # 将大小图例放在右下角或其他空位
    ax.legend(handles=size_handles, loc='lower right', 
              title="Metric Performance", 
              frameon=True, framealpha=0.8,
              handletextpad=0.5, labelspacing=0.8, borderpad=0.8,
              fontsize=8, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()


# --- 5. 执行绘图 ---
plot_bubble_scaling(lengths, data_lengths, data_scores, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speedup_score_bubble.pdf')
# plot_category_speedup(category_labels, speedup_data, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_task.pdf')
# plot_scaling_trend(lengths, data_lengths, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_length.pdf')
# plot_bar_tasks(tasks, data_tasks, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_tasks.pdf')