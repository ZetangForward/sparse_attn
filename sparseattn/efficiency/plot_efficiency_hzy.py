import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import os

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
category_labels = [
    'Single-Document QA', 
    'Multi-Document QA', 
    'Summarization', 
    'Few-shot Learning', 
    'Synthetic Tasks', 
    'Code'
]
lengths = ['8K', '16K', '32K', '64K', '128K']

data_speedup = {
    'Full + Streaming':         [1.06, 1.21, 1.45, 1.83, 2.33],
    # 'Full + Xattn':             [0.96, 1.10, 1.24, 1.44, 1.64],
    'PruLong':                  [1.07, 1.19, 1.41, 1.75, 2.16],
    # 'Xattention':               [1.01, 1.17, 1.37, 1.71, 2.12],
    'MoBA':                     [0.63, 0.64, 0.78, 1.06, 1.32],
    'InfLLM-V2':                [1.00, 0.71, 0.80, 0.86, 1.17],
    'Native Sparse Attention':  [0.39, 0.44, 0.55, 0.78, 1.20]
}

data_scores = {
    'Full + Streaming':         [90.63, 84.33, 81.39, 61.87, 66.54,],      # 性能保持较好
    # 'Full + Xattn':             [92.82, 92, 87.8, 68.23, 78.87,],
    'PruLong':                  [83.54, 69.35, 56.83, 30.88, 33.74,],      # 性能严重下降示例
    # 'Xattention':               [92.96, 91.23, 89.08, 73.91, 77.69,],
    'MoBA':                     [89.05, 67.14, 30.13, 6.13, 1.15,],
    'InfLLM-V2':                [89.3, 80.93, 60.98, 35.9, 32.29,],      
    'Native Sparse Attention':  [73.27, 50.68, 21.39, 22.52, 15.3,]
}

data_sparsity = {
    'Full + Streaming':         [0.46, 0.99, 0.73, 0.97, 0.55,],      # 性能保持较好
    # 'Full + Xattn':             [90.6, 92.15, 61.11, 88.89, 31.12,],
    'PruLong':                  [0.70, 0.70, 0.70, 0.70, 0.70,],      # 性能严重下降示例
    # 'Xattention':               [89.36, 91.54, 62.43, 84.54, 31.52,],
    'MoBA':                     [0.50, 0.50, 0.50, 0.50, 0.50,],
    'InfLLM-V2':                [0.09, 0.79, 0.29, 0.15, 0.33,],      
    'Native Sparse Attention':  [0.86, 0.57, 0.53, 0.65, 0.79,]
}

# --- 3. 样式映射 (保持不变) ---
style_map = {
    'Full + Streaming': {'color': 'tab:blue',   'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 10, 'label': 'Full + Streaming (Ours)'},
    # 'Full + Xattn':     {'color': 'tab:red',    'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 9,  'label': 'Full + Xattn (Ours)'},
    # 'Xattention':       {'color': 'tab:orange', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 5,  'label': 'Xattention'},
    'PruLong':          {'color': 'tab:green',  'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 4,  'label': 'PruLong'},
    'MoBA':             {'color': 'tab:purple', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 3,  'label': 'MoBA'},
    'InfLLM-V2':        {'color': 'tab:brown',  'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 2,  'label': 'InfLLM-V2'},
    'Native Sparse Attention': {'color': 'tab:gray', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 1, 'label': 'Native Sparse Attention'}
}



# tasks = [
#     'Qasper', 'MFQA-En', 'HotpotQA', '2WikiMQA', 
#     'GovRep', 'MultiNews', 'TREC', 'TriviaQA', 
#     'SAMSum', 'Pass.Count', 'Pass.Retr', 
#     'RepoBench', 'LCC'
# ]





# --- 3. 样式映射配置  ---

# style_map = {
#     'Full + Streaming': {
#         'color': 'tab:blue',
#         'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 10, 'label': 'Full + Streaming (Ours)'
#     },
#     'Full + Xattn': {
#         'color': 'tab:red',
#         'marker': 's', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 9,  'label': 'Full + Xattn (Ours)'
#     },
#     'Xattention': {
#         'color': 'tab:orange',
#         'marker': '^', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 5,  'label': 'Xattention'
#     },
#     'PruLong': {
#         'color': 'tab:green',
#         'marker': 'D', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 4,  'label': 'PruLong'
#     },
#     'MoBA': {
#         'color': 'tab:purple',
#         'marker': 'v', 'linestyle': '--',  'linewidth': 1.2, 'zorder': 3,  'label': 'MoBA'
#     },
#     'InfLLM-V2': {
#         'color': 'tab:brown',
#         'marker': '*', 'linestyle': '--',  'linewidth': 1.2, 'zorder': 2,  'label': 'InfLLM-V2'
#     },
#     'Native Sparse Attention': {
#         'color': 'tab:gray',
#         'marker': 'X', 'linestyle': '--',   'linewidth': 1.2, 'zorder': 1,  'label': 'Native Sparse Attention'
#     }
# }


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

# def plot_pareto_frontier(x_cats, data_speed, data_score, filename):
#     fig, ax = plt.subplots(figsize=(7, 5))
    
#     # 定义 Context Length 对应的大小，长度越长，点越大
#     sizes = [50, 80, 110, 150, 200] 

#     for label in data_speed.keys():
#         if label not in style_map: continue
        
#         speed = data_speed[label]
#         score = data_score[label]
#         style = style_map[label]
        
#         # 确保数据维度对齐
#         n_points = min(len(speed), len(score))
        
#         # 画连线（表示随长度变化的轨迹）
#         ax.plot(speed[:n_points], score[:n_points], 
#                 color=style['color'], linestyle=':', alpha=0.5, linewidth=1)
        
#         # 画点
#         for i in range(n_points):
#             ax.scatter(speed[i], score[i], 
#                        s=sizes[i], # 大小代表 Context Length
#                        color=style['color'], 
#                        marker=style['marker'],
#                        edgecolor='white', linewidth=0.8,
#                        zorder=style['zorder'],
#                        label=style.get('label', label) if i == 2 else "") # 只在中间点加标签避免重复

#     # 标注 Context Length (可选，画在某个明显的模型旁边)
#     # 比如在 Full + Streaming 的点旁边标 8K, 128K...
    
#     ax.set_xlabel('Speedup (vs Full Attention)', fontsize=14, fontweight='bold')
#     ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
#     ax.set_title('Efficiency vs. Performance Trade-off', fontsize=14)
    
#     ax.grid(True, linestyle='--', alpha=0.3)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # 简单的图例
#     handles, labels = ax.get_legend_handles_labels()
#     # 去重
#     by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys(), loc='lower left', frameon=True, fontsize=9)

#     plt.tight_layout()
#     plt.savefig(filename, format='pdf', bbox_inches='tight')
#     plt.show()

# # 调用
# plot_pareto_frontier(lengths, data_speedup, data_scores, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/pareto.pdf')
# --- 5. 执行绘图 ---
# plot_bubble_scaling(lengths, data_speedup, data_scores, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speedup_score_bubble.pdf')
# plot_category_speedup(category_labels, speedup_data, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_task.pdf')
# plot_scaling_trend(lengths, data_speedup, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_length.pdf')
# plot_bar_tasks(tasks, data_tasks, '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/speed_tasks.pdf')




# --- 2. 数据准备 ---
category_labels = [
    'Single-Document QA', 
    'Multi-Document QA', 
    'Summarization', 
    'Few-shot Learning', 
    'Synthetic Tasks', 
    'Code'
]
lengths = ['8K', '16K', '32K', '64K', '128K', '256K']

data_speedup = {
    # 'Full + Streaming':         [1.06, 1.21, 1.45, 1.83, 2.33, 1],
    'Full + Xattn':             [0.96, 1.10, 1.24, 1.44, 1.64, 1],
    # 'PruLong':                  [1.07, 1.19, 1.41, 1.75, 2.16, 1],
    'Xattention':               [1.01, 1.17, 1.37, 1.71, 2.12, 1],
    'MoBA':                     [0.63, 0.64, 0.78, 1.06, 1.32, 1],
    'InfLLM-V2':                [1.00, 0.71, 0.80, 0.86, 1.17, 1],
    'Native Sparse Attention':  [0.39, 0.44, 0.55, 0.78, 1.20, 1]
}

data_scores = {
    # 'Full + Streaming':         [90.63, 84.33, 81.39, 61.87, 66.54, 53.39],      # 性能保持较好
    'Full + Xattn':             [92.82, 92, 87.8, 68.23, 78.87, 68.51],
    # 'PruLong':                  [83.54, 69.35, 56.83, 30.88, 33.74, 21.64],      # 性能严重下降示例
    'Xattention':               [92.96, 91.23, 89.08, 73.91, 77.69, 35.82],
    'MoBA':                     [89.05, 67.14, 30.13, 6.13, 1.15, 0],
    'InfLLM-V2':                [89.3, 80.93, 60.98, 35.9, 32.29, 47.27],      
    'Native Sparse Attention':  [73.27, 50.68, 21.39, 22.52, 15.3, 11.42]
}

data_sparsity = {
    # 'Full + Streaming':         [0, 0, 0, 0, 0, 0],
    'Full + Xattn':             [0, 0, 0, 0, 0, 0],
    # 'PruLong':                  [0, 0, 0, 0, 0, 0],
    'Xattention':               [0, 0, 0, 0, 0, 0],
    'MoBA':                     [0, 0, 0, 0, 0, 0],
    'InfLLM-V2':                [0, 0, 0, 0, 0, 0],
    'Native Sparse Attention':  [0, 0, 0, 0, 0, 0]
}

# --- 3. 样式映射 (保持不变) ---
style_map = {
    'Full + Streaming': {'color': 'tab:blue',   'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 10, 'label': 'Full + Streaming (Ours)'},
    'Full + Xattn':     {'color': 'tab:red',    'marker': 'o', 'linestyle': '-',  'linewidth': 1.8, 'zorder': 9,  'label': 'Full + Xattn (Ours)'},
    'Xattention':       {'color': 'tab:orange', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 5,  'label': 'Xattention'},
    'PruLong':          {'color': 'tab:green',  'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 4,  'label': 'PruLong'},
    'MoBA':             {'color': 'tab:purple', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 3,  'label': 'MoBA'},
    'InfLLM-V2':        {'color': 'tab:brown',  'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 2,  'label': 'InfLLM-V2'},
    'Native Sparse Attention': {'color': 'tab:gray', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 1, 'label': 'Native Sparse Attention'}
}

def plot_separated(x_cats, data_speed, data_score, data_sparsity, filename_base):
    """
    filename_base: 例如 '/path/to/side_by_side.pdf'
    会自动保存为:
      - /path/to/side_by_side1.pdf (Score)
      - /path/to/side_by_side2.pdf (Speedup)
      - /path/to/side_by_side3.pdf (Sparsity)
    """
    
    # 处理文件名，去掉后缀，方便拼接 1, 2, 3
    base_path, ext = os.path.splitext(filename_base)
    if not ext: 
        ext = '.pdf'
    
    x = np.arange(len(x_cats))
    
    # =======================================================
    # 图 1：Score (Bar Chart) -> 保存为 ...1.pdf
    # =======================================================
    # 调整 figsize，单张图一般用 (6, 5) 或者 (5, 4) 比较合适
    fig1, ax1 = plt.subplots(figsize=(6, 5)) 
    
    valid_models = [m for m in style_map.keys() if m in data_score]
    n_models = len(valid_models)
    total_width = 0.85          
    bar_width = total_width / n_models
    
    for i, label in enumerate(valid_models):
        if label not in data_score: continue
        values = data_score[label]
        style = style_map[label]
        
        offset = (i - (n_models - 1) / 2) * bar_width
        
        ax1.bar(x + offset, values, width=bar_width,
               color=style['color'],           
               edgecolor='white', linewidth=0.5,
               label=style.get('label', label),
               alpha=0.9,
               zorder=3)

    # ax1.set_title('(a) RULER Score', fontsize=14, pad=10)
    ax1.set_ylabel('Performance', fontweight='bold')
    ax1.set_xlabel('Context Length', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_cats)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 独立图必须加图例
    ax1.legend(loc='best', frameon=False, fontsize=10)
    
    plt.tight_layout()
    save_path1 = f"{base_path}4{ext}" #name
    plt.savefig(save_path1, format='pdf', bbox_inches='tight')
    plt.close(fig1) # 关闭图形释放内存
    print(f"Saved: {save_path1}")

    # =======================================================
    # 图 2：Speedup (Line Chart) -> 保存为 ...2.pdf
    # =======================================================
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    for label, values in data_speed.items():
        if label not in style_map: continue
        style = style_map[label]
        ax2.plot(x, values, 
                marker=style['marker'], color=style['color'], 
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style.get('label', label), zorder=style['zorder'])
    
    ax2.axhline(y=1.0, color='#444444', linestyle=':', linewidth=1.5, zorder=0)
    # ax2.set_title('(c) Efficiency Analysis', fontsize=14, pad=10)
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_xlabel('Context Length', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_cats)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(top=2.25)
    
    # 独立图必须加图例
    ax2.legend(loc='upper left', frameon=False, fontsize=10)
    
    # 原图中的注释
    ax2.text(3, 0.85, 'Full Attention Baseline', 
        fontsize=10, color='#444444', fontstyle='italic',
        ha='left', va='bottom')

    plt.tight_layout()
    save_path2 = f"{base_path}5{ext}" #name
    plt.savefig(save_path2, format='pdf', bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {save_path2}")

    # =======================================================
    # 图 3：Sparsity (Line Chart) -> 保存为 ...3.pdf
    # =======================================================
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    
    for label, values in data_sparsity.items():
        if label not in style_map: continue
        style = style_map[label]
        ax3.plot(x, values, 
                marker=style['marker'], color=style['color'], 
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style.get('label', label), # 这里加上label以便显示图例
                zorder=style['zorder'])

    # ax3.set_title('(b) Sparsity Rate', fontsize=14, pad=10)
    ax3.set_ylabel('Sparsity', fontweight='bold')
    ax3.set_xlabel('Context Length', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_cats)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, 1.1) 
    
    # 独立图必须加图例
    ax3.legend(loc='upper left', frameon=False, fontsize=10)

    plt.tight_layout()
    save_path3 = f"{base_path}6{ext}" #name
    plt.savefig(save_path3, format='pdf', bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {save_path3}")

# 调用
base_file = '/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/side_by_side.pdf'
plot_separated(lengths, data_speedup, data_scores, data_sparsity, base_file)

