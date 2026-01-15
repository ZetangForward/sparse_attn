import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. ICML 专业绘图风格设置
# ==========================================
def set_icml_style():
    """
    配置 Matplotlib 以完全符合 ICML/NeurIPS 投稿标准。
    使用 Times New Roman 字体，控制字号和线宽。
    """
    plt.rcParams.update({
        # 字体设置 (最接近 LaTeX 的渲染)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', # 数学公式字体
        
        # 字号设置 (ICML 标准: Caption=10pt, Axis=8-9pt)
        'font.size': 10,
        'axes.labelsize': 11,  # 坐标轴标题
        'axes.titlesize': 11,  # 图表标题
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        'legend.fontsize': 9,  # 图例
        
        # 布局与线条
        'axes.linewidth': 0.8, # 边框线宽
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0,
        
        # 保存设置
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05, # 减少留白
    })

# ==========================================
# 2. 数据解析 (保持不变)
# ==========================================
def parse_log_file(file_path):
    data_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "head allocate:" in line:
                    list_str = line.split("head allocate:")[-1].strip()
                    try:
                        vector_list = ast.literal_eval(list_str)
                        flat_vector = np.array(vector_list).flatten().astype(int)
                        if len(flat_vector) > 0:
                            data_lines.append(flat_vector)
                    except Exception:
                        continue
    except FileNotFoundError:
        print(f"Error: File not found.")
        return None
    return np.array(data_lines) if data_lines else None

# ==========================================
# 3. 核心绘图逻辑 (美化版)
# ==========================================
def plot_robustness_heatmap_icml(sum_matrix, num_tasks):
    """
    绘制符合 ICML 格式的鲁棒性热力图。
    """
    # ICML 双栏宽度约为 6.75 英寸，高度根据内容调整 (这里设为 3.5 英寸以保持紧凑)
    fig, ax = plt.subplots(figsize=(5.75, 3.5)) 
    
    # 归一化数据: 0.0 (Always Sparse) -> 1.0 (Always Full)
    plot_data = sum_matrix / num_tasks
    
    # 颜色映射: RdBu_r (Red-Blue Reversed)
    # 0.0 (Red) = Always Sparse (Lazy)
    # 0.5 (White) = Dynamic/Mixed
    # 1.0 (Blue) = Always Full (Active)
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    colors = ["#7DA0F9", "#F7F7F7", "#B80D28"]
    
    custom_cmap = LinearSegmentedColormap.from_list("icml_custom", colors, N=100)

    # 绘制热力图
    # rasterized=True 非常重要：防止 PDF 文件因为格点太多而变得巨大，同时保证文字是矢量可编辑的
    sns.heatmap(
        plot_data, 
        cmap=custom_cmap, 
        vmin=0, vmax=1,
        linewidths=0.2,       # 格子之间留细微白线，增加清晰度
        linecolor='white',
        cbar=True,
        ax=ax,
        rasterized=True,      # 优化 PDF 大小
        cbar_kws={
            # 'label': r'Activation Frequency ($\rho$)', # 使用 LaTeX 风格标签
            'ticks': [0, 0.5, 1],
            'shrink': 0.8,    # 缩小 colorbar 高度
            'aspect': 20,     # colorbar 变细
            'pad': 0.02,      # colorbar 离图近一点
        }
    )

    # --- 轴标签美化 ---
    # ax.set_title(r'\textbf{Consistency of Head Allocation Across Tasks}', pad=10) # 模拟加粗
    ax.set_xlabel('Head Index', labelpad=5)
    ax.set_ylabel('Layer Index', labelpad=5)
    
    # --- 刻度优化 ---
    # 32个头，每隔4个标一个数字，防止拥挤
    ax.set_xticks(np.arange(0.5, 32.5, 4))
    ax.set_xticklabels(np.arange(0, 32, 4), rotation=0)
    
    # 36层，每隔4层标一个数字
    ax.set_yticks(np.arange(0.5, 36.5, 4))
    ax.set_yticklabels(np.arange(0, 36, 4), rotation=0)

    # 修改 Colorbar 的标签文字 (Optional)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(['SA', 'Mixed', 'FA'])
    cbar.ax.tick_params(size=0) # 隐藏 colorbar 的刻度线

    # --- 边框处理 ---
    # 给图形加一个黑色细边框，显得更规整
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    plt.tight_layout()
    
    # 保存为 PDF (ICML 要求矢量图)
    save_path = 'per_task_head_robustness.pdf'
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.show()

# ==========================================
# 4. 数据处理与入口
# ==========================================
def analyze_and_plot(data, lines_per_task=36):
    total_rows, num_heads = data.shape
    num_tasks = total_rows // lines_per_task
    num_layers = lines_per_task 

    print(f"Analyzing {num_tasks} tasks...")
    
    # Reshape to (Tasks, Layers, Heads)
    try:
        tensor_data = data[:num_tasks*num_layers].reshape(num_tasks, num_layers, num_heads)
    except ValueError as e:
        print(f"Reshape error: {e}")
        return

    # Sum across tasks (axis 0)
    sum_matrix = np.sum(tensor_data, axis=0) # Shape: (Layers, Heads)
    
    # Plotting
    plot_robustness_heatmap_icml(sum_matrix, num_tasks)

if __name__ == "__main__":
    set_icml_style()
    log_path = '/data1/lcm_lab/qqt/SparseAttn/examples/output_log_per_task.txt'
    
    raw_data = parse_log_file(log_path)
    
    if raw_data is not None:
        analyze_and_plot(raw_data, lines_per_task=36)