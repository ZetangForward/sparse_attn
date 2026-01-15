import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 绘图风格设置 (ICML/NeurIPS 标准) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300

def plot_attention_matrix(matrix, title, filename, color_theme):
    """
    绘制并保存 Attention 矩阵热力图。
    上三角 Mask 区域将显示为白色（值为0）。
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 绘制热力图
    # mask=matrix==0 用于确保 0 值处完全白底，不被 colormap 最浅色干扰
    sns.heatmap(matrix, cmap=color_theme, cbar=False, 
                square=True, linecolor='white', linewidths=0.5,
                xticklabels=False, yticklabels=False, 
                vmin=0, vmax=np.max(matrix), # 确保颜色映射范围
                ax=ax)
    
    # 添加黑色边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        
    # plt.title(title, fontsize=16, pad=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, transparent=True, dpi=300)
    plt.show()

# --- 参数设置 ---
N = 24             # 序列长度 (稍微加大一点展示 sink 效果)
sink_size = 2      # Attention Sink: 前 2 个 token
window_size = 3    # Local Window: 当前 token 前后 3 个
rng = np.random.default_rng(42) # 固定随机种子

# ==========================================
# 1. 生成 Sparse Attention (Sink + Window)
# ==========================================
sparse_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(i + 1): # 只遍历下三角 (Causal)
        
        # 逻辑：是否在 Sink 区域 OR 是否在 Local Window 区域
        is_sink = (j < sink_size)
        is_window = (i - j) <= window_size
        
        if is_sink:
            # Sink 给予较高的固定关注度
            sparse_matrix[i, j] = 5.5 
        elif is_window:
            # Window 随距离衰减，模拟自然语言的局部性
            dist = i - j
            sparse_matrix[i, j] = np.exp(-0.5 * dist) + 5.2
        else:
            # 其他区域虽然在下三角，但被 Sparse 策略 Mask 掉
            sparse_matrix[i, j] = 0.0

# Row-wise Softmax 模拟 (归一化，保持上三角为0)
row_sums = sparse_matrix.sum(axis=1, keepdims=True)
sparse_matrix = np.divide(sparse_matrix, row_sums, out=np.zeros_like(sparse_matrix), where=row_sums!=0)


# ==========================================
# 2. 生成 Full Attention (Causal Lower Triangle)
# ==========================================
full_matrix = np.zeros((N, N))

for i in range(N):
    # 下三角全部填充随机值，模拟 dense attention
    # 加上一些纵向纹理，模拟某些 Key 比较重要 (Retrieval 特性)
    base_scores = rng.uniform(5.7, 10.8, size=(i + 1))
    
    # 模拟特定的几个全局 Token 被高频 Retrieve (例如第5, 12列)
    global_keys = [0,1,2,3,4,6,7,8,10, 11,13,14,15,16,17, 18,19,20,21,22,23]
    for k in global_keys:
        if k <= i:
            base_scores[k] += 5.0 # 增强这些列的信号
            
    # 增加对角线自身关注
    base_scores[i] += 2.5
    
    full_matrix[i, :i+1] = base_scores

# 模拟 Softmax (指数化后归一化)
full_matrix = np.exp(full_matrix)
# 再次强制上三角为 0 (exp(0)=1，所以需要 mask 掉上三角的 exp 结果)
mask = np.tril(np.ones((N, N)))
full_matrix = full_matrix * mask 
# 归一化
row_sums = full_matrix.sum(axis=1, keepdims=True)
full_matrix = np.divide(full_matrix, row_sums, out=np.zeros_like(full_matrix), where=row_sums!=0)


# ==========================================
# 3. 执行绘图
# ==========================================
print("正在生成 ICML 风格 Attention 示意图...")

# Sparse 使用蓝色系 (Cool / Efficient)
plot_attention_matrix(sparse_matrix, 
                      "Sparse Head\n(Sink + Local)", 
                      "viz_sparse_sink_local.png", 
                      "Blues")

# Full 使用橙色系 (Warm / Heavy Computation)
plot_attention_matrix(full_matrix, 
                      "Retrieval Head\n(Full Causal)", 
                      "viz_full_causal.png", 
                      "Oranges")

print("生成完成！")