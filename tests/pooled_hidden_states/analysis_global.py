import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = "/data1/lcm_lab/qqt/SparseAttn/tests/pooled_hidden_states2"
SAVE_PATH = "task_similarity_whitened.pdf"

TASK_MAP = {
    0: "Single QA",
    1: "MultiHop QA",
    2: "Summarization",
    3: "Code",
    4: "In-Context Learning",
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def get_max_layer(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    max_layer = -1
    for f in files:
        match = re.search(r"layer(\d+)_", os.path.basename(f))
        if match:
            lid = int(match.group(1))
            max_layer = max(max_layer, lid)
    return max_layer

def load_and_whiten_data(data_dir, target_layer):
    """
    加载数据并执行【特征白化】(Feature Whitening)
    解决 Rogue Dimensions 导致的相似度坍缩问题
    """
    files = glob.glob(os.path.join(data_dir, f"*layer{target_layer}_*.pt"))
    if not files: return None

    print(f"Loading {len(files)} files for Layer {target_layer}...")
    
    # 1. 收集所有样本到同一个大矩阵中
    all_feats_list = []
    all_tasks_list = []
    
    for f in tqdm(files):
        try:
            data = torch.load(f, map_location="cpu")
            feats = data['pooled_hidden_states'].float() # [B, H, D]
            tasks = data['task_ids'] # [B]
            
            B, H, D = feats.shape
            feats_flat = feats.reshape(-1, D)
            tasks_expanded = tasks.unsqueeze(1).repeat(1, H).reshape(-1)
            
            all_feats_list.append(feats_flat)
            all_tasks_list.append(tasks_expanded)
            
        except Exception as e:
            print(f"Error {f}: {e}")

    if not all_feats_list: return None

    # 合并为大矩阵 [Total_Samples, D]
    X = torch.cat(all_feats_list, dim=0)
    Y = torch.cat(all_tasks_list, dim=0)
    
    print(f"Total samples: {X.shape[0]}, Dimensions: {X.shape[1]}")

    # ================= 核心操作：特征白化 (Z-Score) =================
    # 计算每一列（每个特征维度）的均值和标准差
    # dim=0 表示沿着样本方向计算，得到 [D] 大小的 mean/std
    feat_mean = X.mean(dim=0, keepdim=True)
    feat_std = X.std(dim=0, keepdim=True)
    
    # 避免除以 0
    feat_std = torch.clamp(feat_std, min=1e-8)
    
    # 执行标准化：(X - mean) / std
    # 这会消除 Rogue Dimensions 的统治力
    X_whitened = (X - feat_mean) / feat_std
    print(">>> Applied Feature-wise Whitening (Z-Score) <<<")
    # =============================================================

    # 2. 计算每个任务的 Prototype (基于白化后的特征)
    prototypes = []
    valid_ids = sorted(TASK_MAP.keys())
    
    for t_id in valid_ids:
        mask = (Y == t_id)
        if mask.sum() > 0:
            # 取该任务所有样本的平均
            task_proto = X_whitened[mask].mean(dim=0)
            prototypes.append(task_proto)
        else:
            prototypes.append(torch.zeros(D))
            
    return torch.stack(prototypes)

def compute_cosine_sim(prototypes):
    # 输入已经是白化过的 Prototypes [4, D]
    # 直接计算 Cosine Similarity
    P_norm = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.mm(P_norm, P_norm.t()).numpy()
    return sim_matrix

def plot_heatmap(sim_matrix, layer_id):
    labels = [TASK_MAP[i] for i in sorted(TASK_MAP.keys())]
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    # 使用 coolwarm，这回可以看到真实的差异了
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    ax.set_title(f"Task Similarity (Whitened Features)\nLayer {layer_id}", pad=15)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = sim_matrix[i, j]
            # 对角线可能稍微小于1 (由于数值精度)，强制显示为1
            if i == j: val = 1.0
            
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                    color=text_color, fontsize=10, fontweight='bold')
            
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Cosine Similarity (Whitened)", rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✔ Heatmap saved to {SAVE_PATH}")
    print("\n--- Whitened Similarity Matrix ---")
    print(sim_matrix)

if __name__ == "__main__":
    target_layer = 0
    if target_layer == -1: target_layer = 31
    
    # 1. 加载并白化
    protos = load_and_whiten_data(DATA_DIR, target_layer)
    
    if protos is not None:
        # 2. 计算相似度
        sim_mat = compute_cosine_sim(protos)
        # 3. 绘图
        plot_heatmap(sim_mat, target_layer)
