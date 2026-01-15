import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = "/data1/lcm_lab/qqt/SparseAttn/tests/pooled_latent"
SAVE_PATH = "task_similarity_pairwise_whitening_pooled_latent.pdf"

TASK_MAP = {
    0: "S-Doc QA",
    1: "M-Hop QA",
    2: "Summ",
    3: "Code",
    4: "In-Context",
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

def load_raw_data_grouped(data_dir, target_layer):
    """
    加载原始数据，不进行白化，而是按 Task ID 分组存储。
    返回: Dict { task_id: Tensor[N, D] }
    """
    files = glob.glob(os.path.join(data_dir, f"*layer{target_layer}_*.pt"))
    if not files: return None

    print(f"Loading raw data from {len(files)} files for Layer {target_layer}...")
    
    # 临时存储
    task_data_buffer = {k: [] for k in TASK_MAP.keys()}
    
    for f in tqdm(files, desc="Reading Files"):
        try:
            data = torch.load(f, map_location="cpu")
            feats = data['pooled_hidden_states'].float() # [B, H, D]
            tasks = data['task_ids'] # [B]
            
            B, H, D = feats.shape
            
            # 展平 Head 维度
            feats_flat = feats.reshape(-1, D)
            # 展平 Task 维度
            tasks_expanded = tasks.unsqueeze(1).repeat(1, H).reshape(-1)
            
            # 按任务分流
            for t_id in TASK_MAP.keys():
                mask = (tasks_expanded == t_id)
                if mask.any():
                    task_data_buffer[t_id].append(feats_flat[mask])
                    
        except Exception as e:
            print(f"Error {f}: {e}")

    # 合并 Tensor
    final_data = {}
    for t_id, tensor_list in task_data_buffer.items():
        if tensor_list:
            final_data[t_id] = torch.cat(tensor_list, dim=0)
            print(f"Task {t_id} ({TASK_MAP[t_id]}): {final_data[t_id].shape[0]} samples")
        else:
            print(f"Warning: No data for Task {t_id}")
            final_data[t_id] = None
            
    return final_data

def compute_pairwise_matrix(data_dict):
    """
    核心逻辑：
    遍历所有任务对 (i, j)，只取出这两类数据，
    进行【局部白化】(Local Whitening)，计算余弦相似度。
    """
    task_ids = sorted(TASK_MAP.keys())
    n = len(task_ids)
    sim_matrix = np.zeros((n, n))
    
    print("\n>>> Starting Pairwise Local Whitening <<<")
    
    for idx_i, t_i in enumerate(task_ids):
        for idx_j, t_j in enumerate(task_ids):
            
            # 1. 对角线直接设为 1
            if idx_i == idx_j:
                sim_matrix[idx_i, idx_j] = 1.0
                continue
            
            # 2. 获取数据
            data_i = data_dict.get(t_i)
            data_j = data_dict.get(t_j)
            
            if data_i is None or data_j is None:
                sim_matrix[idx_i, idx_j] = 0.0
                continue
            
            # 3. 拼接数据 (只包含这两个任务!)
            # Shape: [N_i + N_j, D]
            combined_data = torch.cat([data_i, data_j], dim=0)
            
            # 4. 计算局部统计量 (Local Mean & Std)
            local_mean = combined_data.mean(dim=0, keepdim=True)
            local_std = combined_data.std(dim=0, keepdim=True)
            local_std = torch.clamp(local_std, min=1e-8) # 防止除零
            
            # 5. 执行局部白化
            # 注意：这里我们分别对 i 和 j 使用同一个 combined 的统计量进行白化
            # 这样由于去除了中心，两个任务的 Prototype 会倾向于分布在原点两侧
            data_i_whitened = (data_i - local_mean) / local_std
            data_j_whitened = (data_j - local_mean) / local_std
            
            # 6. 计算 Prototype (白化后的均值向量)
            proto_i = data_i_whitened.mean(dim=0).unsqueeze(0) # [1, D]
            proto_j = data_j_whitened.mean(dim=0).unsqueeze(0) # [1, D]
            
            # proto_i = F.normalize(proto_i, p=2, dim=1)  # [1, D]
            # proto_j = F.normalize(proto_j, p=2, dim=1)  # [1, D]
            
            
            # 7. 计算余弦相似度
            # sim = F.cosine_similarity(proto_i, proto_j).item()
            
            sim = torch.mm(proto_i, proto_j.t()).numpy()
            
            sim_normalized = (sim + 1) / 2  # [-1,1] → [0,1]
            sim_matrix[idx_i, idx_j] = sim_normalized
            
            # Optional: 打印一下看看
            if idx_i < idx_j:
                # 确保 sim 是标量
                if isinstance(sim, np.ndarray):
                    sim = sim.item()
                elif torch.is_tensor(sim):
                    sim = sim.item()

                print(f"Pair {TASK_MAP[t_i]} vs {TASK_MAP[t_j]}: Sim = {sim:.4f}")

    return sim_matrix

def sim_matrix(data_dict):
    task_ids = sorted(TASK_MAP.keys())
    n = len(task_ids)
    sim_matrix = np.zeros((n, n))
    
    print("\n>>> Starting Pairwise Local Whitening <<<")
    for idx_i, t_i in enumerate(task_ids):
        for idx_j, t_j in enumerate(task_ids):
            
            # 1. 对角线直接设为 1
            if idx_i == idx_j:
                sim_matrix[idx_i, idx_j] = 1.0
                continue

            data_i = data_dict.get(t_i).mean(dim=0).unsqueeze(0)
            data_j = data_dict.get(t_j).mean(dim=0).unsqueeze(0)
            data_i = F.normalize(data_i.float(), dim=1)
            data_j = F.normalize(data_j.float(), dim=1)
            sim = torch.mm(data_i, data_j.t()).numpy()

            sim_matrix[idx_i, idx_j] = sim
    return sim_matrix
def plot_heatmap(sim_matrix, layer_id):
    labels = [TASK_MAP[i] for i in sorted(TASK_MAP.keys())]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 局部白化后，不相似的任务往往会呈现负相关（接近-1），相似的接近1
    # 颜色映射建议使用 coolwarm (蓝=负, 红=正)
    im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    ax.set_title(f"Pairwise Contrastive Similarity\n(Local Whitening per Pair) - Layer {layer_id}", pad=15, fontsize=12)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = sim_matrix[i, j]
            
            text_color = "white" if abs(val) > 0.4 else "black"
            # 加粗字体显示
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                    color=text_color, fontsize=10, fontweight='bold')
            
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pairwise Cosine Similarity", rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✔ Heatmap saved to {SAVE_PATH}")

if __name__ == "__main__":
    target_layer = 0
    if target_layer == -1: target_layer = 31
    
    # 1. 加载原始数据 (不白化)
    raw_data = load_raw_data_grouped(DATA_DIR, target_layer)
    
    sim_matrix(raw_data)
    
    if raw_data is not None:
        # 2. 计算两两对比矩阵
        sim_mat = compute_pairwise_matrix(raw_data)
        
        # 3. 绘图
        plot_heatmap(sim_mat, target_layer)
        
        print("\nInterpretation Note:")
        print("Values close to -1.0 mean these two tasks are perfectly distinguishable in their local subspace.")
        print("Values > 0 mean they share significant structural similarities even after removing common directions.")