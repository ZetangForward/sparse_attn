import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ========================
# 加载结果
# ========================
data = torch.load(
    "/data/lcm_lab/qqt/project/SparseAttn/erank_analysis_results.pt", map_location="cpu"
)
avg_erank = data["avg_erank"]  # [L, H]
head_groups = data["head_groups"]  # [L, H]
layer_boundaries = data["layer_boundaries"]
retrieval_layers = data["retrieval_layers"]

L, H = avg_erank.shape
E = avg_erank.mean(dim=1).numpy()

# ========================
# 1️⃣ 层-头热力图
# ========================
plt.figure(figsize=(10, 6))
sns.heatmap(
    avg_erank.numpy(),
    cmap="viridis",
    cbar_kws={"label": "Truncated Effective Rank"},
)
plt.title("Per-layer Per-head Effective Rank Heatmap")
plt.xlabel("Attention Head Index")
plt.ylabel("Layer Index")
plt.tight_layout()
plt.savefig("per-layer_per-head_effective_rank.pdf")

# ========================
# 2️⃣ 层平均有效秩趋势 + 标注层分界与检索层
# ========================
plt.figure(figsize=(10, 4))
plt.plot(range(L), E, label="Mean Effective Rank", lw=2)
plt.scatter(
    retrieval_layers,
    [E[i] for i in retrieval_layers],
    color="red",
    s=80,
    marker="*",
    label="Candidate Retrieval Layers",
)

for b in layer_boundaries:
    plt.axvline(x=b, color="orange", linestyle="--", alpha=0.6)

plt.title("Layer-wise Mean Effective Rank")
plt.xlabel("Layer Index")
plt.ylabel("Mean ERank")
plt.legend()
plt.tight_layout()
plt.savefig("mean_effective_rank.pdf")

# ========================
# 3️⃣ 头分组可视化
# ========================
plt.figure(figsize=(10, 6))
sns.heatmap(
    head_groups.numpy(),
    cmap="Spectral_r",
    cbar_kws={"label": "Head Group (Low→High Importance)"},
)
plt.title("Head Group Assignment per Layer")
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.tight_layout()
plt.savefig("head_group_assignment.pdf")
