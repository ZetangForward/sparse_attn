import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------- 1. 绘图风格设置 (ICML/NeurIPS 风格) ----------------
# 如果安装了 seaborn，利用它做底层优化，否则使用原生 matplotlib
try:
    import seaborn as sns
    sns.set_theme(style="ticks")
except ImportError:
    pass

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 论文标准字体
    'mathtext.fontset': 'stix',         # 数学公式字体
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42                  # 确保字体嵌入，可编辑
})

# ---------------- 2. 模型定义 (保持不变) ----------------
class InferenceRouter(nn.Module):
    def __init__(self, d_feature=128, num_kv=32, use_softmax=True):
        super().__init__()
        self.num_kv = num_kv
        self.use_softmax = use_softmax
        self.tau = 1.0

        self.cls_feat_extractor = nn.Sequential(
            nn.Linear(d_feature, 4 * d_feature),
            nn.SiLU(),
            nn.Linear(4 * d_feature, d_feature),
        )

        if self.use_softmax:
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(d_feature, 2 * d_feature),
                nn.SiLU(),
                nn.Linear(2 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1),
            )

    def forward(self, x, cu_seq_len=None):
        if cu_seq_len is not None:
            x_s, x_e = cu_seq_len[0], cu_seq_len[1]
            # 这里的逻辑似乎有点问题，假设 cu_seq_len 是 batch 内的切片
            # 简单起见，这里假设 x 已经是切好的 tensor
            pooled_latent = x.mean(dim=0).unsqueeze(0) 
        else:
            s_len = x.shape[1]
            # --- 关键逻辑：截断处理 ---
            if s_len > 200:
                target = torch.cat([x[:, :100, :], x[:, -100:, :]], dim=1).mean(dim=1)
            else:
                target = x.mean(dim=1)
            pooled_latent = target

        pooled_hidden_states = self.cls_feat_extractor(pooled_latent)
        binary_logits = self.cls_router_head_agnostic(pooled_hidden_states)

        if self.use_softmax:
            z_soft = F.softmax(binary_logits, dim=-1)
            decisions = z_soft[..., 1]
        else:
            decisions = torch.sigmoid(binary_logits / self.tau)

        return decisions


# ---------------- 3. Benchmark 逻辑 ----------------
def run_timing_loop(model, inputs, device, test_iters=100): # 减少iters加快演示
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(*inputs)

    # Timing
    if device == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(*inputs)
        ender.record()
        torch.cuda.synchronize()
        total_time_ms = starter.elapsed_time(ender)
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(*inputs)
        total_time_ms = (time.time() - start_time) * 1000

    return total_time_ms / test_iters

# ---------------- 4. 绘图逻辑 (重写) ----------------
def format_func(value, tick_number):
    """自定义X轴刻度格式化: 512, 1K, 2K, 1M..."""
    if value < 1024:
        return f"{int(value)}"
    elif value < 1024**2:
        return f"{int(value/1024)}K"
    else:
        return f"{int(value/1024**2)}M"

def plot_benchmark_results(seq_lengths, times, device_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 颜色：学术蓝
    color_main = '#1f77b4' 
    
    # 绘制主线
    ax.plot(
        seq_lengths,
        times,
        marker="o",
        markersize=7,
        linestyle="-",
        color=color_main,
        label="Router Latency",
        zorder=3,
        clip_on=False # 防止标记点在边缘被切掉
    )

    # ---- X轴处理 (Log Scale & Custom Ticks) ----
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length", fontweight='bold')
    
    # 设置自定义刻度显示
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    # 强制显示我们数据点对应的刻度
    ax.set_xticks(seq_lengths)
    
    # ---- Y轴处理 ----
    ax.set_ylabel("Inference Latency (ms)", fontweight='bold')
    # 自动调整Y轴范围，让数据居中，不要紧贴上下边缘
    y_min, y_max = min(times), max(times)
    margin = (y_max - y_min) * 0.5 if y_max != y_min else y_min * 0.1
    # 如果数据极其平稳（margin很小），给一个最小的显示范围
    if margin < 0.01: margin = 0.05
    ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # ---- 网格与边框 ----
    ax.grid(True, which="major", axis='y', ls="--", color='gray', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # ---- 图例 ----
    ax.legend(frameon=False, loc='best')

    # ---- 标注 (Annotations) ----
    # 在论文中，不需要标注每个点，通常只标注 Trend 或者 Key Point。
    # 这里我们只标注平均值，或者如果不平稳，只标注首尾。
    avg_latency = np.mean(times)
    ax.text(
        0.05, 0.95, 
        f"Avg Latency: {avg_latency:.3f} ms", 
        transform=ax.transAxes, 
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
    )

    # 保存
    output_filename = f"{device_name}_latency.pdf"
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf')
    print(f"Figure saved as {output_filename}")
    plt.show()

# ---------------- 5. 主程序 ----------------
def visualize_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用 2的幂次，方便 Log2 坐标轴展示
    seq_lengths = [2**i for i in range(9, 21)] # 512 到 1M
    
    d_feat = 128
    num_heads = 32

    standard_times = []
    
    print(f"Initializing model on {device}...")
    model = InferenceRouter(d_feature=d_feat, num_kv=num_heads).to(device).eval()

    print(f"Starting benchmark...")
    # 预先生成所有输入数据，避免把数据生成时间计入（虽然之前也没计入，但这样内存可能爆）
    # 考虑到 1M 长度可能显存不够，我们还是随用随生成，但要注意 timing 范围
    
    for s_len in seq_lengths:
        try:
            # 构造输入
            x = torch.randn(1, s_len, num_heads, d_feat, device=device)
            
            # 计时
            t_std = run_timing_loop(model, (x,), device, test_iters=100)
            standard_times.append(t_std)
            print(f"SeqLen {format_func(s_len, None):>5s}: {t_std:.4f}ms")
            
            # 清理缓存，防止 OOM
            del x
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"OOM or Error at len {s_len}: {e}")
            break

    # 截断 seq_lengths 以匹配成功运行的数据
    valid_lengths = seq_lengths[:len(standard_times)]
    
    if valid_lengths:
        plot_benchmark_results(valid_lengths, standard_times, device)

if __name__ == "__main__":
    visualize_benchmark()