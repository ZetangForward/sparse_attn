# 🚀 SparseAttn

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)

**高性能稀疏注意力机制库 - 为大规模语言模型提供高效的注意力计算**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [API 文档](#-api-文档) • [性能基准](#-性能基准) • [贡献指南](#-贡献指南)

</div>

## 📖 简介

SparseAttn 是一个专为大规模语言模型设计的高性能稀疏注意力库。通过先进的稀疏化技术和 GPU 优化，显著降低注意力计算的内存消耗和计算复杂度，同时保持模型性能。

### 🎯 主要优势

- **🔥 高性能**: 基于 Triton 的自定义 CUDA 内核，提供极致性能
- **💾 内存高效**: 相比传统注意力机制，内存使用量减少 80%+
- **🎛️ 灵活配置**: 支持多种稀疏化策略和参数调优
- **🔧 易于集成**: 与 Transformers 生态系统无缝集成
- **📊 多种模式**: 支持预填充和解码阶段的不同优化策略

## ✨ 功能特性

### 🏗️ 核心组件

#### 1. **Xattention** - 自适应稀疏注意力
- 基于阈值的动态稀疏化
- 支持因果和非因果注意力
- 高度优化的 Triton 内核实现

#### 2. **FlexPrefill** - 灵活预填充策略
- 块级稀疏注意力
- 自适应块选择算法
- 长序列的高效处理

#### 3. **Minference** - 轻量级推理
- 垂直和对角稀疏模式
- 自适应预算分配
- 专为推理阶段优化

#### 4. **FullPrefill** - 完整预填充
- 基于 FlashInfer 的高效实现
- 支持自定义掩码
- 内存和计算的双重优化

### 🏋️ 训练支持
- **分布式训练**: 支持多GPU和多节点训练，具备序列并行能力
- **稀疏微调**: 支持在语言模型上训练稀疏注意力模式的方法
- **灵活稀疏控制**: 可配置的稀疏比例和模式
- **掩码学习**: 专门训练注意力掩码，支持独立学习率
- **正则化技术**: 支持多种稀疏性控制的正则化方法

### 🔧 技术特性

- **🎯 智能稀疏化**: 基于注意力分数的自适应稀疏模式
- **⚡ GPU 加速**: 完全基于 CUDA 的高性能实现
- **🧩 模块化设计**: 可插拔的注意力组件
- **📈 可扩展性**: 支持从小型到超大型模型的各种规模
- **🔒 数值稳定性**: 精心设计的数值计算，确保训练稳定性

## 📁 项目结构

```
SparseAttn/
├── sparseattn/              # 主包
│   ├── __init__.py          # 包初始化
│   ├── training/            # 稀疏注意力训练模块
│   ├── threshold/           # 基于阈值的稀疏注意力模块
│   ├── run_scripts/         # 训练和评估脚本
│   └── src/                 # 核心源代码
│       ├── __init__.py      # 源包初始化
│       ├── Xattention.py    # Xattention 实现
│       ├── Flexprefill.py   # FlexPrefill 实现
│       ├── Minference.py    # Minference 实现
│       ├── Fullprefill.py   # FullPrefill 实现
│       ├── load_llama.py    # LLaMA 模型加载工具
│       └── utils.py         # 工具函数
├── config/                  # 配置文件
│   └── xattn_config.json    # 默认配置
├── examples/                # 示例使用脚本
├── tests/                   # 单元测试
├── docs/                    # 文档
├── third_party/             # 第三方依赖
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 包配置
└── README.md                # 说明文件
```

## 🚀 快速开始

### 📋 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 12.4+
- GPU 内存 24GB+

### ⚙️ 安装

```bash
# 克隆仓库
git clone https://github.com/qqtang-code/SparseAttn.git
cd SparseAttn

# 安装依赖
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu124
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
git clone https://gitee.com/codingQQT/Block-Sparse-Attention.git
cd Block-Sparse-Attention && CUDA_HOME=/usr/local/cuda-12.4/ python setup.py install
pip install -r requirements.txt

# 安装 SparseAttn
pip install -e .
```

### 🎬 基本使用

#### 1. Xattention 稀疏注意力

```python
from sparseattn.src.Xattention import Xattention_prefill
import torch

# 初始化输入张量
batch_size, num_heads, seq_len, head_dim = 1, 32, 4096, 128
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# 执行稀疏注意力计算
output = Xattention_prefill(
    query_states=query,
    key_states=key,
    value_states=value,
    threshold=0.95,  # 稀疏化阈值
    causal=True      # 是否使用因果掩码
)
```

#### 2. FlexPrefill 块稀疏注意力

```python
from sparseattn.src.Flexprefill import Flexprefill_prefill

# 块稀疏注意力计算
output = Flexprefill_prefill(
    query_states=query,
    key_states=key,
    value_states=value,
    block_size=64,      # 块大小
    sparsity_ratio=0.2  # 稀疏比例
)
```

#### 3. Minference 轻量级推理

```python
from sparseattn.src.Minference import Minference_prefill

# 轻量级推理模式
output = Minference_prefill(
    query_states=query,
    key_states=key,
    value_states=value,
    vertical_size=1000,    # 垂直稀疏大小
    slash_size=6096,       # 对角稀疏大小
    adaptive_budget=0.1    # 自适应预算
)
```

### 🏋️ 训练使用

#### 1. 稀疏微调

```bash
# 使用学习的掩码和权重进行微调
cd sparseattn/run_scripts
bash prulong_masksandweights.sh

# 使用固定掩码进行微调（仅训练权重）
bash prulong_masksonly.sh

# 标准SFT基线
bash sft.sh
```

#### 2. 训练配置

关键训练参数：
- `start_head_sparsity`: 注意力头的初始稀疏比例
- `end_head_sparsity`: 注意力头的最终稀疏比例
- `mask_learning_rate`: 掩码参数的学习率
- `reg_learning_rate`: 正则化参数的学习率
- `sparsity_warmup_ratio`: 稀疏预热的训练步数比例
- `seq_parallel_size`: 分布式训练的序列并行度

### 🔧 配置文件

创建配置文件 `config/xattn_config.json`:

```json
{
    "stride": 16,
    "threshold": 0.95,
    "block_size": 64,
    "sparsity_ratio": 0.2,
    "adaptive_budget": 0.1
}
```

## 📚 API 文档

### Xattention

Xattention 提供基于阈值的自适应稀疏注意力计算。

```python
def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    threshold: float = 0.95,
    causal: bool = True
) -> torch.Tensor
```

参数:
- `query_states`: 查询张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `key_states`: 键张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `value_states`: 值张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `threshold`: 稀疏化阈值（默认: 0.95）
- `causal`: 是否应用因果掩码（默认: True）

返回:
- 与输入张量形状相同的输出张量

### FlexPrefill

FlexPrefill 实现具有自适应块选择的块级稀疏注意力。

```python
def Flexprefill_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int = 64,
    sparsity_ratio: float = 0.2
) -> torch.Tensor
```

参数:
- `query_states`: 查询张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `key_states`: 键张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `value_states`: 值张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `block_size`: 每个块的大小（默认: 64）
- `sparsity_ratio`: 选择的块比例（默认: 0.2）

返回:
- 与输入张量形状相同的输出张量

### Minference

Minference 提供具有垂直和对角稀疏模式的轻量级推理。

```python
def Minference_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    vertical_size: int = 1000,
    slash_size: int = 6096,
    adaptive_budget: float = None
) -> torch.Tensor
```

参数:
- `query_states`: 查询张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `key_states`: 键张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `value_states`: 值张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `vertical_size`: 垂直稀疏模式的大小（默认: 1000）
- `slash_size`: 对角稀疏模式的大小（默认: 6096）
- `adaptive_budget`: 自适应预算比例（默认: None）

返回:
- 与输入张量形状相同的输出张量

### FullPrefill

FullPrefill 提供基于 FlashInfer 的完整预填充实现。

```python
def Full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    attention_mask = None
) -> torch.Tensor
```

参数:
- `query_states`: 查询张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `key_states`: 键张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `value_states`: 值张量，形状为 [batch_size, num_heads, seq_len, head_dim]
- `causal`: 是否应用因果掩码（默认: True）
- `attention_mask`: 自定义注意力掩码（默认: None）

返回:
- 与输入张量形状相同的输出张量