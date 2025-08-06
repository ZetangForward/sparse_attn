# SparseAttn Documentation

Welcome to the SparseAttn documentation!

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Training Support](#training-support)
5. [Examples](#examples)
6. [Benchmarks](#benchmarks)
7. [Contributing](#contributing)

## Introduction

SparseAttn is a high-performance sparse attention library designed specifically for large-scale language models. Through advanced sparsification techniques and GPU optimizations, it significantly reduces memory consumption and computational complexity of attention mechanisms while maintaining model performance.

## Installation

To install SparseAttn, follow these steps:

```bash
# Clone the repository
git clone https://github.com/qqtang-code/SparseAttn.git
cd SparseAttn

# Install dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu124
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
git clone https://gitee.com/codingQQT/Block-Sparse-Attention.git
cd Block-Sparse-Attention && CUDA_HOME=/usr/local/cuda-12.4/ python setup.py install
pip install -r requirements.txt

# Install SparseAttn
pip install -e .
```

## API Reference

### Xattention

Xattention provides adaptive sparse attention computation based on thresholding.

### FlexPrefill

FlexPrefill implements block-level sparse attention with adaptive block selection.

### Minference

Minference provides lightweight inference with vertical and diagonal sparsity patterns.

### FullPrefill

FullPrefill provides a complete prefill implementation based on FlashInfer.

## Training Support

SparseAttn includes comprehensive training support for sparse attention mechanisms:

### Distributed Training

The library supports multi-GPU and multi-node training with sequence parallelism through the [DistributedAttention](file:///data/anaconda3/lib/python3.8/site-packages/torch/nn/parallel/distributed.py#L156-L156) module.

### Sparse Fine-tuning

Methods for training sparse attention patterns on language models:
- Training with learned masks and weights
- Training with fixed masks (only train weights)
- Standard SFT baseline

### Flexible Sparsity Control

Configurable sparsity parameters:
- `start_head_sparsity`: Initial sparsity ratio for attention heads
- `end_head_sparsity`: Final sparsity ratio for attention heads
- `mask_learning_rate`: Learning rate for mask parameters
- `reg_learning_rate`: Learning rate for regularization parameters
- `sparsity_warmup_ratio`: Ratio of training steps for sparsity warmup
- `seq_parallel_size`: Sequence parallelism degree for distributed training

### Key Training Modules

1. [lh_trainer.py](file:///data/qqt/project/SparseAttn/sparseattn/training/lh_trainer.py) - Custom trainer with sparse attention support
2. [distributed_attention.py](file:///data/qqt/project/SparseAttn/sparseattn/training/distributed_attention.py) - Distributed attention implementation
3. [modeling_flash_llama.py](file:///data/qqt/project/SparseAttn/sparseattn/training/modeling_flash_llama.py) - LLaMA model with flash attention support
4. [run_scripts/](file:///data/qqt/project/SparseAttn/sparseattn/run_scripts/) - Training and evaluation scripts

## Examples

Example usage scripts can be found in the [examples/](file:///data/qqt/project/SparseAttn/examples/) directory.

## Benchmarks

Performance benchmarks are available in the main README.

## Contributing

Please see the contributing guidelines in the main README.