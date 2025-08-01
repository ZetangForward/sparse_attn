# SparseAttn Documentation

Welcome to the SparseAttn documentation!

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Benchmarks](#benchmarks)
6. [Contributing](#contributing)

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

## Examples

See the [examples](../examples/) directory for usage examples.

## Benchmarks

// TODO: Add benchmark results

## Contributing

// TODO: Add contributing guidelines