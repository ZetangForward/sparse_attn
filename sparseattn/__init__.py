"""
Sparse Attention Library
=========================

A high-performance sparse attention library for large language models.

Modules:
- Xattention: Adaptive sparse attention based on thresholding
- FlexPrefill: Block-level sparse attention with adaptive selection
- Minference: Lightweight inference with vertical and diagonal sparsity
- FullPrefill: Complete prefill implementation based on FlashInfer
- Training: Sparse fine-tuning support for language models
"""

from .src import (
    Xattention_prefill,
    Flexprefill_prefill,
    Minference_prefill,
    Full_prefill,
)

from .threshold import (
    llama_fuse_16,
    llama_fuse_8,
    llama_fuse_4,
)

# Training modules
from .training import DistributedAttention

__all__ = [
    "Xattention_prefill",
    "Flexprefill_prefill",
    "Minference_prefill",
    "Full_prefill",
    "llama_fuse_16",
    "llama_fuse_8",
    "llama_fuse_4",
    "DistributedAttention",
]

__version__ = "0.1.0"
__author__ = "QQTang Code"
