"""
Sparse Attention Library - Threshold-based Sparse Attention Modules
===================================================================

Threshold-based sparse attention implementations for LLaMA models:

- llama_thrshold: Threshold-based fusion methods for different precision levels (16, 8, 4)
"""

from .llama_thrshold import (
    llama_fuse_16,
    llama_fuse_8,
    llama_fuse_4,
)

__all__ = ["llama_fuse_16", "llama_fuse_8", "llama_fuse_4"]
