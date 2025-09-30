"""
Sparse Attention Library - Training Modules
==========================================

Training support for sparse attention mechanisms in large language models.

This module provides essential components for training language models with
sparse attention mechanisms, including custom trainers, distributed attention
implementations, and dataset utilities.

Core Components:
----------------
- DistributedAttention: Distributed attention implementation for multi-GPU training
- Trainer: Custom trainer with sparse attention support and enhanced evaluation metrics
- PawLlamaForCausalLM: LLaMA model with flash attention and sparse training support
- DataArguments: Configuration for dataset processing parameters
- DataCollator: Data collator for preparing batches with attention masks
- ScriptArguments: Extended training arguments for sparse attention training

Examples:
---------
>>> from sparseattn.training import DistributedAttention, Trainer
>>> from sparseattn.training import PawLlamaForCausalLM
>>> from sparseattn.training import DataArguments, DataCollator

For more information, visit: https://github.com/qqtang-code/SparseAttn
"""

from .distributed_attention import DistributedAttention
from .lh_trainer import Trainer
from .modeling_flash_llama import PawLlamaForCausalLM
from .modeling_flash_qwen import PawQwen3ForCausalLM
from .dataset import DataArguments, DataCollator
from .script_arguments import ScriptArguments

# For backward compatibility and ease of use
LlamaForCausalLM = PawLlamaForCausalLM
Qwen3ForCausalLM = PawQwen3ForCausalLM

__all__ = [
    "DistributedAttention",
    "Trainer",
    "PawLlamaForCausalLM",
    "PawQwen3ForCausalLM",
    "DataArguments",
    "DataCollator",
    "ScriptArguments",
    # Aliases for backward compatibility
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
]
