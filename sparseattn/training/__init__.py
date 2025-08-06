"""
Sparse Attention Library - Training Modules
==========================================

Training support for sparse attention mechanisms in large language models.

Modules:
- distributed_attention: Distributed attention implementation for multi-GPU training
- lh_trainer: Custom trainer with sparse attention support
- modeling_flash_llama: LLaMA model with flash attention support
- dataset: Dataset processing utilities
- script_arguments: Training script argument parsing
- attention_mask: Attention mask utilities
- lh_train_language_model: Language model training utilities
"""

from .distributed_attention import DistributedAttention
from .lh_trainer import Trainer
from .modeling_flash_llama import PawLlamaForCausalLM
from .dataset import DataArguments, DataCollator
from .script_arguments import ScriptArguments

__all__ = [
    "DistributedAttention",
    "Trainer",
    "PawLlamaForCausalLM",
    "DataArguments",
    "DataCollator",
    "ScriptArguments",
]