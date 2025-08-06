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
from .lh_trainer import LHTrainer
from .modeling_flash_llama import FlashLlamaForCausalLM
from .dataset import DataArguments, DataCollatorForSupervisedDataset
from .script_arguments import ScriptArguments
from .attention_mask import AttentionMaskConverter
from .lh_train_language_model import train_model

__all__ = [
    "DistributedAttention",
    "LHTrainer",
    "FlashLlamaForCausalLM",
    "DataArguments",
    "DataCollatorForSupervisedDataset",
    "ScriptArguments",
    "AttentionMaskConverter",
    "train_model"
]