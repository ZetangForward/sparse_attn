# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.distributed as dist

import os

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput, LossKwargs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except ImportError:
    raise ImportError(
        "Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
    )
from block_sparse_attn import block_streaming_attn_func

from dataclasses import dataclass

from .distributed_attention import DistributedAttention
from .attention_mask import (
    deterministic_z_from_log_alpha,
    sample_z_from_log_alpha,
    cdf_stretched_concrete,
)

logger = logging.get_logger(__name__)


class PawLlamaConfig(LlamaConfig):
    def __init__(self, *args, **kwargs):
        self.local_window_size = kwargs.pop("local_window_size", 1024)
        self.disable_linear_regularization_term = kwargs.pop(
            "disable_linear_regularization_term", False
        )
        self.suggested_sparsity = kwargs.pop("suggested_sparsity", None)

        # Streaming
        self.toggle_type = kwargs.pop("toggle_type", "streaming")
        self.sink_size = kwargs.pop("sink_size", 128)

        # TriangleMix
        self.triangle_n_last = kwargs.pop("triangle_n_last", 128)

        # Layer-wise sparsity control
        self.enable_layerwise_sparsity = kwargs.pop("enable_layerwise_sparsity", False)

        self.layerwise_sparsity_schedule = kwargs.pop(
            "layerwise_sparsity_schedule", "high-low-high"
        )
        self.layerwise_sparsity_min_ratio = kwargs.pop(
            "layerwise_sparsity_min_ratio", 0.5
        )
        self.layerwise_sparsity_max_ratio = kwargs.pop(
            "layerwise_sparsity_max_ratio", 1.0
        )
        self.layerwise_sparsity_power = kwargs.pop("layerwise_sparsity_power", 1.0)
        self.layerwise_sparsity_weight = kwargs.pop("layerwise_sparsity_weight", 1.0)

        self.erank_analysis_path = kwargs.pop("erank_analysis_path", None)

        super().__init__(*args, **kwargs)


def get_mask(
    log_alpha, training=False, threshold_for_deterministic=None, apply_one=False
):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask


def generate_streaming_info_blocksparse_flash_attn(
    sink_block_num, local_block_num, n_query_heads, device
):
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * n_query_heads,
        device=device,
        dtype=torch.int32,
    )
    return streaming_info


def streaming_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    streaming_info_kwargs: dict,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
    window_size: Tuple[int, int] = (0, 0),
) -> Optional[torch.Tensor]:
    # kv is of shape [total_seqlen, k_or_v, num_heads, head_dim]
    k, v = kv[:, 0, :, :], kv[:, 1, :, :]

    total_seqlen, query_heads, head_dim = q.size()
    key_value_heads = k.size(1)

    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )

    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(
        **streaming_info_kwargs
    )

    attn_output = block_streaming_attn_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        streaming_info,
        max_seqlen,
        max_seqlen,
        p_dropout=dropout_p,
        is_causal=causal,
    )

    return attn_output


def streaming_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    streaming_info_kwargs: dict,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
) -> Optional[torch.Tensor]:
    # kv is of shape [bsz, kv_seq_len, k_or_v, num_heads, head_dim]

    bsz, seqlen, query_heads, head_dim = q.size()
    k, v = kv[:, :, 0, :, :], kv[:, :, 1, :, :]

    key_value_heads = k.size(2)
    kv_seqlen = k.size(1)

    q_unpad = q.view(bsz * seqlen, query_heads, head_dim)
    k_unpad = k.view(bsz * kv_seqlen, key_value_heads, head_dim)
    v_unpad = v.view(bsz * kv_seqlen, key_value_heads, head_dim)

    cu_seqlens_q = torch.arange(
        0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q_unpad.device
    )
    cu_seqlens_kv = torch.arange(
        0,
        (bsz + 1) * kv_seqlen,
        step=kv_seqlen,
        dtype=torch.int32,
        device=k_unpad.device,
    )

    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )

    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q_unpad.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(
        **streaming_info_kwargs
    )

    attn_output = block_streaming_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_kv,
        head_mask_type,
        streaming_info,
        seqlen,
        seqlen,
        p_dropout=dropout_p,
        is_causal=causal,
    )

    return attn_output.reshape(bsz, seqlen, query_heads, head_dim)


def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "variance_epsilon",
            torch.tensor(eps),
            persistent=False,
        )

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)


class FlashRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]
        self._update_cos_sin_cache(
            max_seqlen + seqlen_offset, device=q.device, dtype=q.dtype
        )

        if self.scale is None:
            return apply_rotary_emb_func(
                q,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ), apply_rotary_emb_func(
                k,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            assert False


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        interleaved=False,
        config: Optional[PawLlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = True

        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"

        self._seq_len_cached = 0

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len

            if "dynamic" in self.rope_type:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.config, device, seq_len=seq_len, **self.rope_kwargs
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = (torch.cos(freqs) * self.attention_scaling).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.attention_scaling).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,  # Used in sequence parallelism where each device sees only a chunk of the full sequence
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ):
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
            if seqlen_offset > 0:
                raise ValueError("seqlen_offset is not supported with unpadded_lengths")
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]

        self._update_cos_sin_cache(max_seqlen + seqlen_offset, q.device, q.dtype)

        return apply_rotary_emb_func(
            q,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ), apply_rotary_emb_func(
            k,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    final_shape = list(hidden_states.shape[:-2]) + [-1] + [hidden_states.shape[-1]]
    expand_shape = [-1] * (len(hidden_states.shape) - 1) + [n_rep] + [-1]
    hidden_states = hidden_states.unsqueeze(-2).expand(expand_shape)
    return hidden_states.reshape(final_shape)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PawLlamaConfig,
        context_window_toggle: Optional[int] = 1024,
    ):
        """
        @context_window_toggle: if not None, the attention will be limited to a context window specified by this value
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.distributed_attn_func = DistributedAttention(self.interpolated_attention)

        self._dtype = self.q_proj.weight.dtype
        self.attn_mask_log_alphas = nn.Parameter(
            torch.empty(self.num_key_value_heads, dtype=self._dtype)
        )
        self.attn_mask_log_alphas.data.normal_(
            mean=4.5, std=0.01
        )  # sigmoid(4.5) ≈ 0.989
        self.threshold_for_deterministic = None

        self.context_window_toggle = context_window_toggle

        self.toggle_type = config.toggle_type
        self.sink_blocks = (config.sink_size + 127) // 128
        self.local_blocks = (config.local_window_size + 127) // 128

        if self.toggle_type == "streaming":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
        elif self.toggle_type == "local":
            pass
        elif self.toggle_type == "triangle":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
            self.triangle_n_last = config.triangle_n_last
        elif self.toggle_type == "topk":
            self.context_window_toggle = config.local_window_size
            self.topk_block_size = int(getattr(config, "topk_block_size", 128))
            self.topk_query_block_size = int(
                getattr(config, "topk_query_block_size", 128)
            )
        elif self.toggle_type == "xattn":
            from sparseattn.utils.ops.xattention_fa import xattn_flash_attn_func
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            # self.head_indices = self.num_heads // self.num_key_value_heads
            self.head_indices = self.num_heads
            self.xattn_flash_attn_func = xattn_flash_attn_func
            self.granularity = int(getattr(config, "block_size", 64))
            self.xattn_params = {
                "stride": 16,
                "norm": 1,
                "softmax": True,
                "threshold": 0.9,
                "chunk_size": 16384,
                "select_mode": "inverse",
                "use_triton": True,
                "causal": True,
                "kdb": 1,
                "keep_sink": True,
                "keep_recent": True,
            }
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.threshold_for_deterministic = threshold_for_deterministic

    @torch.no_grad()
    def get_masks(self):
        z = get_mask(
            self.attn_mask_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.threshold_for_deterministic,
        )
        return z

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.attn_mask_log_alphas.data.normal_(mean=value, std=0.01)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        if (
            isinstance(value, float)
            or isinstance(value, int)
            or (isinstance(value, torch.Tensor) and value.numel() == 1)
        ):
            self.attn_mask_log_alphas.data.fill_(value)
        else:
            if isinstance(value, list):
                value = torch.tensor(
                    value, dtype=self._dtype, device=self.attn_mask_log_alphas.device
                )
            value = value.reshape(-1)
            assert value.shape[0] == self.attn_mask_log_alphas.numel(), (
                "Value shape does not match mask shape"
            )
            self.attn_mask_log_alphas.data.copy_(value)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    # ===== Top-K attention helpers =====
    def _topk_attn_padded(
        self, q: torch.Tensor, kv: torch.Tensor, k_top: int
    ) -> torch.Tensor:
        """
        q: [B, T, H, D], kv: [B, S, 2, H, D]，
        returns: [B, T, H, D]
        """
        bsz, tq, heads, dim = q.size()
        k_t = kv[:, :, 0, :, :]  # [B, S, H, D]
        v_t = kv[:, :, 1, :, :]  # [B, S, H, D]
        sk = k_t.size(1)
        device = q.device
        dtype = q.dtype

        scale = (1.0 / self.norm_factor).to(dtype)
        K = max(1, min(int(k_top), sk))
        BS_cfg = int(getattr(self, "topk_block_size", 512))
        BS_env = int(os.environ.get("TOPK_BLOCK_SIZE", 0)) if "os" in globals() else 0
        BS = max(1, BS_env if BS_env > 0 else BS_cfg)
        QBS_cfg = int(getattr(self, "topk_query_block_size", 128))
        QBS_env = (
            int(os.environ.get("TOPK_Q_BLOCK_SIZE", 0)) if "os" in globals() else 0
        )
        QBS = max(1, QBS_env if QBS_env > 0 else QBS_cfg)

        q_bhtd = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        k_bhsd = k_t.permute(0, 2, 1, 3)  # [B, H, S, D]
        v_bhsd = v_t.permute(0, 2, 1, 3)  # [B, H, S, D]

        topk_vals = torch.full(
            (bsz, heads, tq, K), float("-inf"), device=device, dtype=dtype
        )
        topk_idx = torch.zeros((bsz, heads, tq, K), device=device, dtype=torch.long)

        offset = sk - tq
        t_idx = torch.arange(tq, device=device)

        def run_with_bs(cur_bs: int, cur_qbs: int):
            nonlocal topk_vals, topk_idx
            for start in range(0, sk, cur_bs):
                end = min(start + cur_bs, sk)
                bs = end - start
                k_blk = k_bhsd[:, :, start:end, :]  # [B, H, bs, D]
                k_blk_T = k_blk.transpose(-1, -2)  # [B, H, D, bs]
                for t_s in range(0, tq, cur_qbs):
                    t_e = min(t_s + cur_qbs, tq)
                    t_len = t_e - t_s

                    q_blk = q_bhtd[:, :, t_s:t_e, :]  # [B, H, t_len, D]
                    # scores_blk: [B, H, t_len, bs]
                    scores_blk = torch.matmul(q_blk, k_blk_T) * scale

                    k_idx_blk = start + torch.arange(bs, device=device)  # [bs]
                    t_idx_blk = t_idx[t_s:t_e]  # [t_len]
                    allowed = k_idx_blk.view(1, 1, 1, bs) <= (
                        offset + t_idx_blk.view(1, 1, t_len, 1)
                    )
                    scores_blk = scores_blk.masked_fill(~allowed, float("-inf"))

                    Kb = min(K, bs)
                    blk_vals, blk_pos_in_blk = torch.topk(
                        scores_blk, k=Kb, dim=-1
                    )  # [B, H, t_len, Kb]
                    blk_abs_idx = (start + blk_pos_in_blk).to(
                        topk_idx.dtype
                    )  # [B, H, t_len, Kb]

                    prev_vals = topk_vals[:, :, t_s:t_e, :]  # [B, H, t_len, K]
                    prev_idx = topk_idx[:, :, t_s:t_e, :]  # [B, H, t_len, K]

                    cand_vals = torch.cat(
                        [prev_vals, blk_vals], dim=-1
                    )  # [B, H, t_len, K+Kb]
                    cand_idx = torch.cat(
                        [prev_idx, blk_abs_idx], dim=-1
                    )  # [B, H, t_len, K+Kb]

                    new_vals, new_pos = torch.topk(
                        cand_vals, k=K, dim=-1
                    )  # [B, H, t_len, K]
                    new_idx = torch.gather(
                        cand_idx, dim=-1, index=new_pos
                    )  # [B, H, t_len, K]

                    topk_vals[:, :, t_s:t_e, :] = new_vals
                    topk_idx[:, :, t_s:t_e, :] = new_idx

        cur_bs = BS
        cur_qbs = QBS
        while True:
            try:
                run_with_bs(cur_bs, cur_qbs)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                    if cur_qbs > 1:
                        cur_qbs = max(1, cur_qbs // 2)
                        torch.cuda.empty_cache()
                        continue
                    if cur_bs > 1:
                        cur_bs = max(1, cur_bs // 2)
                        torch.cuda.empty_cache()
                        continue
                raise

        out_bhtd = torch.zeros_like(q_bhtd)  # [B, H, T, D]
        for t_s in range(0, tq, QBS):
            t_e = min(t_s + QBS, tq)
            t_len = t_e - t_s
            weights_blk = F.softmax(
                topk_vals[:, :, t_s:t_e, :], dim=-1
            )  # [B, H, t_len, K]
            topk_idx_blk = topk_idx[:, :, t_s:t_e, :]  # [B, H, t_len, K]
            for b in range(bsz):
                for h in range(heads):
                    idx_b_h = topk_idx_blk[b, h]  # [t_len, K]
                    v_b_h = v_bhsd[b, h]  # [S, D]
                    gathered_v_blk = v_b_h[idx_b_h]  # [t_len, K, D]
                    out_bhtd[b, h, t_s:t_e, :] = (
                        weights_blk[b, h].unsqueeze(-1) * gathered_v_blk
                    ).sum(dim=1)  # [t_len, D]

        return out_bhtd.permute(0, 2, 1, 3)  # [B, T, H, D]

    def _topk_attn_varlen(
        self, q: torch.Tensor, kv: torch.Tensor, cu_seqlens: torch.Tensor, k_top: int
    ) -> torch.Tensor:
        """
        q:  [total, H, D], kv: [total, 2, H, D], cu_seqlens: [B+1]
        对每段 [L] 使用分块 Top-K 计算，返回 [total, H, D]
        """
        total, heads, dim = q.size()
        device, dtype = q.device, q.dtype
        scale = (1.0 / self.norm_factor).to(dtype)
        BS_cfg = int(getattr(self, "topk_block_size", 1024))
        BS_env = int(os.environ.get("TOPK_BLOCK_SIZE", 0)) if "os" in globals() else 0
        BS = max(1, BS_env if BS_env > 0 else BS_cfg)
        outputs = []

        B = cu_seqlens.numel() - 1
        for b in range(B):
            start = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())
            L = end - start
            if L == 0:
                continue

            qb = q[start:end]  # [L, H, D]
            kb = kv[start:end, 0, :, :]  # [L, H, D]
            vb = kv[start:end, 1, :, :]  # [L, H, D]

            K = max(1, min(int(k_top), L))

            q_hld = qb.permute(1, 0, 2)  # [H, L, D]
            k_hsd = kb.permute(1, 0, 2)  # [H, L, D]
            v_hsd = vb.permute(1, 0, 2)  # [H, L, D]

            # [H, L, K]
            topk_vals = torch.full(
                (heads, L, K), float("-inf"), device=device, dtype=dtype
            )
            topk_idx = torch.zeros((heads, L, K), device=device, dtype=torch.long)

            t_idx = torch.arange(L, device=device)

            def run_with_bs(cur_bs: int):
                nonlocal topk_vals, topk_idx
                for ks in range(0, L, cur_bs):
                    ke = min(ks + cur_bs, L)
                    bs = ke - ks
                    k_blk = k_hsd[:, ks:ke, :]  # [H, bs, D]
                    # scores_blk: [H, L, bs]
                    scores_blk = torch.matmul(q_hld, k_blk.transpose(-1, -2)) * scale

                    k_idx_blk = ks + torch.arange(bs, device=device)  # [bs]
                    allowed = k_idx_blk.view(1, 1, bs) <= t_idx.view(
                        1, L, 1
                    )  # [H=1, L, bs]
                    scores_blk = scores_blk.masked_fill(~allowed, float("-inf"))

                    Kb = min(K, bs)
                    blk_vals, blk_pos_in_blk = torch.topk(
                        scores_blk, k=Kb, dim=-1
                    )  # [H, L, Kb]
                    blk_abs_idx = (ks + blk_pos_in_blk).to(topk_idx.dtype)  # [H, L, Kb]

                    cand_vals = torch.cat([topk_vals, blk_vals], dim=-1)  # [H, L, K+Kb]
                    cand_idx = torch.cat(
                        [topk_idx, blk_abs_idx], dim=-1
                    )  # [H, L, K+Kb]

                    new_vals, new_pos = torch.topk(cand_vals, k=K, dim=-1)  # [H, L, K]
                    topk_vals = new_vals
                    topk_idx = torch.gather(cand_idx, dim=-1, index=new_pos)

            cur_bs = BS
            while True:
                try:
                    run_with_bs(cur_bs)
                    break
                except RuntimeError as e:
                    if (
                        "CUDA out of memory" in str(e)
                        and cur_bs > 1
                        and torch.cuda.is_available()
                    ):
                        cur_bs = max(1, cur_bs // 2)
                        torch.cuda.empty_cache()
                        continue
                    raise

            out_hld = torch.zeros_like(q_hld)  # [H, L, D]
            for t_s in range(0, L, BS):
                t_e = min(t_s + BS, L)
                t_len = t_e - t_s
                weights_blk = F.softmax(
                    topk_vals[:, t_s:t_e, :], dim=-1
                )  # [H, t_len, K]
                topk_idx_blk = topk_idx[:, t_s:t_e, :]  # [H, t_len, K]
                for h in range(heads):
                    idx_h = topk_idx_blk[h]  # [t_len, K]
                    v_h = v_hsd[h]  # [L, D]
                    gathered_v_blk = v_h[idx_h]  # [t_len, K, D]
                    out_hld[h, t_s:t_e, :] = (
                        weights_blk[h].unsqueeze(-1) * gathered_v_blk
                    ).sum(dim=1)  # [t_len, D]

            outputs.append(out_hld.permute(1, 0, 2))  # [L, H, D]

        return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

    def interpolated_attention(self, q, kv, k, v, unpadded_lengths, z):
        if unpadded_lengths is not None:
            # varlen, ignore padding tokens, efficient for large batch with many paddings
            cu_seqlens, max_seqlen = unpadded_lengths

            attn_output, _, attn_probs = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=0.0,
                softmax_scale=1.0 / self.norm_factor,
                causal=True,
                return_attn_probs=True,
            )
        else:
            attn_output, _, attn_probs = flash_attn_kvpacked_func(
                q,
                kv,
                dropout_p=0.0,
                softmax_scale=1.0 / self.norm_factor,
                causal=True,
                return_attn_probs=True,
            )

        if self.toggle_type == "streaming" or self.toggle_type == "triangle":
            if unpadded_lengths is not None:
                cu_seqlens, max_seqlen = unpadded_lengths
                cw_attn_output = streaming_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    self.streaming_info_kwargs,
                    cu_seqlens,
                    max_seqlen,
                    dropout_p=0.0,
                    causal=True,
                    return_attn_probs=False,
                )
            else:
                cw_attn_output = streaming_attn_kvpacked_func(
                    q,
                    kv,
                    self.streaming_info_kwargs,
                    dropout_p=0.0,
                    causal=True,
                    return_attn_probs=False,
                )
            if self.toggle_type == "triangle":
                if unpadded_lengths is not None:
                    cu_seqlens, _ = unpadded_lengths
                    total = q.size(0)
                    mask = torch.zeros(total, dtype=torch.bool, device=q.device)
                    B = cu_seqlens.numel() - 1
                    n_last = self.triangle_n_last
                    for b in range(B):
                        start = int(cu_seqlens[b].item())
                        end = int(cu_seqlens[b + 1].item())
                        seg_len = end - start
                        take = min(n_last, seg_len)
                        if take > 0:
                            mask[end - take : end] = True
                    cw_attn_output[mask] = attn_output[mask]
                else:
                    seq_len = q.size(1)
                    take = min(getattr(self, "triangle_n_last", 0), seq_len)
                    if take > 0:
                        cw_attn_output[:, -take:] = attn_output[:, -take:]

        elif self.toggle_type == "local":
            if unpadded_lengths is not None:
                # varlen, ignore padding tokens, efficient for large batch with many paddings
                cu_seqlens, max_seqlen = unpadded_lengths

                cw_attn_output = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
            else:
                cw_attn_output = flash_attn_kvpacked_func(
                    q,
                    kv,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
        elif self.toggle_type == "topk":
            k_top = (
                self.context_window_toggle
                if self.context_window_toggle is not None
                else 0
            )
            k_top = max(1, int(k_top))

            if unpadded_lengths is not None:
                cu_seqlens, _ = unpadded_lengths
                if attn_probs is None:
                    cw_attn_output = self._topk_attn_varlen(q, kv, cu_seqlens, k_top)
                else:
                    v = kv[:, 1, :, :]  # [total, H, D]
                    outputs = []
                    B = cu_seqlens.numel() - 1
                    eps = 1e-9
                    for b in range(B):
                        start = int(cu_seqlens[b].item())
                        end = int(cu_seqlens[b + 1].item())
                        L = end - start
                        if L == 0:
                            continue
                        probs_i = attn_probs[start:end, :, :L]  # [L, H, L]
                        v_i = v[start:end, :, :]  # [L, H, D]
                        K = max(1, min(k_top, L))
                        topk_vals, topk_idx = torch.topk(
                            probs_i, k=K, dim=-1
                        )  # [L, H, K]
                        weights = topk_vals / topk_vals.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(eps)
                        v_hld = v_i.permute(1, 0, 2)  # [H, L, D]
                        idx_hlk = topk_idx.permute(1, 0, 2)  # [H, L, K]
                        gathered_v = torch.gather(
                            v_hld.unsqueeze(2).expand(-1, -1, K, -1),  # [H, L, K, D]
                            dim=1,
                            index=idx_hlk.unsqueeze(-1).expand(
                                -1, -1, -1, v_hld.size(-1)
                            ),
                        )  # [H, L, K, D]
                        out_hld = (
                            weights.permute(1, 0, 2).unsqueeze(-1) * gathered_v
                        ).sum(dim=2)
                        outputs.append(out_hld.permute(1, 0, 2))  # [L, H, D]
                    cw_attn_output = (
                        torch.cat(outputs, dim=0)
                        if len(outputs) > 0
                        else torch.empty_like(q)
                    )
            else:
                if attn_probs is None:
                    cw_attn_output = self._topk_attn_padded(q, kv, k_top)
                else:
                    probs_bhts = attn_probs  # [B, H, T, S]
                    Bsz, H, T, S = probs_bhts.shape
                    K = max(1, min(k_top, S))
                    topk_vals, topk_idx = torch.topk(
                        probs_bhts, k=K, dim=-1
                    )  # [B, H, T, K]
                    eps = 1e-9
                    weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(
                        eps
                    )
                    v = kv[:, :, 1, :, :]  # [B, S, H, D]
                    v_bhsd = v.permute(0, 2, 1, 3)  # [B, H, S, D]
                    gathered_v = torch.gather(
                        v_bhsd.unsqueeze(2).expand(
                            -1, -1, T, -1, -1
                        ),  # [B, H, T, S, D]
                        dim=3,
                        index=topk_idx.unsqueeze(-1).expand(
                            -1, -1, -1, -1, v_bhsd.size(-1)
                        ),
                    )  # [B, H, T, K, D]
                    out_bhtd = (weights.unsqueeze(-1) * gathered_v).sum(
                        dim=3
                    )  # [B, H, T, D]
                    cw_attn_output = out_bhtd.permute(0, 2, 1, 3)  # [B, T, H, D]
        elif self.toggle_type == "xattn":  
            # breakpoint()
            # def set_pdb_trace(rank: int = None):
            #     import pdb
            #     import sys
            #     sys.stdin = open(0)  
            #     pdb.set_trace()
            # set_pdb_trace(0)
            if not self.training :
                _, seq_len, _, _ = q.size()         
            if self.training or seq_len != 1:
                if q.dim() == 3:
                    q = q.unsqueeze(0)
                    k = k.unsqueeze(0)
                    v = v.unsqueeze(0)
                # cw_attn_output = self.xattn_flash_attn_func(
                #     q,
                #     k,
                #     v,
                #     self.head_indices,
                #     self.xattn_params,
                #     self.granularity,
                # )
                k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                v = v.repeat_interleave(self.num_key_value_groups, dim=2)
                q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)  # B,H,T,D
                stride = self.xattn_params["stride"]
                threshold = self.xattn_params["threshold"]
                norm = self.xattn_params["norm"]
                from sparseattn.src.Xattention import Xattention_prefill
                cw_attn_output = Xattention_prefill(
                    q,
                    k,
                    v,
                    stride,
                    norm,
                    threshold,
                    use_triton=True,
                ).transpose(1,2)  # B,T,H,D
            else:
                if unpadded_lengths is not None:
                    # varlen, ignore padding tokens, efficient for large batch with many paddings
                    cu_seqlens, max_seqlen = unpadded_lengths

                    cw_attn_output, _, attn_probs = flash_attn_varlen_kvpacked_func(
                        q,
                        kv,
                        cu_seqlens,
                        cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        dropout_p=0.0,
                        softmax_scale=1.0 / self.norm_factor,
                        causal=True,
                        return_attn_probs=True,
                    )
                else:
                    cw_attn_output, _, attn_probs = flash_attn_kvpacked_func(
                        q,
                        kv,
                        dropout_p=0.0,
                        softmax_scale=1.0 / self.norm_factor,
                        causal=True,
                        return_attn_probs=True,
                    )
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

        effective_attn_output = attn_output * z.unsqueeze(-1) + cw_attn_output * (
            1 - z
        ).unsqueeze(-1)

        return effective_attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)

        has_layer_past = past_key_value is not None

        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        # NOTE: Hack to include position_ids, assuming they are increasing uniformly per block
        if position_ids is not None:
            past_len += position_ids.min()

        if unpadded_lengths is not None:
            # We don't use the unpadded_length during rotary embeds and instead create a temporary `batch` dimension
            # This does not actually affect the otucome since the positional embeddings are relative and stay valid
            # This also ensures that in sequence parallelism the correct `past_len` offset is applied to mid-sequence chunks
            q, k = self.rotary_emb(q.unsqueeze(0), k.unsqueeze(0), past_len)
            q, k = q.squeeze(0), k.squeeze(0)
        else:
            q, k = self.rotary_emb(q, k, past_len)
        kv = torch.stack([k, v], -3)
        kv = repeat_kv(kv, self.num_key_value_groups)

        # Cache QKV values
        if has_layer_past:
            new_len = past_len + q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat(
                    [
                        past_kv,
                        torch.empty(
                            hidden_states.size(0),
                            256,
                            2,
                            kv.size(3),
                            kv.size(4),
                            dtype=kv.dtype,
                            device=kv.device,
                        ),
                    ],
                    1,
                )
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv
        past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None

        z_kv = get_mask(
            self.attn_mask_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.threshold_for_deterministic,
        )  # (num_key_value_heads,)
        # Next: expand z_kv to (num_key_value_heads, num_key_value_groups) and then flatten it to (num_heads)
        z = z_kv.unsqueeze(-1).expand(-1, self.num_key_value_groups).reshape(-1)

        if (
            seq_parallel_group is not None
            and dist.is_initialized()
            and dist.get_world_size(seq_parallel_group) > 1
        ):
            attention_func = self.distributed_attn_func
            kwargs = {
                "group": seq_parallel_group,
                "gather_idx": (0 if unpadded_lengths is not None else 1),
            }

            z = torch.split(
                z, self.num_heads // dist.get_world_size(seq_parallel_group), dim=0
            )[dist.get_rank(seq_parallel_group)]
        else:
            attention_func = self.interpolated_attention
            kwargs = {}

        attn_output = attention_func(q, kv, k, v, unpadded_lengths, z, **kwargs)
        if k.dim() == 3:
            attn_output = attn_output.squeeze(0)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None

        return z.sum(), attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PawLlamaConfig,
        context_window_toggle: Optional[int] = 4096,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            config=config, context_window_toggle=context_window_toggle
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._fsdp_wrap = True

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.self_attn.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.self_attn.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.self_attn.reset_masks(value)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        self.self_attn.fill_masks_with_value(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        seq_parallel_group: Optional[Any] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        z_sum, hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (
            z_sum,
            hidden_states,
        )

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = PawLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class BaseModelOutputWithPastAndSparsity(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None
    # Diagnostics
    expected_model_sparsity: Optional[torch.FloatTensor] = None
    lambda1: Optional[torch.FloatTensor] = None
    lambda2: Optional[torch.FloatTensor] = None
    expected_z_mean: Optional[torch.FloatTensor] = None
    expected_z_std: Optional[torch.FloatTensor] = None
    log_alpha_mean: Optional[torch.FloatTensor] = None
    log_alpha_std: Optional[torch.FloatTensor] = None
    # Layer-wise sparsity diagnostics
    layerwise_model_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_target_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_sparsity_loss: Optional[torch.FloatTensor] = None  # scalar


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: PawLlamaConfig
    """

    def __init__(
        self,
        config: PawLlamaConfig,
    ):
        super().__init__(config)
        context_window_toggle = config.local_window_size
        disable_linear_regularization_term = config.disable_linear_regularization_term

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, context_window_toggle=context_window_toggle)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.total_num_heads = config.num_attention_heads * config.num_hidden_layers
        self.total_num_kv_heads = config.num_key_value_heads * config.num_hidden_layers

        self._dtype = self.norm.weight.dtype
        if disable_linear_regularization_term:
            self.sparsity_lambda_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_1 = nn.Parameter(
                torch.tensor([0.0], dtype=self._dtype)
            )
        self.sparsity_lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))

        self.threshold_for_deterministic = None
        if config.suggested_sparsity is not None:
            self.round_masks_for_sparsity(config.suggested_sparsity)

        self._erank_cache = {}
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def _get_avg_erank(self, path: str) -> torch.Tensor:
        key = os.path.abspath(path)
        if key in self._erank_cache:
            return self._erank_cache[key]
        erank_res = torch.load(key, map_location="cpu")
        avg_erank = erank_res["avg_erank"]
        self._erank_cache[key] = avg_erank
        return avg_erank

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        for layer in self.layers:
            layer.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_masks())
        return masks

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        for layer in self.layers:
            layer.reset_masks(value)
        self.sparsity_lambda_1.data.zero_()
        self.sparsity_lambda_2.data.zero_()

    @torch.no_grad()
    def get_sparsity(self):
        masks = self.get_masks()
        total_sum = 0
        for mask in masks:
            total_sum += mask.sum()
        return 1 - (total_sum / self.total_num_kv_heads)

    @torch.no_grad()
    def _pre_save_get_threshold(self):
        orig_threshold = self.threshold_for_deterministic

        sparsity_target = self.get_sparsity()
        l = 0
        r = 1
        while r - l > 1e-8:
            m = (l + r) / 2
            self.set_threshold_for_deterministic(m)
            if self.get_sparsity() > sparsity_target:
                r = m
            else:
                l = m
        m = (l + r) / 2

        self.config.suggested_threshold = m

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        if start_with_keep:
            value_1 = 10.0  # Some high value
            value_2 = -10.0  # Some low value
        else:
            value_1 = -10.0
            value_2 = 10.0
        for l, layer in range(len(self.layers)):
            value = value_1 if l % (width_1 + width_2) < width_1 else value_2
            layer.fill_masks_with_value(value)

    @torch.no_grad()
    def load_masks(self, masks):
        for l in range(len(masks)):
            self.layers[l].fill_masks_with_value(masks[l])

    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        masks = self.get_masks()
        # masks is a list of tensors, each tensor is of shape (num_key_value_heads,)
        # First find the number of high values
        num_high = int(sum([mask.shape[0] for mask in masks]) * (1 - target_sparsity))

        # Find the top-num_high values
        # Break ties randomly
        rng = torch.Generator()
        rng.manual_seed(42)
        value_list = [
            (i, j, masks[i][j], torch.rand(1, generator=rng).item())
            for i in range(len(masks))
            for j in range(masks[i].shape[0])
        ]
        # Sort by the random variable then resort by the value
        value_list.sort(key=lambda x: x[3])
        value_list.sort(key=lambda x: x[2], reverse=True)
        for i, j, _, _ in value_list[:num_high]:
            masks[i][j] = 10.0
        for i, j, _, _ in value_list[num_high:]:
            masks[i][j] = -10.0

        # Load the new masks
        self.load_masks(masks)

        # Return the sparsity
        return self.get_sparsity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
        erank_analysis_path: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if not self.training and use_cache:
            compute_sparsity = False
        else:
            compute_sparsity = True
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # position_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        z_sum = 0 if compute_sparsity else None
        layer_z_sums = []

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    unpadded_lengths,
                    output_attentions,
                    False,
                    seq_parallel_group,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    unpadded_lengths=unpadded_lengths,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    seq_parallel_group=seq_parallel_group,
                )

            z_layer_sum, hidden_states = layer_outputs[0], layer_outputs[1]
            if compute_sparsity:
                z_sum = z_sum + z_layer_sum
            layer_z_sums.append(z_layer_sum)

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if compute_sparsity:
            if (
                seq_parallel_group is not None
                and dist.is_initialized()
                and dist.get_world_size(seq_parallel_group) > 1
            ):
                # Collect z_sum across GPUs in sequence parallel group (i.e., across all the heads during attention)
                z_sums = [
                    torch.zeros_like(z_sum)
                    for _ in range(dist.get_world_size(seq_parallel_group))
                ]
                dist.all_gather(z_sums, z_sum, group=seq_parallel_group)
                z_sum = sum(z_sums)

                gathered_layer_z_sums = []
                for z_l in layer_z_sums:
                    tmp = [
                        torch.zeros_like(z_l)
                        for _ in range(dist.get_world_size(seq_parallel_group))
                    ]
                    dist.all_gather(tmp, z_l, group=seq_parallel_group)
                    gathered_layer_z_sums.append(sum(tmp))
                layer_z_sums = gathered_layer_z_sums

            model_sparsity = 1 - (z_sum / self.total_num_heads)
        else:
            model_sparsity = None
            z_loss = None
        if compute_sparsity:
            layerwise_model_sparsity = None
            layerwise_target = None
            layerwise_loss = None
            if len(layer_z_sums) > 0:
                per_layer_heads = self.config.num_attention_heads
                layerwise_model_sparsity = (
                    1.0 - torch.stack(layer_z_sums) / per_layer_heads
                )  # (num_layers,)

            if target_sparsity is None:
                z_loss = None
            else:
                z_loss = (
                    self.sparsity_lambda_1.reshape([])
                    * (model_sparsity - target_sparsity)
                    + self.sparsity_lambda_2.reshape([])
                    * (model_sparsity - target_sparsity) ** 2
                )
        else:
            layerwise_model_sparsity = None
            layerwise_target = None
            layerwise_loss = None
        if (
            target_sparsity is not None
            and self.config.enable_layerwise_sparsity
            and layerwise_model_sparsity is not None
            and compute_sparsity
        ):
            L = layerwise_model_sparsity.numel()
            device = layerwise_model_sparsity.device

            # new
            path = erank_analysis_path or getattr(
                self.config, "erank_analysis_path", None
            )
            if not path.endswith(".pt"):
                idxs = torch.arange(L, device=device, dtype=torch.float32)
                denom = max(L - 1, 1)
                x = torch.sin(torch.pi * (idxs / denom))
                x = x.pow(float(self.config.layerwise_sparsity_power))

                min_r = float(self.config.layerwise_sparsity_min_ratio)
                max_r = float(self.config.layerwise_sparsity_max_ratio)
                sched = getattr(
                    self.config, "layerwise_sparsity_schedule", "low-high-low"
                )
                if sched == "high-low-high":
                    ratios = max_r - (max_r - min_r) * x
                else:
                    ratios = min_r + (max_r - min_r) * x

                layerwise_target = torch.clamp(
                    ratios * float(target_sparsity), min=0.0, max=1.0
                )
            else:
                avg_erank = self._get_avg_erank(path)  # [L, H] on CPU
                E = avg_erank.mean(dim=1).to(device)  # [L] 每层平均有效秩

                assert E.numel() == L, (
                    f"Mismatch: got {E.numel()} layers from erank but model has {L}"
                )

                E_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
                inv_E = (1.0 / (E_norm + 1e-3)) ** 0.5
                w = inv_E

                L = layerwise_model_sparsity.numel()
                layerwise_target = (w / w.sum()) * (target_sparsity * L)
                layerwise_target = torch.clamp(layerwise_target, 0.0, 1.0)
                s = layerwise_target.sum()
                if s > 0:
                    layerwise_target = layerwise_target / s * (target_sparsity * L)

            diffs = layerwise_model_sparsity - layerwise_target

            def is_main_process():
                if not dist.is_available() or not dist.is_initialized():
                    return True
                return dist.get_rank() == 0

            # if is_main_process():
            #     print("============debug layerwise sparsity loss computation============")
            #     print("debug layerwise_model_sparsity:", layerwise_model_sparsity)
            #     print("debug layerwise_target:", layerwise_target)
            #     print("debug diffs:", diffs)
            #     print("============debug layerwise sparsity loss computation============")

            per_layer_loss = self.sparsity_lambda_1.reshape(
                []
            ) * diffs + self.sparsity_lambda_2.reshape([]) * (diffs**2)
            layerwise_loss = (
                float(self.config.layerwise_sparsity_weight) * per_layer_loss.mean()
            )

            # z_loss = layerwise_loss if z_loss is None else (z_loss + layerwise_loss)
            z_loss = layerwise_loss

        # Diagnostics: compute expected sparsity (no sampling) and stats over masks
        with torch.no_grad():
            try:
                # Collect all log_alphas from layers
                log_alphas = []
                per_layer_expected = []
                total_heads = 0
                for layer in self.layers:
                    la = layer.self_attn.attn_mask_log_alphas
                    # Probability head is active (non-zero) = 1 - CDF(0)
                    p_active_kv = 1 - cdf_stretched_concrete(0, la)
                    # Expand kv heads to query heads
                    p_active = (
                        p_active_kv.unsqueeze(-1)
                        .expand(-1, layer.self_attn.num_key_value_groups)
                        .reshape(-1)
                    )
                    per_layer_expected.append(p_active)
                    total_heads += p_active.numel()

                if len(per_layer_expected) > 0:
                    expected_z_all = torch.cat(per_layer_expected)
                    expected_z_mean = expected_z_all.mean()
                    expected_z_std = expected_z_all.float().std()
                    expected_model_sparsity = 1 - expected_z_mean
                    la_all = torch.cat([t.reshape(-1) for t in log_alphas])
                    log_alpha_mean = la_all.mean()
                    log_alpha_std = la_all.float().std()
                else:
                    expected_model_sparsity = torch.tensor(
                        float("nan"), device=hidden_states.device
                    )
                    expected_z_mean = torch.tensor(
                        float("nan"), device=hidden_states.device
                    )
                    expected_z_std = torch.tensor(
                        float("nan"), device=hidden_states.device
                    )
                    log_alpha_mean = torch.tensor(
                        float("nan"), device=hidden_states.device
                    )
                    log_alpha_std = torch.tensor(
                        float("nan"), device=hidden_states.device
                    )
            except Exception:
                # Be resilient to any unexpected shape/device issues
                expected_model_sparsity = torch.tensor(
                    float("nan"), device=hidden_states.device
                )
                expected_z_mean = torch.tensor(
                    float("nan"), device=hidden_states.device
                )
                expected_z_std = torch.tensor(float("nan"), device=hidden_states.device)
                log_alpha_mean = torch.tensor(float("nan"), device=hidden_states.device)
                log_alpha_std = torch.tensor(float("nan"), device=hidden_states.device)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    model_sparsity,
                    target_sparsity,
                    z_loss,
                    expected_model_sparsity,
                    self.sparsity_lambda_1.reshape([]),
                    self.sparsity_lambda_2.reshape([]),
                    expected_z_mean,
                    expected_z_std,
                    log_alpha_mean,
                    log_alpha_std,
                    layerwise_model_sparsity,
                    layerwise_target,
                    layerwise_loss,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndSparsity(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            model_sparsity=model_sparsity,
            target_sparsity=target_sparsity,
            sparsity_loss=z_loss,
            expected_model_sparsity=expected_model_sparsity,
            lambda1=self.sparsity_lambda_1.reshape([]),
            lambda2=self.sparsity_lambda_2.reshape([]),
            expected_z_mean=expected_z_mean,
            expected_z_std=expected_z_std,
            log_alpha_mean=log_alpha_mean,
            log_alpha_std=log_alpha_std,
            layerwise_model_sparsity=layerwise_model_sparsity,
            layerwise_target_sparsity=layerwise_target,
            layerwise_sparsity_loss=layerwise_loss,
        )


@dataclass
class CausalLMOutputWithPastAndSparsity(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None
    # Diagnostics
    expected_model_sparsity: Optional[torch.FloatTensor] = None
    lambda1: Optional[torch.FloatTensor] = None
    lambda2: Optional[torch.FloatTensor] = None
    expected_z_mean: Optional[torch.FloatTensor] = None
    expected_z_std: Optional[torch.FloatTensor] = None
    log_alpha_mean: Optional[torch.FloatTensor] = None
    log_alpha_std: Optional[torch.FloatTensor] = None
    # Layer-wise sparsity diagnostics
    layerwise_model_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_target_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_sparsity_loss: Optional[torch.FloatTensor] = None  # scalar


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class PawLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.model = LlamaModel(
            config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logit_block_size = int(os.environ.get("LOGIT_BLOCK_SIZE", 0))

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.model.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.model.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.model.reset_masks(value)

    @torch.no_grad()
    def get_sparsity(self):
        return self.model.get_sparsity()

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        self.model.reset_masks_with_stripe_pattern(width_1, width_2, start_with_keep)

    @torch.no_grad()
    def load_masks(self, masks):
        self.model.load_masks(masks)

    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        return self.model.round_masks_for_sparsity(target_sparsity)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_loss(self, hidden_states, labels):
        if (labels != -100).sum() == 0:
            return torch.tensor(
                0.0, device=hidden_states.device, dtype=hidden_states.dtype
            )
        min_len = min(hidden_states.size(0), labels.size(0))
        hidden_states = hidden_states[:min_len]
        labels = labels[:min_len]

        logits = self.lm_head(hidden_states)
        if len(logits.shape) > 2:
            logits = logits.transpose(-1, -2)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            reduction=("sum" if getattr(self, "token_scaled_loss", False) else "mean"),
        )

    def save_pretrained(self, *args, **kwargs):
        # First save the suggested threshold
        self.model._pre_save_get_threshold()
        return super().save_pretrained(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        return_token_losses: bool = False,
        shifted_labels: Optional[torch.LongTensor] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        erank_analysis_path: Optional[str] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if seq_lengths is not None:
            if inputs_embeds is not None:
                assert len(inputs_embeds.shape) == 2, (
                    "inputs_embeds should be a 2D tensor with `seq_lengths`"
                )
                # assert inputs_embeds.size(0) == seq_lengths.sum(), "inputs_embeds and seq_lengths should have the same batch size"
            else:
                assert len(input_ids.shape) == 1, (
                    "input_ids should be a 1D tensor with `seq_lengths`"
                )
                # assert input_ids.size(0) == seq_lengths.sum(), "input_ids and seq_lengths should have the same batch size"

            assert attention_mask is None or attention_mask.all().item(), (
                "attention_mask should be None or all ones for `seq_lengths`"
            )
            assert not use_cache, "use_cache is not supported with `seq_lengths`"
            cu_seqlens = F.pad(
                torch.cumsum(seq_lengths, dim=0, dtype=torch.torch.int32), (1, 0)
            )
            max_seqlen = seq_lengths.max().item()

            unpadded_lengths = (cu_seqlens, max_seqlen)
        elif (
            (attention_mask is not None) and (not attention_mask.all().item())
        ) and not use_cache:
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    inputs_embeds, attention_mask
                )
            else:
                bsz = input_ids.size(0)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen)
        else:
            unpadded_lengths = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # Use inference_mode during generation/prefill to avoid building graphs / caching many traces.
        if not self.training:
            # inference_mode is slightly faster and stronger than no_grad
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    unpadded_lengths=unpadded_lengths,
                    seq_parallel_group=seq_parallel_group,
                    target_sparsity=target_sparsity,
                    erank_analysis_path=erank_analysis_path,
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                unpadded_lengths=unpadded_lengths,
                seq_parallel_group=seq_parallel_group,
                target_sparsity=target_sparsity,
                erank_analysis_path=erank_analysis_path,
            )

        hidden_states = outputs[0]

        if seq_lengths is None and unpadded_lengths is not None:
            hidden_states = pad_input(hidden_states, unpad_indices, bsz, max_seqlen)

        if labels is not None or shifted_labels is not None:
            if shifted_labels is not None:
                labels = shifted_labels.reshape(-1)
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            else:
                labels = labels[..., 1:].reshape(-1)
                hidden_states = hidden_states[..., :-1, :].reshape(
                    -1, hidden_states.size(-1)
                )

            if self.logit_block_size > 0:
                num_valid_labels = (labels != -100).sum()
                hidden_states = torch.split(hidden_states, self.logit_block_size, dim=0)
                labels = torch.split(labels, self.logit_block_size, dim=0)

                if getattr(self, "token_scaled_loss", False):
                    loss = sum(
                        torch.utils.checkpoint.checkpoint(
                            self.compute_loss,
                            hidden_state_block,
                            label_block,
                            use_reentrant=False,
                        )
                        for hidden_state_block, label_block in zip(
                            hidden_states, labels
                        )
                    )
                else:
                    loss = sum(
                        ((label_block != -100).sum() / num_valid_labels)
                        * torch.utils.checkpoint.checkpoint(
                            self.compute_loss,
                            hidden_state_block,
                            label_block,
                            use_reentrant=False,
                        )
                        for hidden_state_block, label_block in zip(
                            hidden_states, labels
                        )
                    )
            else:
                loss = self.compute_loss(hidden_states, labels)

            logits = None
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndSparsity(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            model_sparsity=outputs.model_sparsity,
            target_sparsity=outputs.target_sparsity,
            sparsity_loss=outputs.sparsity_loss,
            expected_model_sparsity=outputs.expected_model_sparsity,
            lambda1=outputs.lambda1,
            lambda2=outputs.lambda2,
            expected_z_mean=outputs.expected_z_mean,
            expected_z_std=outputs.expected_z_std,
            log_alpha_mean=outputs.log_alpha_mean,
            log_alpha_std=outputs.log_alpha_std,
            layerwise_model_sparsity=outputs.layerwise_model_sparsity,
            layerwise_target_sparsity=outputs.layerwise_target_sparsity,
            layerwise_sparsity_loss=outputs.layerwise_sparsity_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
