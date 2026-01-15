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
"""PyTorch Qwen3 model."""

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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func, flash_attn_func
from flash_attn.bert_padding import unpad_input, pad_input
import math

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except ImportError:
    raise ImportError(
        "Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
    )
from block_sparse_attn import block_streaming_attn_func

from dataclasses import dataclass

from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4

logger = logging.get_logger(__name__)


class PawQwen3Config(Qwen3Config):
    def __init__(self, *args, **kwargs):
        self.local_window_size = kwargs.pop("local_window_size", 1024)
        self.disable_linear_regularization_term = kwargs.pop(
            "disable_linear_regularization_term", False
        )
        self.suggested_sparsity = kwargs.pop("suggested_sparsity", None)

        # Streaming
        self.toggle_type = kwargs.pop("toggle_type", "streaming")
        self.sink_size = kwargs.pop("sink_size", 128)

        # retrieval_mode
        self.retrieval_mode = kwargs.pop("retrieval_mode", "full")

        # Head Router
        self.pooling_mode = kwargs.pop("pooling_mode", "first_token")

        self.use_task_emb_for_mask = kwargs.pop("use_task_emb_for_mask", False)

        # TriangleMix
        self.triangle_n_last = kwargs.pop("triangle_n_last", 128)

        # ada-sparsity
        self.enable_ada_sparsity = kwargs.pop("enable_ada_sparsity", False)

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

        # 新增：top-k 注意力的超参（每个 query 仅保留前 k 个 key）
        self.topk_k = kwargs.pop("topk_k", 32)
        self.pooling_seq = kwargs.pop("pooling_seq", True)
        self.enable_lambda_task = kwargs.pop("enable_lambda_task", False)
        self.use_softmax = kwargs.pop("use_softmax", False)

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


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


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


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        interleaved=False,
        config: Optional[PawQwen3Config] = None,
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

        rope_q = apply_rotary_emb_func(
            q,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        rope_k = apply_rotary_emb_func(
            k,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return rope_q, rope_k


class Qwen3MLP(nn.Module):
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
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionRouter(nn.Module):
    def __init__(
        self,
        input_dim,
        num_key_value_heads,
        d_feature=128,
        use_task_emb=False,
        temp=1.0,
        hard=False,
        router_type="mlp",
        use_gumbel=True,
        learnable_temp=False,
        dropout=0.1,
        use_softmax=True,
        pooling_mode="ctx_q",
    ):
        super().__init__()
        self.num_kv = num_key_value_heads
        self.use_task_emb = use_task_emb
        self.router_type = router_type
        self.use_gumbel = use_gumbel
        self.learnable_temp = learnable_temp
        self.pooling_mode = pooling_mode
        self.use_softmax = use_softmax

        self.cls_feat_extractor = nn.Sequential(
            nn.Linear(d_feature, 4 * d_feature),
            nn.SiLU(),
            nn.Linear(4 * d_feature, d_feature),
        )

        if self.use_softmax:
            logger.info("using softmax for attention router")
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            logger.info("use sigmoid function for attention router")
            self.cls_router_head_agnostic = nn.Sequential(
                nn.Linear(d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1),
                nn.LayerNorm([self.num_kv, 1], elementwise_affine=False),
            )

        if self.use_task_emb:
            self.task_embedding = nn.Embedding(4, d_feature)

        # ---- learnable temperature ----
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temp)))
        else:
            self.tau = temp

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.cls_router_head_agnostic[0].weight, a=math.sqrt(5)
        )
        nn.init.zeros_(self.cls_router_head_agnostic[0].bias)

        nn.init.kaiming_uniform_(
            self.cls_router_head_agnostic[2].weight, a=math.sqrt(5)
        )
        nn.init.zeros_(self.cls_router_head_agnostic[2].bias)

        nn.init.zeros_(self.cls_router_head_agnostic[4].weight)
        nn.init.constant_(self.cls_router_head_agnostic[4].bias, 1.0)

    def forward(
        self,
        x,
        cu_seq_len=None,
        range_ids: torch.Tensor = None,
        task_ids: Optional[torch.Tensor] = None,
        current_tau: Optional[torch.Tensor] = None,
    ):
        """
        x: [cu_seq_len, H, D]
        cu_seq_len: [0, seq_len_1, seq_len_2 + seq_len_1, ...]
        range_ids: [B, 6]
        task_ids: [B]

        return:
            {
              'decisions': [B, H],
              'hard_decisions': [B, H, 2],
              'sparse_mask': [B, H],
              'logits': [B, H, 1]
            }
        """
        bsz = (cu_seq_len.shape[0] - 1) if cu_seq_len is not None else 1

        if self.pooling_mode == "first_token":
            if cu_seq_len is not None:
                pooled_latent = self._segment_pooling(
                    x, range_ids, ["first_token"], cu_seq_len
                )  # [B, H, D]
            else:
                pooled_latent = self._segment_pooling_single_batch(
                    x, range_ids, ["first_token"]
                )
        elif self.pooling_mode == "q":
            if cu_seq_len is not None:
                pooled_latent = self._segment_pooling(
                    x, range_ids, ["q"], cu_seq_len
                )  # [B, H, D]
            else:
                pooled_latent = self._segment_pooling_single_batch(x, range_ids, ["q"])
        elif self.pooling_mode == "ctx_q":
            if cu_seq_len is not None:
                B = cu_seq_len.shape[0] - 1
                H, D = x.shape[1:]
                sample_features = []
                for i in range(B):
                    x_s, x_e = cu_seq_len[i], cu_seq_len[i + 1]
                    seg_slice = x[x_s:x_e]  # [Ti, H, D]
                    seg_pooled = seg_slice.mean(dim=0)  # [H, D]
                    sample_features.append(seg_pooled)

                pooled_latent = torch.stack(sample_features, dim=0)
            else:
                target = torch.concat([x[:, :100, :], x[:, -100:, :]], dim=1).mean(
                    dim=1
                )
                pooled_latent = target  # [H, D]
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        if self.use_task_emb:
            if self.training:
                task_emb = self.task_embedding(task_ids)  # [B, D]
                task_emb_expanded = task_emb.unsqueeze(1)
                pooled_latent = pooled_latent + task_emb_expanded
            else:
                pooled_latent = pooled_latent

        pooled_hidden_states = self.cls_feat_extractor(pooled_latent)

        binary_logits = self.cls_router_head_agnostic(pooled_hidden_states)

        if self.learnable_temp:
            tau = torch.exp(self.log_temp).clamp(0.3, 1.0)
        else:
            tau = current_tau if current_tau is not None else self.tau

        u = torch.rand_like(binary_logits)
        eps = 1e-8
        g = -torch.log(-torch.log(u + eps) + eps)

        if not self.use_softmax:
            z_soft = torch.sigmoid((binary_logits + g) / tau)
            z_hard = (z_soft > 0.5).float()
            z = z_hard + (z_soft - z_soft.detach())  # [B, H, 1]
            entropy = -(
                z_soft * torch.log(z_soft + eps)
                + (1 - z_soft) * torch.log(1 - z_soft + eps)
            )
        else:
            z_soft = F.softmax(binary_logits, dim=-1)
            z_hard = torch.zeros_like(z_soft).scatter_(
                -1, z_soft.argmax(-1, keepdim=True), 1.0
            )
            z = z_hard + (z_soft - z_soft.detach())  # [B, H, 2]
            z = z[..., 1]  # [B, H]
            z_soft = z_soft[..., 1]
            z_soft = z_soft.unsqueeze(-1)
            z = z.unsqueeze(-1)
            entropy = -(z_soft * torch.log(z_soft + eps)).sum(dim=-1).mean()

        return {
            "pooled_hidden_states": pooled_hidden_states,  # [B, H, D]
            "decisions": z_soft,
            "hard_decisions": z_hard,
            "sparse_mask": z,  # [B, H], 这是一个 STE Tensor
            "logits": binary_logits,
            "entropy": entropy,
        }

    def _segment_pooling_single_batch(
        self, pooled_input: torch.Tensor, range_ids: torch.Tensor, segments: list
    ) -> torch.Tensor:
        B, S, H, D = pooled_input.shape
        pooled_features_list = []

        POOL_MAP = {
            "first_token": (0, 1),
            "ctx": (2, 3),
            "q": (4, 5),
            "a": (6, 7),
            "ctx_q": (2, 5),
        }
        for i in range(B):
            sample_features = []

            for seg in segments:
                start_idx, end_idx = POOL_MAP[seg]
                start, end = (
                    range_ids[i, start_idx : end_idx + 1].tolist()[0],
                    range_ids[i, start_idx : end_idx + 1].tolist()[-1],
                )
                if end >= start:
                    # seg_slice = pooled_input[i, start : end + 1, :, :]
                    start_slice = pooled_input[i, start : start + 100, :, :]
                    end_slice = pooled_input[i, end - 100 : end + 1, :, :]
                    combined_slice = torch.cat((start_slice, end_slice), dim=0)
                    seg_pooled = combined_slice.mean(dim=0)  # [H, D]
                else:
                    seg_pooled = torch.zeros(H, D, device=pooled_input.device)

                sample_features.append(seg_pooled)

            if sample_features:
                combined_feature = torch.stack(sample_features, dim=0).mean(
                    dim=0
                )  # [H, D]
            else:
                combined_feature = torch.zeros(H, D, device=pooled_input.device)

            pooled_features_list.append(combined_feature)

        return torch.stack(pooled_features_list, dim=0)  # [B, H, D]

    def _segment_pooling(
        self,
        x: torch.Tensor,
        range_ids: torch.Tensor,
        segments: list[str],
        cu_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): [cu_seqlen, H, D]
            range_ids (torch.Tensor): _description_
            segments (list[str]): _description_
            cu_seq_len (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        POOL_MAP = {
            "first_token": (0, 1),
            "ctx": (2, 3),
            "q": (4, 5),
            "a": (6, 7),
            "ctx_q": (2, 5),
        }

        B = cu_seq_len.shape[0] - 1
        H, D = x.shape[1:]
        pooled_features_list = []

        for i in range(B):
            sample_features = []
            x_s, x_e = cu_seq_len[i], cu_seq_len[i + 1]
            for seg in segments:
                start_idx, end_idx = POOL_MAP[seg]
                start, end = (
                    range_ids[i, start_idx : end_idx + 1].tolist()[0],
                    range_ids[i, start_idx : end_idx + 1].tolist()[-1],
                )

                if end >= start:
                    start_slice = pooled_input[i, start : start + 100, :, :]
                    end_slice = pooled_input[i, end - 99 : end + 1, :, :]
                    combined_slice = torch.cat((start_slice, end_slice), dim=0)
                    seg_pooled = combined_slice.mean(dim=0)  # [H, D]
                else:
                    seg_pooled = torch.zeros(H, D, device=x.device)

                sample_features.append(seg_pooled)

            if sample_features:
                combined_feature = torch.stack(sample_features, dim=0).mean(
                    dim=0
                )  # [H, D]
            else:
                combined_feature = torch.zeros(H, D, device=x.device)

            pooled_features_list.append(combined_feature)

        return torch.stack(pooled_features_list, dim=0)  # [B, H, D]


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PawQwen3Config,
        context_window_toggle: Optional[int] = 1024,
    ):
        """
        @context_window_toggle: if not None, the attention will be limited to a context window specified by this value
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

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

        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

        self._dtype = self.q_proj.weight.dtype
        self.attn_mask_log_alphas = nn.Parameter(
            torch.empty(self.num_key_value_heads, dtype=self._dtype)
        )
        self.attn_mask_log_alphas.data.normal_(
            mean=4.5, std=0.01
        )  # sigmoid(4.5) ≈ 0.989
        self.threshold_for_deterministic = None

        self.mask_allocator = AttentionRouter(
            input_dim=self.hidden_size,
            num_key_value_heads=self.num_key_value_heads,
            # head_dim = self.head_dim,
            d_feature=self.head_dim,
            use_task_emb=getattr(config, "use_task_emb_for_mask", False),
            temp=getattr(config, "mask_temp", 1.0),
            hard=getattr(config, "mask_hard_sample", False),
            pooling_mode=getattr(config, "pooling_mode", "first_token"),
            use_softmax=getattr(config, "use_softmax", False),
        )

        self.context_window_toggle = context_window_toggle

        self.toggle_type = config.toggle_type
        self.sink_blocks = (config.sink_size + 127) // 128
        self.local_blocks = (config.local_window_size + 127) // 128

        self.retrieval_mode = config.retrieval_mode

        if self.retrieval_mode == "xattn" or self.toggle_type == "streaming":
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
            self.topk_k = int(getattr(config, "topk_k", 2048))
            self.topk_q_chunk = int(os.environ.get("TOPK_Q_CHUNK", 128))
            self.topk_k_chunk = int(os.environ.get("TOPK_K_CHUNK", 4096))
        elif self.toggle_type == "xattn" or self.toggle_type == "full":
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
        elif self.toggle_type == "none":
            pass
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
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        v = self.v_proj(hidden_states).view(hidden_shape)
        has_layer_past = past_key_value is not None

        if not has_layer_past:
            if unpadded_lengths is not None:
                res = self.mask_allocator(
                    k, unpadded_lengths[0], range_ids, task_ids
                )
            else:
                res = self.mask_allocator(k, None, range_ids, task_ids)

            z_kv_batch, entropy, pooled_hidden_states = (
                res["sparse_mask"],
                res["entropy"],
                res["pooled_hidden_states"],
            )
            z_constrast = res["decisions"]

            if z_kv_batch.shape[-2] == self.num_key_value_heads:
                z_kv_batch = z_kv_batch.repeat_interleave(self.num_key_value_groups, 1)
        else:
            # decode
            z_kv_batch = past_key_value[2]

        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        if position_ids is not None:
            past_len += position_ids.min()

        q, k = self.rotary_emb(q, k, past_len, unpadded_lengths)

        kv = torch.stack([k, v], -3)

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
        past_key_value = (
            (past_kv, past_len + q.size(1), z_kv_batch) if use_cache else None
        )
        
        k = kv[:,:,0,:,:]
        v = kv[:,:,1,:,:]

        if not has_layer_past:
            bsz, seqlen, _, _ = q.size()
            if not torch.is_tensor(seqlen):
                seqlen = torch.tensor(seqlen, dtype=torch.int32, device=q.device)
                
            cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
            p_dropout = 0.0
            head_mask_type = torch.where(
                z_kv_batch[0, :, 0] == 1,
                torch.tensor(0, dtype=torch.int, device=z_kv_batch.device),
                torch.tensor(-1, dtype=torch.int, device=z_kv_batch.device),
            )

            streaming_info = torch.tensor([self.sink_blocks, self.local_blocks] * self.num_heads, device=q.device, dtype=torch.int32)
            attn_output = block_streaming_attn_func(
                q.squeeze(0).contiguous(),
                k.squeeze(0).contiguous(),
                v.squeeze(0).contiguous(),
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                seqlen,
                seqlen,
                p_dropout,
                deterministic=False,
                softmax_scale=None,
                is_causal=True,
                return_attn_probs=False,
            ).unsqueeze(0).contiguous()
        else:
            # bsz, seqlen, _, _ = k.size()
            # cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
            # cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
            # max_seqlen_q_ = 1
            # max_seqlen_k_ = seqlen
            # p_dropout = 0.0
            # head_mask_type = torch.where(
            #     z_kv_batch[0, :, 0] == 1,
            #     torch.tensor(0, dtype=torch.int, device=z_kv_batch.device),
            #     torch.tensor(-1, dtype=torch.int, device=z_kv_batch.device),
            # )
            # streaming_info = torch.tensor([self.sink_blocks, self.local_blocks] * self.num_heads, device=q.device, dtype=torch.int32)
            
            # attn_output = block_streaming_attn_func(
            #     q.squeeze(0).contiguous(),
            #     k.squeeze(0).contiguous(),
            #     v.squeeze(0).contiguous(),
            #     cu_seqlens_q,
            #     cu_seqlens_k,
            #     head_mask_type,
            #     streaming_info,
            #     max_seqlen_q_,
            #     max_seqlen_k_,
            #     p_dropout,
            #     deterministic=False,
            #     softmax_scale=None,
            #     is_causal=True,
            #     return_attn_probs=False,
            # ).unsqueeze(0).contiguous()
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),  # -1 means infinite context window
                softcap=0.0, # 0.0 means deactivated
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output.to(self.o_proj.weight.dtype))

        attn_weights = None

        # if not has_layer_past:
        #     print(f"task_ids: {task_ids}, head allocate: {[x.tolist() for x in z_kv_batch]}")

        # z: [B, H, 1] -> [B, H] -> [B]
        return (
            z_kv_batch.squeeze(-1).sum(dim=-1),
            None,
            None,
            None,
            attn_output,
            attn_weights,
            past_key_value,
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PawQwen3Config,
        context_window_toggle: Optional[int] = 4096,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            config=config, context_window_toggle=context_window_toggle
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
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
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
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
        (
            z_sum,
            entropy,
            pooled_hidden_states,
            z_constrast,
            hidden_states,
            self_attn_weights,
            present_key_value,
        ) = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
            segment_ids=segment_ids,
            range_ids=range_ids,
            task_ids=task_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (
            z_sum,
            entropy,
            pooled_hidden_states,
            z_constrast,
            hidden_states,
        )

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = PawQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

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
    # contrastive_loss
    contrastive_loss: Optional[torch.FloatTensor] = None
    head_contrastive_loss: Optional[torch.FloatTensor] = None
    log_z_loss: Optional[torch.FloatTensor] = None
    head_entropy: Optional[torch.FloatTensor] = None


class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: PawQwen3Config
    """

    def __init__(
        self,
        config: PawQwen3Config,
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
                Qwen3DecoderLayer(config, context_window_toggle=context_window_toggle)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
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

        if self.config.enable_lambda_task:
            self.num_tasks = 5
            self.sparsity_lambda1_task = nn.Parameter(
                torch.zeros(self.num_tasks, dtype=self._dtype)
            )
            self.sparsity_lambda2_task = nn.Parameter(
                torch.zeros(self.num_tasks, dtype=self._dtype)
            )
        else:
            self.sparsity_lambda1_task = None
            self.sparsity_lambda2_task = None

        self.threshold_for_deterministic = None
        if config.suggested_sparsity is not None:
            self.round_masks_for_sparsity(config.suggested_sparsity)

        self._erank_cache = {}
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def reset_parameters(self):
        if self.config.enable_lambda_task:
            self.sparsity_lambda1_task.data.copy_(
                torch.rand_like(self.sparsity_lambda1_task) * 0.5
            )
            self.sparsity_lambda2_task.data.copy_(
                torch.rand_like(self.sparsity_lambda2_task) * 0.5
            )

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
    def _get_avg_erank(self, path: str) -> torch.Tensor:
        key = os.path.abspath(path)
        if key in self._erank_cache:
            return self._erank_cache[key]
        erank_res = torch.load(key, map_location="cpu")
        print(f"Loaded e-rank results from {key}: {erank_res}")
        avg_erank = erank_res["avg_erank"]
        self._erank_cache[key] = avg_erank
        return avg_erank

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        if start_with_keep:
            value_1 = 10.0  # Some high value
            value_2 = -10.0  # Some low value
        else:
            value_1 = -10.0
            value_2 = 10.0
        for l, layer in enumerate(self.layers):
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

        self.load_masks(masks)

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
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        erank_analysis_path: Optional[str] = None,
        enable_contrastive_loss: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        compute_sparsity = self.training
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

        z_sum = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is not None and len(past_key_values) > idx:
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None

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
                    segment_ids=segment_ids,
                    range_ids=range_ids,
                    task_ids=task_ids,
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
                    segment_ids=segment_ids,
                    range_ids=range_ids,
                    task_ids=task_ids,
                )

            z_layer_sum, entropy, pooled_hidden_states, z_constrast, hidden_states = (
                layer_outputs[0],
                layer_outputs[1],
                layer_outputs[2],
                layer_outputs[3],
                layer_outputs[4],
            )

            z_layer_sum = z_layer_sum.to(hidden_states.device)

            if z_sum is None:
                z_sum = z_layer_sum
            else:
                z_sum = z_sum.to(z_layer_sum.device)
                z_sum = z_sum + z_layer_sum

            if use_cache:
                next_decoder_cache += (layer_outputs[6 if output_attentions else 5],)

            if output_attentions:
                all_self_attns += (layer_outputs[5],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        model_sparsity = 1 - (z_sum / self.total_num_heads)

        if not return_dict:
            # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, model_sparsity, target_sparsity, z_loss] if v is not None)
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    model_sparsity,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndSparsity(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            model_sparsity=model_sparsity,
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
    # contrastive_loss
    contrastive_loss: Optional[torch.FloatTensor] = None  # scalar
    head_contrastive_loss: Optional[torch.FloatTensor] = None
    # task_ids
    task_ids: Optional[torch.FloatTensor] = None
    log_z_loss: Optional[torch.FloatTensor] = None
    head_entropy: Optional[torch.FloatTensor] = None


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class PawQwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        enable_contrastive_loss=False,
    ):
        super().__init__(config)
        self.model = Qwen3Model(
            config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logit_block_size = int(os.environ.get("LOGIT_BLOCK_SIZE", 16384))
        self.enable_contrastive_loss = enable_contrastive_loss
        self.prefill_sparsity = None
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
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
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
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            attention_mask is not None and not use_cache and attention_mask.size(0) != 1
        ):
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    inputs_embeds, attention_mask
                )
            else:
                bsz = input_ids.size(0)
                tmp = input_ids.unsqueeze(-1)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    tmp, attention_mask
                )
                max_seqlen_for_pad_seq = attention_mask.size(-1)
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen)
        else:
            unpadded_lengths = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            segment_ids=segment_ids,
            range_ids=range_ids,
            task_ids=task_ids,
            enable_contrastive_loss=self.enable_contrastive_loss,
        )

        if input_ids.shape[1] > 1 and use_cache:
            self.prefill_sparsity = outputs.model_sparsity.detach()

        hidden_states = outputs[0]
        if seq_lengths is None and unpadded_lengths is not None:
            hidden_states = pad_input(
                hidden_states, unpad_indices, bsz, max_seqlen_for_pad_seq
            )
        if labels is not None or shifted_labels is not None:
            if shifted_labels is not None:
                labels = shifted_labels.reshape(-1)
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            else:
                labels = labels[..., 1:].reshape(-1).contiguous()
                hidden_states = (
                    hidden_states[..., :-1, :]
                    .reshape(-1, hidden_states.size(-1))
                    .contiguous()
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
                        ((label_block != -100).sum() / max(num_valid_labels.item(), 1))
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
            logits = self.lm_head(hidden_states[:, -1:, :])
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

        # These are static or need special handling during generation
        custom_keys = ["segment_ids", "range_ids", "task_ids"]
        for key in custom_keys:
            if key in kwargs:
                value = kwargs[key]
                # For segment_ids: may need to extend to match input_ids length (if generating)
                if key == "segment_ids" and value is not None:
                    # Extend segment_ids with answer segment ID (3) for new tokens
                    if value.shape[1] < input_ids.shape[1]:
                        pad_len = input_ids.shape[1] - value.shape[1]
                        pad_seg = torch.full(
                            (value.shape[0], pad_len),
                            fill_value=3,  # answer segment ID (as in training)
                            dtype=value.dtype,
                            device=value.device,
                        )
                        value = torch.cat([value, pad_seg], dim=1)
                model_inputs[key] = value

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


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen"
    tokenizer = AutoTokenizer.from_pretrained(model_path)



    model = PawQwen3ForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
    ).to("cuda:4")


    #text = "Artificial intelligence is transforming the world in profound ways, from healthcare and education to transportation and entertainment. Researchers continue to push the boundaries of what machines can learn and understand, raising both exciting opportunities and important ethical questions about the future of human-AI collaboration."

    text = "<|im_start|>user\nBelow is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n1. undertaker 2. dashboard 3. undertaker 4. mouth 5. health 6. real 7. saviour 8. raiment 9. silly 10. silly 11. company 12. appetizer 13. ovary 14. silly 15. rations 16. dashboard 17. pension 18. undertaker 19. appetizer 20. boutique 21. company 22. limestone 23. brush 24. health 25. saviour 26. eurocentrism 27. mukluk 28. gift 29. silly 30. finer 31. pension 32. real 33. undertaker 34. health 35. gearshift 36. undertaker 37. upgrade 38. people 39. health 40. ladle 41. ovary 42. carbon 43. octavo 44. transom 45. essay 46. ovary 47. real 48. exceed 49. health 50. donut 51. real 52. people 53. appetizer 54. mukluk 55. donut 56. expectancy 57. dashboard 58. voiceless 59. scattered 60. silly 61. expectancy 62. health 63. voiceless 64. octavo 65. dashboard 66. crisp 67. gearshift 68. dashboard 69. appetizer 70. voiceless 71. octavo 72. donut 73. boutique 74. dashboard 75. octavo 76. crisp 77. western 78. gearshift 79. exceed 80. silly 81. voiceless 82. raiment 83. people 84. diving 85. silly 86. silly 87. cabana 88. vase 89. undertaker 90. gift 91. cabana 92. webpage 93. upgrade 94. carbon 95. dashboard 96. octavo 97. brush 98. upgrade 99. brush 100. carbon 101. vase 102. cabana 103. rations 104. limestone 105. voiceless 106. western 107. scattered 108. rations 109. silly 110. voiceless 111. upgrade 112. mouth 113. gift 114. octavo 115. undertaker 116. raiment 117. octavo 118. dashboard 119. appetizer 120. expectancy 121. appetizer 122. gift 123. pension 124. real 125. finer 126. upgrade 127. webpage 128. mouth 129. saviour 130. appetizer 131. boutique 132. exceed 133. dashboard 134. voiceless 135. ellipse 136. gift 137. octavo 138. undertaker 139. appetizer 140. health 141. real 142. upgrade 143. diving 144. transom 145. upgrade 146. gift 147. voiceless 148. upgrade 149. dashboard 150. gift 151. ellipse 152. scattered 153. real 154. essay 155. octavo 156. western 157. gift 158. transom 159. health 160. upgrade 161. upgrade 162. ladle 163. ellipse 164. real 165. diving 166. ladle 167. finer 168. health 169. eurocentrism 170. vase 171. mukluk 172. company 173. octavo 174. appetizer 175. silly 176. gift 177. voiceless 178. limestone 179. essay 180. undertaker 181. crisp 182. eurocentrism 183. real 184. appetizer 185. gift 186. undertaker 187. health 188. real 189. voiceless 190. webpage\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:1. health 2. upgrade 3. undertaker 4. silly 5. dashboard 6. appetizer 7. real 8. gift 9. voiceless 10. octavo\nBelow is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n1. tramp 2. possess 3. eyeball 4. endure 5. delight 6. skate 7. littleneck 8. wire 9. chicory 10. solid 11. squid 12. detention 13. plaintiff 14. wet-bar 15. adapter 16. expertise 17. external 18. warm-up 19. chain 20. commuter 21. stopwatch 22. commitment 23. accommodation 24. astronomy 25. misfit 26. campus 27. stir 28. heron 29. tongue 30. armadillo 31. pneumonia 32. soothsay 33. sea 34. fibroblast 35. officiate 36. harmonise 37. mythology 38. catalog 39. reliability 40. step-daughter 41. armadillo 42. homeland 43. galley 44. overweight 45. prey 46. unfasten 47. fiction 48. tutu 49. limitation 50. ship 51. survey 52. rape 53. cable 54. toll 55. connotation 56. anklet 57. plant 58. foot 59. supreme 60. moment 61. reindeer 62. adult 63. dramatic 64. espadrille 65. darn 66. odd 67. calculator 68. junker 69. fibroblast 70. strawberry 71. delight 72. bowtie 73. postfix 74. selling 75. heater 76. footstep 77. plant 78. pard 79. rely 80. graduation 81. armadillo 82. east 83. chorus 84. step 85. training 86. pint 87. prophet 88. fang 89. payee 90. liquid 91. reliability 92. rim 93. harbour 94. intensify 95. tight 96. sulfur 97. session 98. exam 99. climb 100. barium 101. nectarine 102. pup 103. endurable 104. wash 105. picturesque 106. helicopter 107. hydrofoil 108. skyscraper 109. lapdog 110. employment 111. light 112. inquisitive 113. terrible 114. teaching 115. boil 116. grab 117. solid 118. midline 119. odometer 120. restriction 121. rib 122. courage 123. gossip 124. hurried 125. utilization 126. thunderstorm 127. astrolabe 128. discreet 129. whirl 130. mattress 131. shaker 132. inch 133. pail 134. highway 135. precede 136. physics 137. mayor 138. fierce 139. exercise 140. leave 141. emission 142. cow 143. screw-up 144. mallard 145. launch 146. taco 147. cause 148. thyme 149. precision 150. delight 151. overweight 152. solidity 153. sharp 154. annoying 155. botany 156. add 157. pint 158. violation 159. lamb 160. jaded 161. pound 162. rally 163. republic 164. feedback 165. knitting 166. exhibition 167. going 168. filthy 169. designer 170. marble 171. cemetery 172. western 173. proponent 174. moonlight 175. dramaturge 176. load 177. frock 178. paranoia 179. step 180. extract 181. examination 182. offering 183. senior 184. guarded 185. presume 186. offense 187. gobbler 188. methodology 189. midwife 190. catalog 191. hit 192. wet-bar 193. gamy 194. attorney 195. warden 196. postfix 197. yam 198. outlook 199. cling 200. trove 201. differential 202. detective 203. news 204. respond 205. seminar 206. abusive 207. personnel 208. brow 209. gesture 210. bank 211. pawnshop 212. squid 213. beer 214. guarded 215. maintainer 216. anger 217. turnover 218. remain 219. alb 220. eyrie 221. intelligence 222. deck 223. enquiry 224. plot 225. observation 226. sardine 227. hydrofoil 228. spelling 229. spark 230. suspenders 231. half 232. connection 233. campus 234. technician 235. delight 236. photographer 237. bandolier 238. leading 239. loutish 240. shackle 241. part 242. chem 243. orangutan 244. artist 245. presentation 246. store 247. tenuous 248. brow 249. apprehension 250. underestimate 251. abacus 252. gaiters 253. walker 254. envious 255. verdict 256. councilperson 257. spleen 258. waste 259. finicky 260. operating 261. misfit 262. basketball 263. publicize 264. countryside 265. cross-stitch 266. enquiry 267. joy 268. chem 269. sidewalk 270. talent 271. endoderm 272. command 273. doorpost 274. armadillo 275. filthy 276. modify 277. strange 278. ore 279. convention 280. songbird 281. cicada 282. supreme 283. fiddle 284. intervenor 285. set 286. supreme 287. sardine 288. obesity 289. tank-top 290. plant 291. prune 292. catsup 293. vibrissae 294. camper 295. luxuriant 296. finisher 297. inflation 298. painting 299. stepdaughter 300. ship 301. cultured 302. sail 303. bounce 304. latte 305. rainy 306. stealth 307. neglect 308. traveler 309. dial 310. zoom 311. atrium 312. abusive 313. facility 314. elimination 315. drummer 316. avoid 317. wind 318. treasury 319. migration 320. diving 321. breezy 322. morale 323. research 324. lycra 325. peace 326. tangerine 327. suspension 328. chap 329. noodles 330. chime 331. homicide 332. hornet 333. respond 334. deduce 335. takeover 336. ischemia 337. beret 338. name 339. supreme 340. difference 341. tribe 342. tangy 343. flat 344. lightscreen 345. deputy 346. meteor 347. oats 348. winery 349. campus 350. silkworm 351. blister 352. prince 353. bourgeoisie 354. law 355. chip 356. quartz 357. governance 358. recruit 359. local 360. tranquil 361. rutabaga 362. bakeware 363. mutt 364. collapse 365. tag 366. layer 367. obstacle 368. yam 369. trance 370. supreme 371. plant 372. abusive 373. exhibition 374. tremor 375. luncheonette 376. freelance 377. singular 378. leek 379. walker 380. ziggurat 381. freak 382. comfort 383. crest 384. drill 385. designer 386. plant 387. plant 388. rinse 389. policy 390. push 391. chain 392. submitter 393. painful 394. row 395. estate 396. e-book 397. comic 398. passion 399. decision-making 400. witch-hunt 401. shaw 402. rooster 403. afterthought 404. commuter 405. remember 406. tangy 407. part 408. doggie 409. fierce 410. afraid 411. wire 412. reamer 413. anticipate 414. turf 415. student 416. mundane 417. landform 418. aware 419. command 420. zoom 421. helmet 422. leptocephalus 423. minimum 424. going 425. channel 426. goggles 427. obesity 428. deputy 429. thaw 430. kept 431. barstool 432. simplification 433. aardvark 434. workshop 435. pilaf 436. toe 437. trachoma 438. arrange 439. subsidiary 440. warning 441. toothsome 442. murder 443. ark 444. nail 445. counsellor 446. puppet 447. ammunition 448. measles 449. gobbler 450. heirloom 451. gerbil 452. opponent 453. delight 454. campus 455. pint 456. sustenance 457. savings 458. backyard 459. lip 460. adverb 461. decorate 462. abusive 463. marker 464. kamikaze 465. cable 466. sandal 467. hazel 468. strawberry 469. primary 470. pinkie 471. heater 472. supreme 473. stare 474. latte 475. venti 476. flight 477. statistic 478. sectional 479. deep 480. photo 481. everybody 482. plant 483. dentist 484. disdain 485. codling 486. cord 487. badge 488. dance 489. effacement 490. playground 491. pinch 492. sombrero 493. doubter 494. verdict 495. tow-truck 496. feedback 497. smile 498. pint 499. dignity 500. exercise 501. cloudy 502. production 503. misfit 504. axis 505. tramp 506. supply 507. seashore 508. performance 509. plant 510. minute 511. mesenchyme 512. breakthrough 513. casket 514. polyester 515. upbeat 516. physics 517. glossy 518. depth 519. possess 520. sidewalk 521. wilderness 522. paramecium 523. joyous 524. sardine 525. wobble 526. dilution 527. protect 528. enthusiastic 529. delight 530. commitment 531. policy 532. successful 533. precede 534. transplantation 535. calculator 536. endoderm 537. tremor 538. innate 539. information 540. contractor 541. gossip 542. tight 543. abusive 544. criterion 545. pint 546. assurance 547. possession 548. copying 549. spoil 550. measles 551. hurdler 552. walker 553. hub 554. tooth 555. keeper 556. faulty 557. underestimate 558. supreme 559. estimate 560. importance 561. fame 562. rocket 563. anger 564. devastation 565. omission 566. hook 567. retailer 568. delight 569. scallion 570. meteor 571. fertile 572. quaint 573. nitrogen 574. unaccountable 575. row 576. falling-out 577. reign 578. draconian 579. shaker 580. pronoun 581. migration 582. rely 583. equal 584. magnitude 585. wend 586. build 587. multiply 588. faulty 589. political 590. plant 591. wide-eyed 592. rebellious 593. ziggurat 594. figure 595. plain 596. eggplant 597. harbour 598. row 599. measure 600. snore 601. time 602. rescue 603. disco 604. finger 605. neighbourhood 606. fierce 607. climb 608. be 609. minnow 610. discharge 611. casket 612. smile 613. eligibility 614. careful 615. suppose 616. large 617. deposit 618. kebab 619. porcelain 620. hellcat 621. bourgeoisie 622. depth 623. clarify 624. git 625. smolt 626. supreme 627. row 628. row 629. delight 630. time 631. hood 632. accordance 633. finisher 634. instruction 635. boatload 636. buzzard 637. hose 638. espadrille 639. cause 640. socialism 641. hit 642. mineshaft 643. heavenly 644. footstool 645. plant 646. man 647. prince 648. tablet 649. facility 650. perspective 651. streamline 652. morale 653. potato 654. excursion 655. upbeat 656. blow 657. credible 658. pint 659. pinch 660. nun 661. cup 662. joy 663. highlight 664. campus 665. watch 666. add 667. astronomy 668. mayor 669. hypnotic 670. armadillo 671. mud 672. violet 673. spoil 674. fright 675. analogue 676. bake 677. industrious 678. terrorist 679. tribe 680. drummer 681. specialist 682. labourer 683. painting 684. delight 685. barrage 686. warm-up 687. be 688. commuter 689. finicky 690. spear 691. nutty 692. navigation 693. fisherman 694. normalization 695. project 696. crusader 697. filly 698. puppet 699. flood 700. research 701. abusive 702. video 703. waterspout 704. mallard 705. disposer 706. row 707. spark 708. leek 709. gelatin 710. soundness 711. trove 712. intestine 713. blazer 714. noun 715. plant 716. wiring 717. overweight 718. teaching 719. toot 720. spiffy 721. bout 722. wiring 723. delight 724. laundry 725. wilderness 726. allegation 727. wash 728. suppression 729. knitting 730. alpenglow 731. grab 732. malice 733. sadness 734. tool 735. limb 736. methodology 737. note 738. debt 739. news 740. sardine 741. historical 742. inflation 743. hair 744. repulsive 745. arrange 746. flat 747. dishwasher 748. hutch 749. hiking 750. itchy 751. abusive 752. connection 753. ozone 754. warden 755. beer 756. plant 757. tapioca 758. subsequent 759. trainer 760. trapezoid 761. junker 762. chip 763. row 764. hometown 765. minor-league 766. mandate 767. meek 768. abusive 769. thrust 770. lightscreen 771. perfection 772. campus 773. sardine 774. gamebird 775. boatload 776. recess 777. workshop 778. belief 779. pantry 780. throne 781. treasury 782. encourage 783. production 784. helmet 785. breezy 786. jazz 787. revolution 788. silkworm 789. neck 790. sun 791. sustenance 792. crackers 793. elimination 794. cenotaph 795. gelatin 796. retailer 797. tongue 798. porpoise 799. fisherman 800. significance 801. plant 802. hosiery 803. streamline 804. sewer 805. layer 806. session 807. keeper 808. hyphenation 809. parser 810. quail 811. anatomy 812. knit 813. sharp 814. fright 815. parser 816. peace 817. construction 818. reset 819. slaw 820. drummer 821. adult 822. encyclopedia 823. knit 824. walker 825. sewer 826. bruise 827. sty 828. scenario 829. x-rated 830. friend 831. trachoma 832. juvenile 833. gas 834. walker 835. environment 836. inspire 837. pipe 838. pint 839. softdrink 840. profession 841. chime 842. millimeter 843. delight 844. campus 845. catastrophe 846. normalization 847. upper 848. airline 849. alliance 850. abusive 851. emission 852. encourage 853. allegation 854. affiliate 855. councilperson 856. cymbal 857. oregano 858. specialty 859. wilderness 860. goggles 861. mattress 862. cigarette 863. coalition 864. dance 865. empty 866. puppet 867. consumption 868. gift 869. anklet 870. anxious 871. fame 872. prey 873. homonym 874. flat 875. mythology 876. buckle 877. mirror 878. competence 879. standpoint 880. cinder 881. solidarity 882. materialistic 883. gyro 884. utilization 885. postfix 886. dimension 887. moan 888. hose 889. miter 890. operate 891. judge 892. almanac 893. classification 894. lentil 895. bloom 896. flume 897. silly 898. armadillo 899. mallard 900. ashamed 901. rose 902. moan 903. subgroup 904. defective 905. plant 906. tag 907. passenger 908. perfection 909. runner 910. grab 911. smelly 912. strait 913. spelling 914. clavicle 915. ballpark 916. hometown 917. ordinary 918. snob 919. elimination 920. freezing 921. harbour 922. modem 923. frontier 924. polenta 925. try 926. elderberry 927. astrolabe 928. darn 929. trance 930. raisin 931. captain 932. invoice 933. thunderstorm 934. jockey 935. cuisine 936. implication 937. guava 938. undress 939. skate 940. lyrics 941. malice 942. political 943. project 944. trophy 945. tech 946. spring 947. marked 948. year 949. proponent 950. implication 951. chap 952. dignity 953. cicada 954. happiness 955. everybody 956. interview 957. limb 958. western 959. environment 960. buckle 961. browsing 962. sunroom 963. barrage 964. tasty 965. scribble 966. pint 967. aware 968. exile 969. row 970. tambourine 971. soulmate 972. record 973. rose 974. spank 975. window 976. distinct 977. dressing 978. rainy 979. ping 980. laundry 981. toe 982. various 983. plate 984. cuisine 985. applause 986. nice 987. employment 988. armadillo 989. plot 990. leg 991. closing 992. sardine 993. pink 994. cattle 995. tomatillo 996. step 997. limb 998. observation 999. malice 1000. installation 1001. pants 1002. grouse 1003. stare 1004. tambourine 1005. transfer 1006. manicure 1007. prize 1008. abusive 1009. divalent 1010. pilaf 1011. disability 1012. mutt 1013. unpack 1014. fiber 1015. pea 1016. streamline 1017. pink 1018. packaging 1019. defective 1020. tribe 1021. zippy 1022. command 1023. panoramic 1024. zippy 1025. spring 1026. shelter 1027. marxism 1028. income 1029. runner 1030. toll 1031. toothpaste 1032. marker 1033. delight 1034. seminar 1035. storey 1036. muffin 1037. sardine 1038. eponym 1039. bend 1040. soda 1041. hawk 1042. sardine 1043. potato 1044. noun 1045. festival 1046. pheasant 1047. cemetery 1048. cupola 1049. skip 1050. midwife 1051. prevalence 1052. allegation 1053. maintainer 1054. large 1055. congressperson 1056. tooth 1057. expansion 1058. asterisk 1059. walker 1060. supreme 1061. wealthy 1062. brisket 1063. casket 1064. push 1065. washbasin 1066. venti 1067. celsius 1068. watch 1069. postage 1070. establishment 1071. quotation 1072. enlist 1073. training 1074. paint 1075. fright 1076. row 1077. campus 1078. meteor 1079. nutrient 1080. measles 1081. stylus 1082. share 1083. timer 1084. spank 1085. empty 1086. neighbourhood 1087. autoimmunity 1088. mayonnaise 1089. bake 1090. soothsay 1091. eyeliner 1092. geek 1093. frock 1094. eyebrow 1095. discretion 1096. titanium 1097. chard 1098. abusive 1099. homeownership 1100. medicine 1101. taco 1102. unarmed 1103. glib 1104. underneath 1105. pupil 1106. lady 1107. armadillo 1108. fortune 1109. try 1110. walker 1111. terrorism 1112. reign 1113. scenario 1114. opera 1115. wall 1116. contractor 1117. obstacle 1118. metro 1119. cashier 1120. lieu 1121. bath 1122. asset 1123. physics 1124. board 1125. sorghum 1126. tranquil 1127. minor-league 1128. feedback 1129. grapefruit 1130. tablet 1131. quail 1132. walker 1133. teammate 1134. dogwood 1135. redirect 1136. competence 1137. crucifixion 1138. accommodation 1139. cloudy 1140. opposite 1141. taco 1142. miter 1143. marionberry 1144. bout 1145. maintainer 1146. examination 1147. rocket 1148. soulmate 1149. east 1150. gig 1151. boatload 1152. base 1153. pinkie 1154. delight 1155. zealous 1156. timer 1157. contact lens 1158. omelet 1159. consider 1160. source 1161. reader 1162. abacus 1163. step-daughter 1164. bill 1165. instant 1166. pronoun 1167. cling 1168. policeman 1169. blouse 1170. gerbil 1171. amazement 1172. journalism 1173. cactus 1174. washbasin 1175. gaze 1176. hunt 1177. smelly 1178. mentor 1179. sardine 1180. diversity 1181. heavenly 1182. quest 1183. anxious 1184. apple 1185. drum 1186. endoderm 1187. flanker 1188. decorate 1189. length 1190. reversal 1191. inquisitive 1192. midline 1193. homeland 1194. hoof 1195. diving 1196. be 1197. hook 1198. underneath 1199. bruise 1200. row 1201. envy 1202. spin 1203. cause 1204. glass 1205. invoice 1206. prairie 1207. noun 1208. shopping 1209. basketball 1210. shine 1211. knuckle 1212. upper 1213. grouse 1214. supreme 1215. medicine 1216. client 1217. deposition 1218. flippant 1219. hydrocarbon 1220. knotty 1221. campus 1222. agonizing 1223. webpage 1224. sandpaper 1225. snarl 1226. zoom 1227. x-rated 1228. hellcat 1229. understood 1230. collaboration 1231. beginner 1232. maelstrom 1233. reader 1234. walker 1235. consider 1236. collaboration 1237. modify 1238. installation 1239. swamp 1240. submitter 1241. collapse 1242. yielding 1243. rib 1244. crest 1245. station 1246. noodles 1247. spear 1248. contact lens 1249. tram 1250. millimeter 1251. git 1252. walker 1253. espadrille 1254. poison 1255. chow 1256. alpenglow 1257. nutrient 1258. trustee 1259. porcelain 1260. heron 1261. hair 1262. kept 1263. couple 1264. timeline 1265. comedy 1266. profit 1267. tension 1268. set 1269. gravy 1270. campus 1271. external 1272. scholar 1273. grasshopper 1274. evaporation 1275. side 1276. stylus 1277. reset 1278. fibroblast 1279. astonishing 1280. leptocephalus 1281. cry 1282. see 1283. walker 1284. glossy 1285. apology 1286. artery 1287. cemetery 1288. cord 1289. terrible 1290. guitar 1291. migration 1292. athletics 1293. skunk 1294. x-rated 1295. shine 1296. sardine 1297. opponent 1298. sardine 1299. armadillo 1300. channel 1301. bounce 1302. tutu 1303. eyeball 1304. hypnotic 1305. supreme 1306. snore 1307. walker 1308. envy 1309. commotion 1310. mine 1311. classification 1312. bank 1313. sectional 1314. pumpernickel 1315. paramecium 1316. user 1317. windage 1318. primary 1319. rebellion 1320. soda 1321. importance 1322. extract 1323. orangutan 1324. intervenor 1325. fresco 1326. campus 1327. terminology 1328. record 1329. rope 1330. light 1331. walker 1332. family 1333. yielding 1334. pattern 1335. floodplain 1336. rebel 1337. terrace 1338. tambourine 1339. magnificent 1340. ounce 1341. exam 1342. determined 1343. airship 1344. heap 1345. sonnet 1346. development 1347. anger 1348. diving 1349. training 1350. conference 1351. emphasis 1352. gloom 1353. pattern 1354. beaver 1355. sardine 1356. atom 1357. festival 1358. rabbit 1359. prizefight 1360. nightingale 1361. sermon 1362. neighbour 1363. sardine 1364. mocha 1365. chalk 1366. step-sister 1367. appetite 1368. rescue 1369. sustenance 1370. suspenders 1371. brisket 1372. octave 1373. pneumonia 1374. upward 1375. sibling 1376. invincible 1377. basketball 1378. gesture 1379. frontier 1380. solidity 1381. luncheonette 1382. side 1383. drainage 1384. throne 1385. shallows 1386. personality 1387. doggie 1388. smiling 1389. plover 1390. sadness 1391. sail 1392. competitor 1393. lipstick 1394. sty 1395. estrogen 1396. wheat 1397. recollection 1398. set 1399. mud 1400. steady 1401. pint 1402. neighbour 1403. howitzer 1404. trench 1405. estrogen 1406. ruffle 1407. unfasten 1408. supreme 1409. celsius 1410. rehabilitate 1411. sniffle 1412. insert 1413. catastrophe 1414. dramatic 1415. pawnshop 1416. solidarity 1417. pronoun 1418. ashamed 1419. cartload 1420. enhance 1421. alert 1422. development 1423. swine 1424. lycra 1425. mutt 1426. custom 1427. fanny-pack 1428. alpenhorn 1429. cofactor 1430. talent 1431. status 1432. pipe 1433. fine 1434. poetry 1435. alb 1436. carnival 1437. instrumentalist 1438. story-telling 1439. starter 1440. nice 1441. vitamin 1442. ammunition 1443. offering 1444. tough-guy 1445. outlay 1446. restriction 1447. jazzy 1448. confusion 1449. float 1450. desktop 1451. pen 1452. large 1453. couple 1454. hope 1455. sloppy 1456. minor-league 1457. motivate 1458. pinecone 1459. paperwork 1460. fang 1461. cigarette 1462. motive 1463. mile 1464. portrait 1465. gyro 1466. earplug 1467. magnet 1468. hyena 1469. polenta 1470. quartz 1471. ship 1472. c-clamp 1473. elderberry 1474. base 1475. spank 1476. waterspout 1477. webpage 1478. pinecone 1479. sorghum 1480. skunk 1481. flight 1482. earth 1483. patient 1484. protect 1485. stepdaughter 1486. development 1487. rabbit 1488. cracker 1489. piracy 1490. step-sister 1491. courage 1492. minimum 1493. chicken 1494. ozone 1495. heron 1496. investment 1497. invincible 1498. termination 1499. athletics 1500. year 1501. throne 1502. search 1503. rebel 1504. joyous 1505. protective 1506. abuse 1507. pint 1508. defeated 1509. rostrum 1510. scaffold 1511. winery 1512. estate 1513. holder 1514. reader 1515. kayak 1516. sty 1517. promise 1518. terracotta 1519. agonizing 1520. native 1521. naive 1522. tram 1523. doorpost 1524. production 1525. bake 1526. remnant 1527. row 1528. abundant 1529. handicap 1530. sardine 1531. lose 1532. raise 1533. stopwatch 1534. angina 1535. delight 1536. favorite 1537. abusive 1538. campus 1539. section 1540. afraid 1541. plant 1542. shaker 1543. thrust 1544. empire 1545. walker 1546. supreme 1547. insert 1548. anticipate 1549. prune 1550. ignore 1551. ordinary 1552. opponent 1553. wire 1554. time 1555. globe 1556. airship 1557. article 1558. assertion 1559. comic 1560. lemon 1561. highway 1562. harm 1563. abusive 1564. freak 1565. campus 1566. republican 1567. dresser 1568. thought 1569. family 1570. toot 1571. squeegee 1572. connotation 1573. clarify 1574. quotation 1575. glib 1576. plant 1577. dressing 1578. mobster 1579. taro 1580. counsellor 1581. share 1582. fascinated 1583. possession 1584. floodplain 1585. lady 1586. promote 1587. tool 1588. oncology 1589. takeover 1590. cord 1591. crayfish 1592. likelihood 1593. omelet 1594. cinder 1595. dispense 1596. compile 1597. midwife 1598. supreme 1599. pint 1600. supply 1601. row 1602. spend 1603. venti 1604. postage 1605. praised 1606. messenger 1607. adapt 1608. hood 1609. mandate 1610. sweatsuit 1611. admin 1612. description 1613. armadillo 1614. beneficiary 1615. intelligence 1616. squid 1617. eliminate 1618. leprosy 1619. odd 1620. illusion 1621. memory 1622. refrigerator 1623. minnow 1624. row 1625. cashier 1626. tasty 1627. corduroy 1628. hellcat 1629. subject 1630. residence 1631. ram 1632. division 1633. tortoise 1634. paranoia 1635. delight 1636. amazement 1637. dentist 1638. exam 1639. salami 1640. upper 1641. squatter 1642. sideboard 1643. skylight 1644. precede 1645. suite 1646. abusive 1647. bend 1648. quotation 1649. element 1650. disarmament 1651. lychee 1652. sensitive 1653. embossing 1654. rape 1655. campus 1656. archaeology 1657. terror 1658. competitor 1659. knitting 1660. sultan 1661. economic 1662. knotty 1663. chair 1664. novel 1665. familiar 1666. gravy 1667. halloween 1668. proceedings 1669. deck 1670. veldt 1671. sharp 1672. nursing 1673. spin 1674. oregano 1675. ukulele 1676. fatigues 1677. pint 1678. rabbit 1679. grip 1680. isolation 1681. excursion 1682. upbeat 1683. currant 1684. row 1685. cenotaph 1686. briefing 1687. leave 1688. ectodermal 1689. atrium 1690. subsidiary 1691. maracas 1692. pennant 1693. blow 1694. unusual 1695. plant 1696. inspire 1697. tank-top 1698. article 1699. economic 1700. wealthy 1701. socialism 1702. lose 1703. pint 1704. stuff 1705. suite 1706. mapping 1707. ram 1708. western 1709. unusual 1710. lynx 1711. magnet 1712. starter 1713. watermelon 1714. neighbourhood 1715. pint 1716. precedent 1717. caddy 1718. pard 1719. campus 1720. raisin 1721. outlay 1722. magnitude 1723. tape 1724. molasses 1725. record 1726. howitzer 1727. fiction 1728. verify 1729. enthusiastic 1730. dial 1731. itchy 1732. heap 1733. rib 1734. mariachi 1735. unarmed 1736. spokeswoman 1737. nail 1738. packaging 1739. fascinated 1740. polyester 1741. delight 1742. discovery 1743. animal 1744. playground 1745. airfare 1746. throat 1747. fertile 1748. jet 1749. shocking 1750. jaded 1751. hope 1752. saw 1753. detection 1754. frock 1755. submitter 1756. drill 1757. steward 1758. selling 1759. dispense 1760. better 1761. squatter 1762. construction 1763. waffle 1764. scale 1765. gauntlet 1766. mutation 1767. trench 1768. congressperson 1769. delight 1770. section 1771. steady 1772. remark 1773. skyscraper 1774. adorable 1775. client 1776. governance 1777. kamikaze 1778. dressing 1779. hiking 1780. restored 1781. figure 1782. oregano 1783. gate 1784. hosiery 1785. minibus 1786. board 1787. nutty 1788. grasshopper 1789. camper 1790. tempo 1791. pint 1792. ghost 1793. supreme 1794. paperback 1795. beaver 1796. rostrum 1797. tomorrow 1798. pup 1799. licensing 1800. vibrissae 1801. information 1802. oats 1803. slider 1804. odd 1805. felony 1806. promote 1807. thrush 1808. freak 1809. brainy 1810. importance 1811. league 1812. verify 1813. screw-up 1814. terminology 1815. internet 1816. mouth 1817. celsius 1818. keyboarding 1819. scientific 1820. peanut 1821. crest 1822. sadness 1823. scene 1824. pint 1825. harm 1826. invoice 1827. marked 1828. alpenhorn 1829. messenger 1830. wall 1831. cluster 1832. economic 1833. adapter 1834. arrogance 1835. eyebrow 1836. peep 1837. SUV 1838. accordance 1839. subset 1840. terrace 1841. element 1842. spoon 1843. adapt 1844. minimum 1845. cenotaph 1846. ascent 1847. division 1848. violation 1849. abusive 1850. fanny-pack 1851. sardine 1852. chutney 1853. tag 1854. arrogance 1855. exaggeration 1856. warning 1857. criterion 1858. doubter 1859. rehabilitate 1860. leprosy 1861. connection 1862. barstool 1863. rinse 1864. abrupt 1865. electricity 1866. specialty 1867. jazz 1868. orangutan 1869. tiger 1870. cross-stitch 1871. mobster 1872. cone 1873. festival 1874. float 1875. cone 1876. timer 1877. luxuriant 1878. gift 1879. commotion 1880. curiosity 1881. tape 1882. assertion 1883. rescue 1884. falling-out 1885. solicitation 1886. enlist 1887. fatigues 1888. prow 1889. gig 1890. better 1891. tomorrow 1892. kept 1893. paperwork 1894. simplification 1895. sunlight 1896. retrospective 1897. estate 1898. petitioner 1899. skylight 1900. station 1901. obstacle 1902. hub 1903. lipstick 1904. watermelon 1905. chime 1906. starter 1907. grief 1908. takeover 1909. ski 1910. recess 1911. ischemia 1912. filthy 1913. oafish 1914. bend 1915. cluster 1916. waste 1917. gloom 1918. screw-up 1919. debt 1920. guidance 1921. presentation 1922. cable 1923. debt 1924. defender 1925. terrorism 1926. enhance 1927. kayak 1928. shelter 1929. gyro 1930. hometown 1931. eatable 1932. ukulele 1933. closing 1934. walker 1935. elderberry 1936. marble 1937. runner 1938. animal 1939. cartload 1940. waste 1941. injunction 1942. therapist 1943. zoot-suit 1944. strait 1945. fine 1946. beginner 1947. wrinkle 1948. butter 1949. disco 1950. mocha 1951. neglect 1952. animal 1953. vitro 1954. chord 1955. flume 1956. crusader 1957. plunger 1958. fatigues 1959. detention 1960. perennial 1961. angina 1962. career 1963. armadillo 1964. sneaky 1965. nursing 1966. various 1967. spoon 1968. eyrie 1969. sale 1970. aardvark 1971. commotion 1972. codling 1973. chair 1974. hen 1975. standpoint 1976. personality 1977. armadillo 1978. shelter 1979. encourage 1980. belief 1981. examination 1982. compile 1983. project 1984. happiness 1985. empty 1986. labourer 1987. defense 1988. suspect 1989. trap 1990. thaw 1991. connotation 1992. naive 1993. determined 1994. cartload 1995. squeegee 1996. scholar 1997. hyena 1998. beret 1999. pharmacist 2000. slang 2001. tornado 2002. cheat 2003. pneumonia 2004. sibling 2005. sniffle 2006. prize 2007. scholar 2008. vertigo 2009. selfish 2010. ripe 2011. snarl 2012. intensify 2013. plant 2014. plant 2015. sewer 2016. vibrissae 2017. peep 2018. asset 2019. harm 2020. isolation 2021. neighbour 2022. exhibition 2023. mayonnaise 2024. try 2025. twilight 2026. agony 2027. user 2028. pail 2029. comic 2030. delight 2031. briefing 2032. invincible 2033. pint 2034. supervisor 2035. bankbook 2036. squatter 2037. gauntlet 2038. cotton 2039. pup 2040. fanny-pack 2041. picturesque 2042. metric 2043. ruffle 2044. annoying 2045. event 2046. desktop 2047. tapioca 2048. ashamed 2049. gunpowder 2050. sardine 2051. note 2052. scientific 2053. twilight 2054. asterisk 2055. elver 2056. scene 2057. subcomponent 2058. crystallography 2059. jockey 2060. countess 2061. repulsive 2062. perennial 2063. miter 2064. oafish 2065. underneath 2066. sardine 2067. admin 2068. sparkling 2069. hydrocarbon 2070. snore 2071. beret 2072. elver 2073. eyeball 2074. mole 2075. kebab 2076. wardrobe 2077. undress 2078. source 2079. cockroach 2080. equity 2081. deduce 2082. pennant 2083. cultured 2084. neck 2085. midline 2086. precedent 2087. policeman 2088. figure 2089. nightingale 2090. slot 2091. soundness 2092. stumbling 2093. rooster 2094. offense 2095. apple 2096. hardhat 2097. crucifixion 2098. hotel 2099. important 2100. zoo 2101. taro 2102. odometer 2103. chow 2104. precedent 2105. backyard 2106. geometry 2107. diversity 2108. dishwasher 2109. innate 2110. mentor 2111. jockey 2112. eggplant 2113. quail 2114. captain 2115. billing 2116. dissemination 2117. pause 2118. tailbud 2119. track 2120. goddess 2121. slang 2122. league 2123. armadillo 2124. close 2125. lycra 2126. hurdle 2127. perfection 2128. observation 2129. abusive 2130. shallows 2131. affiliate 2132. gaiters 2133. dissemination 2134. fortune 2135. gamebird 2136. cockroach 2137. careful 2138. sneaky 2139. tooth 2140. pound 2141. traveler 2142. video 2143. washbasin 2144. wedge 2145. hen 2146. jumper 2147. trial 2148. teaching 2149. gaudy 2150. cry 2151. typewriter 2152. nougat 2153. station 2154. abusive 2155. intestine 2156. anatomy 2157. lyrics 2158. bankbook 2159. alert 2160. socialism 2161. hamster 2162. octagon 2163. protective 2164. neon 2165. cockroach 2166. evaporation 2167. opera 2168. tenuous 2169. barium 2170. strait 2171. eliminate 2172. effacement 2173. laughable 2174. disability 2175. wheat 2176. sweltering 2177. breakthrough 2178. llama 2179. humorous 2180. pennant 2181. slay 2182. chauffeur 2183. collaboration 2184. remnant 2185. stumbling 2186. cactus 2187. rainy 2188. multiply 2189. estimate 2190. dictaphone 2191. tasty 2192. careful 2193. walker 2194. window 2195. suppose 2196. mill 2197. aardvark 2198. inform 2199. comedy 2200. significance 2201. profit 2202. leading 2203. status 2204. pajamas 2205. nectarine 2206. boudoir 2207. swimsuit 2208. juggernaut 2209. orientation 2210. adverb 2211. user 2212. modem 2213. ark 2214. nougat 2215. abusive 2216. swimsuit 2217. wind 2218. owl 2219. plowman 2220. drainage 2221. ripe 2222. weight 2223. senior 2224. raise 2225. hunchback 2226. dictaphone 2227. presentation 2228. marxism 2229. jaded 2230. plant 2231. accordance 2232. acquisition 2233. portfolio 2234. subgroup 2235. walker 2236. lumber 2237. plant 2238. vascular 2239. yawn 2240. community 2241. pantry 2242. obesity 2243. performance 2244. arrogance 2245. add 2246. armadillo 2247. removal 2248. granola 2249. rooster 2250. lapdog 2251. vitamin 2252. community 2253. batting 2254. wardrobe 2255. baby 2256. tremor 2257. nutty 2258. geek 2259. operating 2260. countryside 2261. rope 2262. embellishment 2263. retrospective 2264. tow-truck 2265. lose 2266. hiking 2267. mesenchyme 2268. tide 2269. flanker 2270. fine 2271. lady 2272. globe 2273. transplantation 2274. chem 2275. region 2276. note 2277. magnificent 2278. campaign 2279. glass 2280. tablet 2281. drainage 2282. slot 2283. tempo 2284. omission 2285. inch 2286. sledge 2287. sour 2288. petitioner 2289. sardine 2290. changeable 2291. hunchback 2292. story-telling 2293. freelance 2294. middleman 2295. pharmacist 2296. rim 2297. sparkling 2298. flood 2299. sardine 2300. pump 2301. mentor 2302. leptocephalus 2303. successful 2304. whispering 2305. detective 2306. prairie 2307. bet 2308. discovery 2309. pegboard 2310. nail 2311. cautious 2312. freelance 2313. retrospective 2314. prize 2315. nymph 2316. catsup 2317. pard 2318. pharmacist 2319. abusive 2320. publicize 2321. abusive 2322. likelihood 2323. discharge 2324. suspenders 2325. shaw 2326. tummy 2327. yawn 2328. snob 2329. bough 2330. happiness 2331. half-brother 2332. virus 2333. estrogen 2334. campus 2335. rutabaga 2336. subsequent 2337. sorghum 2338. sleep 2339. octave 2340. bird 2341. backyard 2342. molasses 2343. precision 2344. spiffy 2345. jumbo 2346. waffle 2347. insert 2348. fisherman 2349. campus 2350. diversity 2351. cultivator 2352. vascular 2353. passion 2354. sneaky 2355. oncology 2356. allergist 2357. walker 2358. methodology 2359. hornet 2360. crayfish 2361. pea 2362. rebel 2363. participant 2364. mayonnaise 2365. e-book 2366. highway 2367. pause 2368. thrust 2369. campus 2370. hydrocarbon 2371. supreme 2372. cofactor 2373. protective 2374. hydrofoil 2375. chap 2376. implication 2377. terror 2378. millimeter 2379. push 2380. lyrics 2381. particle 2382. ghost 2383. walker 2384. beer 2385. glib 2386. sandpaper 2387. section 2388. decision-making 2389. credible 2390. endurable 2391. magnitude 2392. delight 2393. batting 2394. sardine 2395. trap 2396. homeland 2397. bough 2398. chord 2399. restructure 2400. seal 2401. pail 2402. pressroom 2403. better 2404. chorus 2405. throat 2406. laughable 2407. probe 2408. silly 2409. scale 2410. contention 2411. important 2412. sea 2413. remain 2414. corduroy 2415. trapezoid 2416. wail 2417. curiosity 2418. income 2419. hub 2420. sake 2421. ammunition 2422. cracker 2423. deposit 2424. source 2425. industrious 2426. solicitation 2427. favorite 2428. journalism 2429. badge 2430. person 2431. airfare 2432. dogwood 2433. dishwasher 2434. subject 2435. kamikaze 2436. recollection 2437. technician 2438. gaze 2439. terracotta 2440. acoustics 2441. mirror 2442. drill 2443. nun 2444. slay 2445. familiar 2446. expertise 2447. injunction 2448. store 2449. seal 2450. refrigerator 2451. muffin 2452. botany 2453. restructure 2454. exaggeration 2455. jet 2456. remark 2457. local 2458. agreement 2459. slay 2460. billing 2461. retailer 2462. emphasis 2463. helmet 2464. strange 2465. armadillo 2466. inspire 2467. transfer 2468. used 2469. unique 2470. instrument 2471. wend 2472. parallelogram 2473. cow 2474. custom 2475. bandolier 2476. amazement 2477. tolerance 2478. embellishment 2479. therapist 2480. perp 2481. gate 2482. conspirator 2483. teen 2484. verify 2485. piracy 2486. littleneck 2487. supply 2488. investment 2489. watermelon 2490. maracas 2491. edition 2492. presume 2493. astronomy 2494. gravy 2495. rely 2496. ripe 2497. psychiatrist 2498. disco 2499. quest 2500. friend 2501. councilor 2502. particle 2503. witch-hunt 2504. guitar 2505. briefing 2506. counsellor 2507. normalization 2508. sunbeam 2509. gamy 2510. sideboard 2511. row 2512. motive 2513. motive 2514. solid 2515. adapt 2516. tortoise 2517. distinct 2518. pint 2519. selling 2520. video 2521. waffle 2522. sunlight 2523. captain 2524. petitioner 2525. terrace 2526. meek 2527. prizefight 2528. heap 2529. nice 2530. ore 2531. agony 2532. historical 2533. defender 2534. terrorist 2535. boudoir 2536. quest 2537. dignity 2538. jumper 2539. pint 2540. armadillo 2541. glass 2542. rocket 2543. reign 2544. bloom 2545. tough-guy 2546. publicize 2547. row 2548. ischemia 2549. search 2550. drum 2551. magnificent 2552. navigation 2553. sardine 2554. shine 2555. track 2556. knuckle 2557. gerbil 2558. half 2559. hurried 2560. archaeology 2561. breakthrough 2562. panoramic 2563. curiosity 2564. toot 2565. bill 2566. matter 2567. couple 2568. lynx 2569. weight 2570. rope 2571. flood 2572. ghost 2573. abundant 2574. mole 2575. advance 2576. plaintiff 2577. envy 2578. collapse 2579. layer 2580. pump 2581. calculator 2582. replica 2583. snarl 2584. might 2585. tutu 2586. remark 2587. differential 2588. novel 2589. scatter 2590. agreement 2591. pajamas 2592. adorable 2593. hazel 2594. beginner 2595. conclude 2596. cheat 2597. evaporation 2598. almanac 2599. noxious 2600. sour 2601. puddle 2602. singular 2603. walker 2604. corduroy 2605. trap 2606. fiddle 2607. thyme 2608. footstool 2609. jog 2610. lumber 2611. polyester 2612. eatable 2613. joyous 2614. cigarette 2615. taro 2616. nut 2617. pint 2618. outlook 2619. walker 2620. hood 2621. cashier 2622. seminar 2623. vinyl 2624. reamer 2625. chord 2626. piracy 2627. law 2628. spending 2629. shocking 2630. tension 2631. licensing 2632. reindeer 2633. lychee 2634. juggle 2635. bathroom 2636. student 2637. pipe 2638. grapefruit 2639. supreme 2640. armadillo 2641. excellent 2642. footstep 2643. decision-making 2644. sink 2645. waterspout 2646. spelling 2647. app 2648. fancy 2649. adapter 2650. spend 2651. leprosy 2652. trophy 2653. devastation 2654. dune buggy 2655. possess 2656. unique 2657. salami 2658. afraid 2659. spending 2660. toothsome 2661. pinecone 2662. spend 2663. rally 2664. webmail 2665. pint 2666. perpendicular 2667. softening 2668. perspective 2669. walker 2670. search 2671. prophet 2672. halloween 2673. classification 2674. mutation 2675. strange 2676. autoimmunity 2677. status 2678. postage 2679. mill 2680. floor 2681. enhance 2682. geometry 2683. pilaf 2684. walker 2685. mere 2686. fertile 2687. silo 2688. attorney 2689. landform 2690. sail 2691. avoid 2692. walker 2693. industrious 2694. transplantation 2695. winery 2696. sectional 2697. dune buggy 2698. octave 2699. timeline 2700. highlight 2701. fiction 2702. campus 2703. toothpaste 2704. wide-eyed 2705. quaint 2706. slang 2707. underpass 2708. biosphere 2709. reversal 2710. ark 2711. filly 2712. holder 2713. homicide 2714. moonlight 2715. bandolier 2716. botany 2717. tramp 2718. littleneck 2719. tow-truck 2720. sundial 2721. floor 2722. environment 2723. buzzard 2724. nymph 2725. seashore 2726. chair 2727. refrigerator 2728. bath 2729. reminder 2730. possession 2731. sensitive 2732. scribble 2733. pantry 2734. vertigo 2735. noxious 2736. notion 2737. bank 2738. peep 2739. walker 2740. spark 2741. moment 2742. memory 2743. manage 2744. pegboard 2745. tattoo 2746. abuse 2747. commitment 2748. armadillo 2749. tolerance 2750. tight 2751. expedition 2752. sulfur 2753. might 2754. agony 2755. disdain 2756. browsing 2757. trance 2758. clock 2759. vinyl 2760. beneficiary 2761. opera 2762. webmail 2763. gossip 2764. felony 2765. plover 2766. ruffle 2767. zealous 2768. peanut 2769. trainer 2770. colonisation 2771. naive 2772. bruise 2773. assurance 2774. might 2775. remnant 2776. native 2777. governance 2778. thought 2779. law 2780. agreement 2781. equity 2782. supervisor 2783. hornet 2784. smolt 2785. rotation 2786. selfish 2787. sardine 2788. arrange 2789. cliff 2790. zany 2791. clavicle 2792. gloom 2793. brilliant 2794. diffuse 2795. perp 2796. prevalence 2797. photo 2798. memory 2799. sardine 2800. metro 2801. political 2802. plunger 2803. codling 2804. summarize 2805. rotation 2806. campus 2807. particle 2808. excursion 2809. puddle 2810. licensing 2811. side 2812. discreet 2813. artist 2814. mapping 2815. scientific 2816. chicory 2817. name 2818. mere 2819. disposer 2820. primary 2821. congressperson 2822. armadillo 2823. thought 2824. hypnotic 2825. tech 2826. board 2827. divert 2828. humorous 2829. galley 2830. row 2831. heater 2832. detection 2833. spiffy 2834. shutdown 2835. eponym 2836. outlook 2837. dance 2838. tummy 2839. shutdown 2840. agonizing 2841. supreme 2842. praised 2843. foot 2844. pagoda 2845. see 2846. promise 2847. footstep 2848. unarmed 2849. marxism 2850. standpoint 2851. eyebrow 2852. catastrophe 2853. event 2854. gas 2855. owl 2856. campus 2857. campus 2858. sandal 2859. hutch 2860. nucleotidase 2861. floodplain 2862. reminder 2863. dramatic 2864. leave 2865. barrage 2866. function 2867. division 2868. grapefruit 2869. delight 2870. noodles 2871. external 2872. survey 2873. pressroom 2874. wail 2875. jet 2876. installation 2877. employment 2878. base 2879. republican 2880. consumption 2881. neon 2882. quartz 2883. armadillo 2884. proceedings 2885. brainy 2886. parallelogram 2887. autoimmunity 2888. crystallography 2889. paperback 2890. helicopter 2891. juvenile 2892. intention 2893. crayfish 2894. sideboard 2895. rebellious 2896. minute 2897. leg 2898. maintain 2899. labourer 2900. hurdle 2901. liquid 2902. loutish 2903. lumber 2904. amount 2905. rostrum 2906. embossing 2907. plant 2908. walker 2909. pen 2910. cofactor 2911. cup 2912. bricklaying 2913. suppression 2914. campaign 2915. triad 2916. filly 2917. delight 2918. desktop 2919. coordinate 2920. chicken 2921. maelstrom 2922. throat 2923. probe 2924. plant 2925. assertion 2926. equal 2927. disarmament 2928. sunroom 2929. oats 2930. guidance 2931. reindeer 2932. shower 2933. defender 2934. apple 2935. whispering 2936. terminology 2937. successful 2938. romantic 2939. endure 2940. wrinkle 2941. guarded 2942. ascent 2943. yam 2944. analogue 2945. metric 2946. significance 2947. tornado 2948. wet-bar 2949. cross-stitch 2950. plate 2951. tempo 2952. detection 2953. vitamin 2954. steward 2955. dramaturge 2956. trachoma 2957. adorable 2958. itchy 2959. barium 2960. sink 2961. bowtie 2962. catalog 2963. contention 2964. triad 2965. matter 2966. freezing 2967. republican 2968. reversal 2969. warning 2970. currant 2971. avoid 2972. detective 2973. clavicle 2974. abusive 2975. removal 2976. discretion 2977. supreme 2978. plowman 2979. geek 2980. biosphere 2981. titanium 2982. navigation 2983. blazer 2984. divalent 2985. sunbeam 2986. session 2987. teen 2988. paint 2989. typewriter 2990. courage 2991. bulb 2992. pumpernickel 2993. rotation 2994. cone 2995. laughable 2996. gaudy 2997. photographer 2998. row 2999. c-clamp 3000. webmail 3001. pint 3002. statistic 3003. abrupt 3004. lip 3005. sardine 3006. atom 3007. pressroom 3008. rebellion 3009. fisting 3010. slider 3011. dimension 3012. whimsical 3013. sender 3014. latte 3015. emission 3016. ascent 3017. appetite 3018. advance 3019. closing 3020. statistic 3021. galley 3022. pheasant 3023. keeper 3024. savings 3025. juggle 3026. favorite 3027. songbird 3028. soothsay 3029. mist 3030. boudoir 3031. pride 3032. draconian 3033. vertigo 3034. delight 3035. sour 3036. intention 3037. release 3038. crucifixion 3039. manicure 3040. embellishment 3041. internet 3042. cattle 3043. summarize 3044. mirror 3045. hardhat 3046. armament 3047. draconian 3048. potato 3049. shopping 3050. talent 3051. comfort 3052. endurable 3053. everybody 3054. minibus 3055. mouth 3056. countryside 3057. intestine 3058. cacao 3059. description 3060. hawk 3061. hulking 3062. poison 3063. nut 3064. sardine 3065. sledge 3066. delight 3067. enrollment 3068. perp 3069. peace 3070. bet 3071. bakeware 3072. saw 3073. dictaphone 3074. establishment 3075. ballpark 3076. confusion 3077. finicky 3078. flanker 3079. junker 3080. probe 3081. pumpernickel 3082. savings 3083. defense 3084. row 3085. nectarine 3086. sledge 3087. terrorism 3088. paint 3089. maracas 3090. teen 3091. aware 3092. whispering 3093. goddess 3094. hair 3095. orientation 3096. suspension 3097. metro 3098. emphasis 3099. rutabaga 3100. deposit 3101. cultivar 3102. ectodermal 3103. tangerine 3104. warm-up 3105. sonnet 3106. athletics 3107. graduation 3108. male 3109. yawn 3110. plant 3111. facility 3112. vitro 3113. pegboard 3114. silo 3115. ordinary 3116. llama 3117. shaw 3118. git 3119. inform 3120. supreme 3121. spear 3122. outlaw 3123. whirl 3124. marker 3125. skip 3126. strawberry 3127. exile 3128. slot 3129. cotton 3130. sandpaper 3131. delight 3132. scatter 3133. cymbal 3134. heavenly 3135. holistic 3136. close 3137. gelatin 3138. multiply 3139. oncology 3140. limitation 3141. proponent 3142. student 3143. app 3144. outlaw 3145. tough-guy 3146. pinch 3147. novel 3148. mouth 3149. tattoo 3150. sea 3151. load 3152. embossing 3153. dresser 3154. tank-top 3155. suspect 3156. exile 3157. c-clamp 3158. poetry 3159. excellent 3160. freezing 3161. ectodermal 3162. tailbud 3163. allergist 3164. walker 3165. trustee 3166. salami 3167. plain 3168. timeline 3169. shackle 3170. laundry 3171. electricity 3172. bulb 3173. motivate 3174. gunpowder 3175. cry 3176. anxious 3177. turnover 3178. assurance 3179. discharge 3180. spin 3181. almanac 3182. ape 3183. airline 3184. coalition 3185. scallion 3186. stumbling 3187. cup 3188. alpenhorn 3189. electricity 3190. browsing 3191. standard 3192. plot 3193. abusive 3194. half-brother 3195. sweatsuit 3196. diffuse 3197. anklet 3198. seashore 3199. handicap 3200. terror 3201. countess 3202. slavery 3203. tiger 3204. portfolio 3205. hotel 3206. store 3207. climb 3208. consumption 3209. rebellious 3210. lentil 3211. supreme 3212. subprime 3213. shutdown 3214. nut 3215. bet 3216. tide 3217. minute 3218. jazzy 3219. sidewalk 3220. fiber 3221. plant 3222. deposition 3223. tummy 3224. cacao 3225. snob 3226. expedition 3227. accommodation 3228. news 3229. part 3230. offering 3231. adult 3232. envious 3233. artery 3234. aquarium 3235. local 3236. row 3237. suppression 3238. manage 3239. countess 3240. tapioca 3241. earplug 3242. bloom 3243. jazzy 3244. mutation 3245. owl 3246. neck 3247. cheat 3248. sunlight 3249. softening 3250. sensitive 3251. perpendicular 3252. rebellion 3253. hunt 3254. participant 3255. disillusioned 3256. drum 3257. grouse 3258. changeable 3259. deduce 3260. sloppy 3261. shopping 3262. dial 3263. region 3264. messenger 3265. gorilla 3266. pheasant 3267. crystallography 3268. cleavage 3269. cacao 3270. spoon 3271. geometry 3272. anatomy 3273. armadillo 3274. smile 3275. quote 3276. person 3277. patio 3278. pagoda 3279. encyclopedia 3280. carry 3281. clasp 3282. prophet 3283. sleep 3284. competence 3285. solidarity 3286. airship 3287. mesenchyme 3288. conclude 3289. caddy 3290. tranquil 3291. manicure 3292. decorate 3293. senior 3294. solicitation 3295. annoying 3296. inquisitive 3297. trustee 3298. cupola 3299. subset 3300. empire 3301. quaint 3302. smiling 3303. harmonise 3304. hutch 3305. playground 3306. league 3307. mineshaft 3308. plant 3309. tolerance 3310. psychiatrist 3311. singular 3312. earth 3313. clasp 3314. encyclopedia 3315. brainy 3316. harmonise 3317. rape 3318. pen 3319. zoo 3320. tornado 3321. precision 3322. interview 3323. romantic 3324. pants 3325. moonlight 3326. joy 3327. supreme 3328. swamp 3329. leg 3330. mundane 3331. hulking 3332. comedy 3333. fang 3334. pint 3335. specialist 3336. prune 3337. teammate 3338. ski 3339. granddaughter 3340. homeownership 3341. row 3342. mile 3343. man 3344. hose 3345. omelet 3346. pride 3347. flight 3348. deposition 3349. orientation 3350. eyrie 3351. story-telling 3352. grasshopper 3353. ping 3354. hawk 3355. nutrition 3356. standard 3357. songbird 3358. abusive 3359. distinct 3360. cautious 3361. extract 3362. nightingale 3363. sniffle 3364. pupil 3365. mud 3366. crab 3367. mocha 3368. shackle 3369. bakeware 3370. outlay 3371. stuff 3372. operate 3373. function 3374. storey 3375. upward 3376. alliance 3377. affiliate 3378. crab 3379. differential 3380. deep 3381. jazz 3382. painful 3383. buckle 3384. earplug 3385. lightscreen 3386. sardine 3387. compile 3388. dissemination 3389. tailbud 3390. lieu 3391. cultured 3392. removal 3393. campus 3394. traveler 3395. swimsuit 3396. description 3397. shower 3398. row 3399. research 3400. finger 3401. personnel 3402. price 3403. bathroom 3404. admin 3405. unaccountable 3406. row 3407. competitor 3408. violet 3409. soulmate 3410. spleen 3411. afterthought 3412. adverb 3413. stare 3414. coordinate 3415. pint 3416. abusive 3417. middleman 3418. operate 3419. eliminate 3420. floor 3421. campus 3422. officiate 3423. credible 3424. steady 3425. wobble 3426. softdrink 3427. discovery 3428. fortune 3429. alliance 3430. impartial 3431. profit 3432. difference 3433. dramaturge 3434. dentist 3435. channel 3436. revolution 3437. outlaw 3438. sibling 3439. equity 3440. subsequent 3441. violet 3442. cloudy 3443. plover 3444. cicada 3445. scatter 3446. felony 3447. cultivar 3448. difference 3449. family 3450. airline 3451. client 3452. understood 3453. guidance 3454. seal 3455. mine 3456. minnow 3457. finisher 3458. reminder 3459. information 3460. SUV 3461. republic 3462. homeownership 3463. cuisine 3464. hurdler 3465. revolution 3466. campaign 3467. ignore 3468. simplification 3469. gate 3470. quote 3471. sunroom 3472. unusual 3473. hurdler 3474. typewriter 3475. portrait 3476. row 3477. frontier 3478. amount 3479. guitar 3480. sake 3481. colonisation 3482. male 3483. eyeliner 3484. materialistic 3485. baby 3486. magnet 3487. fresco 3488. bricklaying 3489. plunger 3490. jog 3491. thyme 3492. trench 3493. skyscraper 3494. supreme 3495. turf 3496. cliff 3497. tangy 3498. career 3499. promote 3500. prevalence 3501. custom 3502. windage 3503. chalk 3504. virus 3505. perspective 3506. length 3507. hook 3508. region 3509. skylight 3510. campus 3511. redirect 3512. metric 3513. slaw 3514. zany 3515. steward 3516. illusion 3517. wheat 3518. clock 3519. expertise 3520. dresser 3521. income 3522. effacement 3523. thaw 3524. fiber 3525. sparkling 3526. copying 3527. juggernaut 3528. turf 3529. caddy 3530. nymph 3531. inflation 3532. poetry 3533. grip 3534. recollection 3535. male 3536. zoot-suit 3537. zippy 3538. scribble 3539. function 3540. photo 3541. plain 3542. plaintiff 3543. east 3544. recruit 3545. wend 3546. gaudy 3547. workshop 3548. parallelogram 3549. apprehension 3550. acoustics 3551. mist 3552. diffuse 3553. intention 3554. hamster 3555. supreme 3556. offer 3557. chauffeur 3558. passenger 3559. plant 3560. brow 3561. scenario 3562. saw 3563. matter 3564. mere 3565. delight 3566. crab 3567. dogwood 3568. triad 3569. moment 3570. conspirator 3571. copying 3572. juvenile 3573. nougat 3574. establishment 3575. verse 3576. subject 3577. confusion 3578. fisting 3579. blazer 3580. juggle 3581. poison 3582. slavery 3583. standard 3584. residence 3585. peer-to-peer 3586. preserves 3587. recruit 3588. mundane 3589. materialistic 3590. cinder 3591. walker 3592. archaeology 3593. discretion 3594. pattern 3595. disillusioned 3596. stepdaughter 3597. understood 3598. tattoo 3599. excellent 3600. hulking 3601. thrush 3602. suite 3603. cultivator 3604. polenta 3605. granola 3606. length 3607. homicide 3608. chicken 3609. hunchback 3610. step-daughter 3611. marionberry 3612. termination 3613. conspirator 3614. omission 3615. bird 3616. smoggy 3617. crusader 3618. pint 3619. gorilla 3620. mythology 3621. offer 3622. campus 3623. currant 3624. armadillo 3625. unlock 3626. instrumentalist 3627. supreme 3628. fame 3629. lamb 3630. ounce 3631. pint 3632. flume 3633. deep 3634. sardine 3635. hen 3636. SUV 3637. dilution 3638. alpenglow 3639. crackers 3640. cultivator 3641. aquarium 3642. zany 3643. construction 3644. pause 3645. biosphere 3646. termination 3647. astrolabe 3648. leading 3649. brisket 3650. divalent 3651. impartial 3652. slavery 3653. payee 3654. picturesque 3655. sweltering 3656. enthusiastic 3657. clarify 3658. bourgeoisie 3659. sweatsuit 3660. conversation 3661. supreme 3662. used 3663. internet 3664. trove 3665. blister 3666. row 3667. reliability 3668. policeman 3669. trial 3670. inch 3671. transfer 3672. profession 3673. hyphenation 3674. sombrero 3675. float 3676. chorus 3677. injunction 3678. slaw 3679. sunbeam 3680. delight 3681. pants 3682. trophy 3683. veldt 3684. painting 3685. doubter 3686. cleavage 3687. equal 3688. appetite 3689. artery 3690. boil 3691. mineshaft 3692. turnover 3693. prizefight 3694. analogue 3695. puddle 3696. sale 3697. remember 3698. load 3699. crackers 3700. tomatillo 3701. disability 3702. armadillo 3703. survey 3704. instant 3705. clock 3706. apprehension 3707. e-book 3708. butter 3709. row 3710. performance 3711. investment 3712. conclude 3713. hardhat 3714. lemon 3715. aquarium 3716. solidity 3717. darn 3718. walker 3719. conversation 3720. subcomponent 3721. stump 3722. used 3723. beaver 3724. murder 3725. pint 3726. redirect 3727. expedition 3728. tiger 3729. mapping 3730. peer-to-peer 3731. spoil 3732. nitrogen 3733. important 3734. cotton 3735. patient 3736. patio 3737. unlock 3738. marked 3739. clasp 3740. toothsome 3741. sultan 3742. whimsical 3743. tenuous 3744. helicopter 3745. spokeswoman 3746. parser 3747. native 3748. smoggy 3749. remember 3750. swine 3751. paperwork 3752. stylus 3753. supervisor 3754. opposite 3755. trapezoid 3756. mill 3757. illusion 3758. article 3759. applause 3760. weight 3761. heirloom 3762. pajamas 3763. see 3764. devastation 3765. highlight 3766. hyphenation 3767. wardrobe 3768. presume 3769. inform 3770. conference 3771. ram 3772. track 3773. wall 3774. grip 3775. wobble 3776. participant 3777. bricklaying 3778. soda 3779. luncheonette 3780. subsidiary 3781. cracker 3782. perennial 3783. bathroom 3784. defective 3785. cupola 3786. beneficiary 3787. personality 3788. changeable 3789. repulsive 3790. jumper 3791. walker 3792. wedge 3793. chicory 3794. witch-hunt 3795. thunderstorm 3796. opposite 3797. half-brother 3798. skip 3799. dolor 3800. councilor 3801. coordinate 3802. hotel 3803. pound 3804. restored 3805. skate 3806. plant 3807. remain 3808. airfare 3809. cymbal 3810. apology 3811. bowtie 3812. nun 3813. butter 3814. technician 3815. hit 3816. instrument 3817. gas 3818. tram 3819. bough 3820. gesture 3821. blouse 3822. delight 3823. carnival 3824. modem 3825. defeated 3826. cattle 3827. nutrient 3828. sardine 3829. row 3830. wrinkle 3831. sun 3832. smolt 3833. toothpaste 3834. operating 3835. finger 3836. sermon 3837. unpack 3838. sonnet 3839. republic 3840. gaze 3841. prince 3842. brilliant 3843. llama 3844. footstool 3845. zoot-suit 3846. smiling 3847. underpass 3848. sink 3849. build 3850. softening 3851. sender 3852. sweltering 3853. guava 3854. chain 3855. bath 3856. titanium 3857. moan 3858. detention 3859. comfort 3860. determined 3861. trial 3862. spokeswoman 3863. soundness 3864. mine 3865. judge 3866. unlock 3867. raise 3868. doggie 3869. supreme 3870. twilight 3871. promise 3872. lieu 3873. brilliant 3874. sender 3875. catsup 3876. angina 3877. bird 3878. upward 3879. odometer 3880. goddess 3881. jumbo 3882. fresco 3883. loutish 3884. policy 3885. stealth 3886. flippant 3887. storey 3888. knit 3889. sleep 3890. lip 3891. replica 3892. peanut 3893. convention 3894. gorilla 3895. grief 3896. subgroup 3897. designer 3898. motivate 3899. silly 3900. raisin 3901. colonisation 3902. specialist 3903. abundant 3904. divert 3905. octagon 3906. shallows 3907. abusive 3908. nutrition 3909. half 3910. marble 3911. cultivar 3912. paranoia 3913. graduation 3914. cliff 3915. friend 3916. chalk 3917. kayak 3918. goggles 3919. knuckle 3920. share 3921. abusive 3922. innate 3923. familiar 3924. supreme 3925. scaffold 3926. proceedings 3927. mobster 3928. halloween 3929. sultan 3930. hyena 3931. maintain 3932. middleman 3933. carnival 3934. kebab 3935. offense 3936. preserves 3937. pride 3938. gobbler 3939. instruction 3940. verse 3941. abusive 3942. verdict 3943. suspect 3944. rehabilitate 3945. atrium 3946. fancy 3947. deputy 3948. ziggurat 3949. pint 3950. thrush 3951. cactus 3952. earth 3953. pawnshop 3954. window 3955. chow 3956. rim 3957. badge 3958. armament 3959. ape 3960. knotty 3961. armadillo 3962. tape 3963. mandate 3964. treasury 3965. spending 3966. year 3967. maelstrom 3968. abusive 3969. mariachi 3970. sale 3971. expansion 3972. spring 3973. howitzer 3974. asset 3975. minibus 3976. portfolio 3977. measure 3978. smoggy 3979. warden 3980. contention 3981. gamy 3982. instrument 3983. delight 3984. patient 3985. judge 3986. bounce 3987. chutney 3988. blouse 3989. praised 3990. beach 3991. suppose 3992. plowman 3993. name 3994. keyboarding 3995. tech 3996. molasses 3997. heirloom 3998. beach 3999. underestimate 4000. octagon 4001. lemon 4002. sombrero 4003. maintain 4004. supreme 4005. suspension 4006. smelly 4007. wiring 4008. price 4009. astonishing 4010. tension 4011. gauntlet 4012. reamer 4013. cautious 4014. psychiatrist 4015. discreet 4016. app 4017. abuse 4018. empire 4019. nucleotidase 4020. intervenor 4021. stump 4022. contact lens 4023. conference 4024. amount 4025. wail 4026. granola 4027. scene 4028. abrupt 4029. ounce 4030. eyeliner 4031. instant 4032. defense 4033. pint 4034. depth 4035. homonym 4036. panoramic 4037. webpage 4038. sermon 4039. passion 4040. homonym 4041. supreme 4042. release 4043. cling 4044. watch 4045. cleavage 4046. carry 4047. chutney 4048. gamebird 4049. campus 4050. ozone 4051. neglect 4052. holistic 4053. slider 4054. bulb 4055. porcelain 4056. eggplant 4057. bill 4058. sardine 4059. patio 4060. terrorist 4061. element 4062. silkworm 4063. utilization 4064. exaggeration 4065. tongue 4066. faulty 4067. scallion 4068. enlist 4069. ore 4070. subset 4071. mayor 4072. armadillo 4073. row 4074. peer-to-peer 4075. shocking 4076. pagoda 4077. officiate 4078. tide 4079. delight 4080. gig 4081. apology 4082. jog 4083. summarize 4084. pink 4085. lynx 4086. career 4087. fascinated 4088. advance 4089. paramecium 4090. eligibility 4091. disarmament 4092. luxuriant 4093. stump 4094. veldt 4095. axis 4096. chard 4097. vinyl 4098. boil 4099. painful 4100. preserves 4101. limitation 4102. expansion 4103. beach 4104. armadillo 4105. unique 4106. chip 4107. chauffeur 4108. humorous 4109. man 4110. mile 4111. tangerine 4112. personnel 4113. sandal 4114. community 4115. anticipate 4116. price 4117. keyboarding 4118. shower 4119. abusive 4120. holder 4121. grief 4122. carry 4123. yielding 4124. teammate 4125. release 4126. sun 4127. hosiery 4128. softdrink 4129. ukulele 4130. globe 4131. going 4132. eponym 4133. attorney 4134. barstool 4135. edition 4136. holistic 4137. dispense 4138. processing 4139. astonishing 4140. journalism 4141. gaiters 4142. edition 4143. swine 4144. vascular 4145. billing 4146. lychee 4147. terracotta 4148. dolor 4149. atom 4150. swamp 4151. instrumentalist 4152. tortoise 4153. payee 4154. sake 4155. nucleotidase 4156. convention 4157. bankbook 4158. allergist 4159. buzzard 4160. residence 4161. hurried 4162. campus 4163. reset 4164. zoo 4165. enrollment 4166. hunt 4167. disposer 4168. tomatillo 4169. isolation 4170. gift 4171. light 4172. plate 4173. trainer 4174. row 4175. dimension 4176. bout 4177. flippant 4178. estimate 4179. scale 4180. fancy 4181. cluster 4182. violation 4183. asterisk 4184. tool 4185. recess 4186. camper 4187. various 4188. profession 4189. prey 4190. prow 4191. guava 4192. glossy 4193. hoof 4194. silo 4195. doorpost 4196. mist 4197. toll 4198. cow 4199. porpoise 4200. vitro 4201. intelligence 4202. respond 4203. skunk 4204. axis 4205. coalition 4206. oafish 4207. instruction 4208. manage 4209. lapdog 4210. sloppy 4211. rinse 4212. blow 4213. foot 4214. campus 4215. undress 4216. event 4217. meek 4218. consider 4219. delight 4220. verse 4221. alert 4222. plant 4223. nursing 4224. noxious 4225. enrollment 4226. disillusioned 4227. unaccountable 4228. close 4229. virus 4230. murder 4231. subcomponent 4232. jumbo 4233. restructure 4234. stir 4235. prairie 4236. porpoise 4237. enquiry 4238. stuff 4239. divert 4240. terrible 4241. processing 4242. medicine 4243. restriction 4244. packaging 4245. belief 4246. criterion 4247. pea 4248. impartial 4249. blister 4250. launch 4251. sardine 4252. eatable 4253. sundial 4254. armadillo 4255. exercise 4256. conversation 4257. whimsical 4258. hope 4259. unfasten 4260. hurdle 4261. chard 4262. armament 4263. ski 4264. abacus 4265. gunpowder 4266. abusive 4267. quote 4268. sundial 4269. liquid 4270. sulfur 4271. intensify 4272. protect 4273. pump 4274. wide-eyed 4275. armadillo 4276. dilution 4277. rally 4278. nutrition 4279. acoustics 4280. interview 4281. dune buggy 4282. whirl 4283. spleen 4284. zealous 4285. baby 4286. tomorrow 4287. paperback 4288. leek 4289. handicap 4290. councilor 4291. therapist 4292. councilperson 4293. mattress 4294. notion 4295. rose 4296. build 4297. hazel 4298. perpendicular 4299. likelihood 4300. stealth 4301. morale 4302. portrait 4303. pupil 4304. elver 4305. wedge 4306. eligibility 4307. processing 4308. stopwatch 4309. plant 4310. selfish 4311. batting 4312. alb 4313. muffin 4314. ping 4315. disdain 4316. juggernaut 4317. underpass 4318. granddaughter 4319. person 4320. applause 4321. acquisition 4322. unpack 4323. mariachi 4324. replica 4325. deck 4326. subprime 4327. fisting 4328. envious 4329. falling-out 4330. acquisition 4331. windage 4332. photographer 4333. breezy 4334. romantic 4335. stir 4336. lipstick 4337. ape 4338. step-sister 4339. measure 4340. granddaughter 4341. modify 4342. ignore 4343. scaffold 4344. restored 4345. hoof 4346. nitrogen 4347. lentil 4348. notion 4349. toe 4350. hamster 4351. endure 4352. offer 4353. ballpark 4354. contractor 4355. launch 4356. marionberry 4357. artist 4358. wind 4359. pint 4360. lamb 4361. landform 4362. wash 4363. historical 4364. subprime 4365. campus 4366. prow 4367. specialty 4368. fiddle 4369. squeegee 4370. armadillo 4371. row 4372. wealthy 4373. passenger 4374. neon 4375. defeated 4376. dolor 4377. pinkie 4378. afterthought 4379. armadillo 4380. mole\nQuestion: What are the 10 most common words in the above list? Answer: The top 10 words that appear most often in the list are:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    context_length = input_ids.shape[-1]
    output_ids = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        pad_token_id=tokenizer.eos_token_id, 
        use_cache=True, 
    )[0]
    outputs = tokenizer.decode(output_ids[context_length:], skip_special_tokens=True)
    print(outputs)
