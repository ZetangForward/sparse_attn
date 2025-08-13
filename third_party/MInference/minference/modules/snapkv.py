# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Refer to the code in https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py,
# https://github.com/Zefan-Cai/PyramidKV/blob/main/pyramidkv/pyramidkv_utils.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quest import repeat_kv

class SnapKVCluster:
    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        n_rep=None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.n_rep = n_rep

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        n_rep=None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.n_rep = n_rep

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        capacity_override=None,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]

        if self.n_rep is None:
            bsz, num_heads, q_len, head_dim = query_states.shape
        else:
            bsz, num_kv_heads, q_len, head_dim = key_states.shape
            num_heads = num_kv_heads * self.n_rep
        
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]

        # Use override if provided, otherwise use the class default
        current_max_capacity = capacity_override if capacity_override is not None else self.max_capacity_prompt

        if q_len < current_max_capacity:
            return key_states, value_states
        else:
            if self.n_rep is None:
                key_states_for_calculation = key_states
            else:
                key_states_for_calculation = repeat_kv(key_states, self.n_rep)

            attn_weights = torch.matmul(
                query_states[..., -self.window_size :, :], key_states_for_calculation.transpose(2, 3)
            ) / math.sqrt(head_dim)
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[
                :, :, -self.window_size :, -self.window_size :
            ] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights_sum = attn_weights[
                :, :, -self.window_size :, : -self.window_size
            ].sum(dim=-2)
            if self.pooling == "avgpool":
                attn_cache = F.avg_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            elif self.pooling == "maxpool":
                attn_cache = F.max_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            else:
                raise ValueError("Pooling method not supported")
            
            if self.n_rep is not None:
                # The attn is are currently [bsz, num_heads, window_size, q_len]
                # We need to convert them to [bsz, num_kv_heads, window_size, q_len] by mean pooling
                l = attn_cache.shape[-1]
                attn_cache = attn_cache.view(bsz, num_kv_heads, self.n_rep, l)
                attn_cache = attn_cache.mean(dim=2)

            # indices are arranged in descending order of importance
            if current_max_capacity <= self.window_size:
                key_states = k_cur
                value_states = v_cur
            else:
                indices = attn_cache.topk(
                    min(
                        current_max_capacity - self.window_size,
                        attn_cache.shape[-1]
                    ), 
                    dim=-1
                ).indices
                indices = indices.sort(dim=-1)[0] # New: sort in the correct order of tokens
                
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                    dim=2, index=indices
                )
                v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                    dim=2, index=indices
                )
                k_cur = key_states[:, :, -self.window_size :, :]
                v_cur = value_states[:, :, -self.window_size :, :]
                
                key_states = torch.cat([k_past_compress, k_cur], dim=2)
                value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class StreamingLLMKVCluster:
    def __init__(
        self,
        window_size=4096 - 128,
        max_capacity_prompt=4096,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(
        self,
        window_size=4096 - 128,
        max_capacity_prompt=4096,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        capacity_override=None,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # Use override if provided, otherwise use the class default
        current_max_capacity = capacity_override if capacity_override is not None else self.max_capacity_prompt

        if q_len < current_max_capacity:
            return key_states, value_states
        else:
            indices = torch.tensor(
                range(current_max_capacity - self.window_size), dtype=torch.int64
            ).to(key_states.device)
            indices = (
                indices.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bsz, num_heads, 1, head_dim)
            )

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states
