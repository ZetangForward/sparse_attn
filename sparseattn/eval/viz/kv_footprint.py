import torch
import functools
from typing import Tuple
import numpy as np
import math


def causal_mask_active(total_len: int):
    return total_len * (total_len + 1) / 2


def streaming_mask_stats(
    input_len: int,
    output_len: int,
    prefill_chunk_size: int,
    sink_token_size: int,
    local_window_size: int,
):
    active_entries = 0
    prefill_boundaries = list(np.arange(0, input_len, prefill_chunk_size)) + [input_len]

    peak_points = []

    for a, b in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
        local_tokens = a - max(0, a - local_window_size)
        sink_tokens = a - local_tokens - max(0, a - local_tokens - sink_token_size)
        active_entries += causal_mask_active(b - a) + (b - a) * (
            local_tokens + sink_tokens
        )
        peak_points.append(b - a + local_tokens + sink_tokens)
    for a in range(input_len, input_len + output_len):
        local_tokens = a - max(0, a - local_window_size)
        sink_tokens = a - local_tokens - max(0, a - local_tokens - sink_token_size)
        active_entries += 1 + local_tokens + sink_tokens
    peak_points.append(1 + local_tokens + sink_tokens)

    return active_entries, np.array(peak_points)


def global_mask_stats(
    input_len: int,
    output_len: int,
    prefill_chunk_size: int,
    kv_sparsity: float,
    kv_window: int = 64,
):
    active_entries = 0
    prefill_boundaries = list(np.arange(0, input_len, prefill_chunk_size)) + [input_len]

    peak_points = []

    past_kvs = 0
    for a, b in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
        active_entries += causal_mask_active(b - a) + (past_kvs * (b - a))
        peak_points.append(past_kvs + (b - a))
        past_kvs += min(round((1 - kv_sparsity) * (b - a)) + kv_window, b - a)

    active_entries += causal_mask_active(output_len) + (past_kvs * (output_len))
    peak_points.append(past_kvs + (output_len))

    return active_entries, np.array(peak_points)


def streaming_mask_stats_locret(
    input_len: int,
    output_len: int,
    prefill_chunk_size: int,
    locret_sparsity: float,
    locret_local_len: int,
    locret_stabilizers: int,
):
    active_entries = 0

    # Usual prefill boundaries up to input_len - locret_local_len; the last locret_local_len is its own prefill chunk
    if locret_local_len >= input_len:
        prefill_boundaries = [0, input_len]
    else:
        prefill_boundaries = list(
            np.arange(0, input_len - locret_local_len, prefill_chunk_size)
        ) + [input_len - locret_local_len, input_len]

    peak_points = []

    for a, b in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
        # stabilizers are what local_window_size is for pyramid/snapkv
        local_tokens = a - max(0, a - locret_stabilizers)
        active_entries += causal_mask_active(b - a) + (b - a) * local_tokens
        peak_points.append(b - a + local_tokens)
    for a in range(input_len, input_len + output_len):
        local_tokens = a - max(0, a - locret_stabilizers)
        active_entries += 1 + local_tokens
    peak_points.append(1 + local_tokens)

    return active_entries, np.array(peak_points)


def global_mask_stats_locret(
    input_len: int,
    output_len: int,
    prefill_chunk_size: int,
    locret_sparsity: float,
    locret_local_len: int,
    locret_stabilizers: int,
):
    active_entries = 0
    if locret_local_len >= input_len:
        prefill_boundaries = [0, input_len]
    else:
        prefill_boundaries = list(
            np.arange(0, input_len - locret_local_len, prefill_chunk_size)
        ) + [input_len - locret_local_len, input_len]

    peak_points = []

    past_kvs = 0
    for a, b in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
        active_entries += causal_mask_active(b - a) + (past_kvs * (b - a))
        peak_points.append(past_kvs + (b - a))
        past_kvs += min(
            round((1 - locret_sparsity) * (b - a)) + locret_stabilizers, b - a
        )

    active_entries += causal_mask_active(output_len) + (past_kvs * (output_len))
    peak_points.append(past_kvs + (output_len))

    return active_entries, np.array(peak_points)


@functools.lru_cache(maxsize=1000)
def get_kv_footprint(
    prompt_len: int,
    response_len: int,
    prefill_chunk_size: int,
    sink_tokens: int,
    local_window_size: int,
    kv_sparsity: float,
    head_sparsity: float,
) -> Tuple[float, float]:
    streaming_active_entries, streaming_peak_points = streaming_mask_stats(
        prompt_len, response_len, prefill_chunk_size, sink_tokens, local_window_size
    )
    global_active_entries, global_peak_points = global_mask_stats(
        prompt_len, response_len, prefill_chunk_size, kv_sparsity
    )
    full_active_entries, full_peak_points = (
        causal_mask_active(prompt_len + response_len),
        np.array([prompt_len + response_len]),
    )
    print("streaming_active_entries:", streaming_active_entries)

    streaming_footprint = streaming_active_entries / full_active_entries
    global_footprint = global_active_entries / full_active_entries
    print("streaming_footprint:", streaming_footprint)
    print("global_footprint:", global_footprint)

    kv_footprint = streaming_footprint * head_sparsity + global_footprint * (
        1 - head_sparsity
    )
    kv_peak = (
        head_sparsity * streaming_peak_points + (1 - head_sparsity) * global_peak_points
    ).max() / full_peak_points.max()

    return kv_footprint, kv_peak


@functools.lru_cache(maxsize=1000)
def get_kv_footprint_locret(
    prompt_len: int,
    response_len: int,
    prefill_chunk_size: int,
    locret_sparsity: float,
    locret_local_len: int,
    locret_stabilizers: int,
) -> Tuple[float, float]:
    global_active_entries, global_peak_points = global_mask_stats_locret(
        prompt_len,
        response_len,
        prefill_chunk_size,
        locret_sparsity,
        locret_local_len,
        locret_stabilizers,
    )
    full_active_entries, full_peak_points = (
        causal_mask_active(prompt_len + response_len),
        np.array([prompt_len + response_len]),
    )

    global_footprint = global_active_entries / full_active_entries

    kv_footprint = global_footprint
    kv_peak = global_peak_points.max() / full_peak_points.max()

    return kv_footprint, kv_peak


def calculate_kv_statistics(
    prompt_lens: int,
    response_lens: int,
    prefill_chunk_size: int,
    head_sparsity: float,
    sink_tokens: int,
    local_window_size: int,
    kv_sparsity: float,
) -> Tuple[float, float]:
    kv_footprint = []
    kv_peak = []

    for prompt_len, response_len in zip(prompt_lens, response_lens):
        a, b = get_kv_footprint(
            prompt_len,
            response_len,
            prefill_chunk_size,
            sink_tokens,
            local_window_size,
            kv_sparsity,
            head_sparsity,
        )
        kv_footprint.append(a)
        kv_peak.append(b)

    return np.mean(kv_footprint), np.mean(kv_peak)


def calculate_kv_statistics_locret(
    prompt_lens: int,
    response_lens: int,
    prefill_chunk_size: int,
    locret_sparsity: float,
    locret_local_len: int,
    locret_stabilizers: int,
) -> Tuple[float, float]:
    kv_footprint = []
    kv_peak = []

    for prompt_len, response_len in zip(prompt_lens, response_lens):
        a, b = get_kv_footprint_locret(
            prompt_len,
            response_len,
            prefill_chunk_size,
            locret_sparsity,
            locret_local_len,
            locret_stabilizers,
        )
        kv_footprint.append(a)
        kv_peak.append(b)

    return np.mean(kv_footprint), np.mean(kv_peak)

# XAttention block-level KV footprint estimation

# @functools.lru_cache(maxsize=1000)
# def get_kv_footprint_xattn(
#     prompt_len: int,
#     response_len: int,
#     chunk_size: int = 16384, 
#     block_size: int = 64, 
#     threshold: float = 0.9,
#     head_sparsity: float = 0.5,
#     kv_sparsity: float = 0.2,
# ) -> Tuple[float, float]:

#     empirical_points = {
#         4_000: 0.5538,
#         8_000: 0.4355,
#         16_000: 0.2891,
#         32_000: 0.2793,
#         64_000: 0.1132,
#         128_000: 0.0732,
#     }
#     lengths = np.array(list(empirical_points.keys()), dtype=np.float64)
#     sparsities = np.array(list(empirical_points.values()), dtype=np.float64)
#     log_L = np.log(lengths)
#     log_S = np.log(sparsities)
#     slope, intercept = np.polyfit(log_L, log_S, deg=1)  # log_S ≈ slope * log_L + intercept
#     a = float(np.exp(intercept))
#     b = float(slope)

#     total_tokens = int(prompt_len + response_len)
#     retained_ratio = a * (total_tokens ** b)
#     retained_ratio = float(np.clip(retained_ratio, 0.01, 0.9))
    
#     retained_tokens = total_tokens * retained_ratio
#     num_chunks = np.ceil(total_tokens / chunk_size)
#     active_entries = retained_tokens / num_chunks

#     # full baseline (causal dense for whole sequence)
#     full_active_entries = causal_mask_active(total_tokens)
#     if full_active_entries <= 0:
#         return 0.0, 0.0
#     full_peak_points = np.array([prompt_len + response_len])

#     global_active_entries, global_peak_points = global_mask_stats(
#         prompt_len, response_len, chunk_size, kv_sparsity
#     )

#     global_footprint = global_active_entries / full_active_entries

#     kv_footprint = (
#         active_entries * head_sparsity + global_footprint * ( 1 - head_sparsity)
#         ) / full_peak_points.max()
#     kv_peak = retained_tokens / total_tokens
#     return kv_footprint, kv_peak


@functools.lru_cache(maxsize=1000)
def get_kv_footprint_xattn(
    prompt_len: int,
    response_len: int,
    chunk_size: int = 16384, 
    block_size: int = 64, 
    threshold: float = 0.9,
    head_sparsity: float = 0.5,
    kv_sparsity: float = 0.2,
) -> Tuple[float, float]:
    # empirical_points = {
    #     4_000: 0.5538,
    #     8_000: 0.4355,
    #     16_000: 0.2891,
    #     32_000: 0.2793,
    #     64_000: 0.1132,
    #     128_000: 0.0732,
    # }
    empirical_points = {
        4_000: 0.5538,
        8_000: 0.4355,
        16_000: 0.2891,
        32_000: 0.2793,
        64_000: 0.1132,
        128_000: 0.0732,
    }
    lengths = np.array(list(empirical_points.keys()), dtype=np.float64)
    sparsities = np.array(list(empirical_points.values()), dtype=np.float64)
    log_L = np.log(lengths)
    log_S = np.log(sparsities)
    slope, intercept = np.polyfit(log_L, log_S, deg=1)
    a = float(np.exp(intercept))
    b = float(slope)

    total_tokens = int(prompt_len + response_len)
    retained_ratio = a * (prompt_len ** b)
    retained_ratio = float(np.clip(retained_ratio, 0.01, 0.9))

    prefill_boundaries = list(np.arange(0, prompt_len, chunk_size)) + [prompt_len]

    peak_points = []

    active_entries = causal_mask_active(prompt_len * retained_ratio)
    for a, b in zip(prefill_boundaries[:-1], prefill_boundaries[1:]):
        active_entries += causal_mask_active(b - a)

    for a in range(prompt_len, prompt_len + response_len):
        active_entries += 1 + prompt_len
    peak_points.append(1 + response_len)
    print("XAttention active_entries:", active_entries)
    
    full_active_entries = causal_mask_active(total_tokens)
    if full_active_entries <= 0:
        return 0.0, 0.0

    kv_footprint = active_entries / full_active_entries
    kv_peak = (active_entries + response_len) / total_tokens
    print("XAttention kv_footprint:", kv_footprint)
    
    full_active_entries, full_peak_points = (
        causal_mask_active(prompt_len + response_len),
        np.array([prompt_len + response_len]),
    )
    
    global_active_entries, global_peak_points = global_mask_stats(
        prompt_len, response_len, chunk_size, kv_sparsity
    )

    global_footprint = global_active_entries / full_active_entries

    kv_footprint = min(1.0, kv_footprint * head_sparsity + global_footprint * (
        1 - head_sparsity
    ))
    kv_peak = (
        head_sparsity * kv_footprint + (1 - head_sparsity) * global_peak_points
    ).max() / full_peak_points.max()


    return kv_footprint, kv_peak


def calculate_kv_statistics_xattn(
    prompt_lens,
    response_lens,
    chunk_size: int = 16384,
    block_size: int = 64,
    threshold: float = 0.9,
) -> Tuple[float, float]:
    kv_footprint = []
    kv_peak = []
    for prompt_len, response_len in zip(prompt_lens, response_lens):
        a, b = get_kv_footprint_xattn(
            prompt_len, response_len, chunk_size, block_size, threshold
        )
        kv_footprint.append(a)
        kv_peak.append(b)
    return np.mean(kv_footprint), np.mean(kv_peak)