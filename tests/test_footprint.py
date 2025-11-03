import numpy as np
import matplotlib.pyplot as plt

def compare_kv_footprints(
    input_lengths,
    response_length=0,
    streaming_params=None,
    locret_params=None,
    xattn_params=None,
    save_path=None,
):
    """
    对比三种机制 (Streaming / LocRet / XAttention) 的 KV footprint 与 input 长度的关系。

    Args:
        input_lengths (list[int]): 不同输入长度。
        response_length (int): 输出长度，默认为0。
        streaming_params (dict): 传入 Streaming 参数。
        locret_params (dict): 传入 LocRet 参数。
        xattn_params (dict): 传入 XAttention 参数。
        save_path (str): 可选路径，保存图像文件。
    """

    from sparseattn.eval.viz.kv_footprint import (
        get_kv_footprint,
        get_kv_footprint_locret,
        get_kv_footprint_xattn,
    )

    streaming_ratios = []
    locret_ratios = []
    xattn_ratios = []

    for L in input_lengths:
        # Streaming
        if streaming_params is not None:
            f, _ = get_kv_footprint(
                L,
                response_length,
                **streaming_params,
            )
            streaming_ratios.append(f)
        # LocRet
        if locret_params is not None:
            f, _ = get_kv_footprint_locret(
                L,
                response_length,
                **locret_params,
            )
            locret_ratios.append(f)
        # XAttention
        if xattn_params is not None:
            f, _ = get_kv_footprint_xattn(
                L,
                response_length,
                **xattn_params,
            )
            xattn_ratios.append(f)

    plt.figure(figsize=(7, 4))
    if streaming_ratios:
        plt.plot(input_lengths, np.array(streaming_ratios) * 100, label="Streaming", marker="o")
    if locret_ratios:
        plt.plot(input_lengths, np.array(locret_ratios) * 100, label="LocRet", marker="^")
    if xattn_ratios:
        plt.plot(input_lengths, np.array(xattn_ratios) * 100, label="XAttention", marker="s")

    plt.xlabel("Input length (tokens)")
    plt.ylabel("KV footprint (%)")
    plt.title("KV Footprint Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    import os
    os.makedirs(os.path.dirname("/data/lcm_lab/qqt/project/SparseAttn/tests/footprint/"), exist_ok=True)
    plt.savefig("/data/lcm_lab/qqt/project/SparseAttn/tests/footprint/xattn_footprint.pdf", bbox_inches="tight", dpi=300)

compare_kv_footprints(
    input_lengths=[4_096, 8_000, 16_000, 32_000, 64_000, 128_000],
    response_length=2_000,
    streaming_params={
        "prefill_chunk_size": 16384,
        "head_sparsity": 0.5,
        "sink_tokens": 128,
        "local_window_size": 1024,
        "kv_sparsity": 0.2,
    },
    xattn_params={
        "chunk_size": 16384,
        "block_size": 128,
        "threshold": 0.9,
        "head_sparsity": 0.5,
        "kv_sparsity": 0.2,
    },
)
