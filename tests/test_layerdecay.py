import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingArguments:
    world_size: int = 1
    max_steps: int = 1000
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 0
    seed: int = 42
    seq_parallel_size: int = 1


@dataclass
class DataArguments:
    single_seq: bool = False
    subsplit_length: Optional[int] = None
    per_device_varlen_padding: int = 4_294_967_296
    per_device_max_tokens: int = 4_294_967_296
    apply_instruct_masks: bool = False


import os
from streaming import StreamingDataset, Stream
from itertools import islice
from typing import Dict, Any, Iterator
from collections.abc import Iterator as ABCIterator


class SafeStream(Stream):
    def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
        unique_extension = (
            "." + str(os.getenv("SLURM_JOB_ID", "local")) + "-" + str(os.getpid())
        )
        super()._decompress_shard_part(
            zip_info, zip_filename, raw_filename + unique_extension, compression
        )
        os.rename(raw_filename + unique_extension, raw_filename)


class SortByLengthDataset(StreamingDataset):
    def __init__(self, *args, sort_by_length_size=1, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sort_by_length_size = sort_by_length_size
        self.data_args = data_args

    def _negative_item_cost(self, item):
        import math

        if "indices" in item:
            total_length = sum(end - start for start, end in item["indices"])
            return -math.log(max(total_length, 1))
        elif "length" in item:
            return -math.log(max(int(item["length"]), 1))
        else:
            return -math.log(max(len(item["input_ids"]), 1))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.sort_by_length_size <= 1:
            yield from super().__iter__()
        else:
            iterator = super().__iter__()
            while True:
                block = list(islice(iterator, self.sort_by_length_size))
                if not block:
                    return
                yield from sorted(block, key=self._negative_item_cost)


def build_dataset_for_debug(paths, data_args: DataArguments):
    streams = []
    for path in paths:
        clean_path = path.split("@")[0].split("#")[0]
        streams.append(SafeStream(remote=clean_path, local=clean_path))

    return SortByLengthDataset(
        streams=streams,
        shuffle=False,
        shuffle_seed=42,
        batch_size=1,
        epoch_size=None,
        sort_by_length_size=1,
        data_args=data_args,
        replication=1,
    )


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "/data/hf_models/Qwen3-4B"
DATA_PATH = "/data/lcm_lab/qqt/project/SparseAttn/sparseattn/data"
MAX_LEN = 1024 * 12
NUM_SAMPLES = 500
HEAD_GROUPS = 8
FIXED_K = 32
EPS = 1e-12

_MAX_TOKENS_FOR_GPU_SVD = 1024 * 12
_MAX_SAMPLED_TOKENS = 1024 * 12
_USE_SVD_LOW_RANK = True


def _eigvals_from_cov_on_cpu(cov_cpu):
    eigvals = torch.linalg.eigvalsh(cov_cpu)
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals = torch.flip(eigvals, dims=[0])
    return eigvals


def compute_truncated_erank(
    Q_head: torch.Tensor,
    fixed_k: int = FIXED_K,
    max_tokens_for_gpu_svd: int = _MAX_TOKENS_FOR_GPU_SVD,
    max_sampled_tokens: int = _MAX_SAMPLED_TOKENS,
    use_svd_lowrank: bool = _USE_SVD_LOW_RANK,
):
    """
    自适应计算单个 head 的截断 effective-rank。
    Q_head: Tensor [seq_len, head_dim], dtype float16/float32/float64 on some device.
    策略：
      - 若 seq_len <= max_tokens_for_gpu_svd，尝试在当前 device 上用低秩 SVD（svd_lowrank 或 full SVD）
      - 否则对 token 下采样到 max_sampled_tokens，然后在 CPU 上计算 cov = Q^T Q / (n-1)，对 cov 做 eig
    返回：
      - erank_k (float)
    """
    seq_len, head_dim = Q_head.shape
    if seq_len <= 1:
        return 1.0

    try:
        if seq_len <= max_tokens_for_gpu_svd and Q_head.device.type == "cuda":
            if use_svd_lowrank and hasattr(torch.linalg, "svd_lowrank"):
                q = min(fixed_k + 8, min(head_dim, seq_len))
                Q32 = Q_head.float()
                u, s, v = torch.linalg.svd_lowrank(Q32, q=q)
                s_vals = s**2
                eigvals = torch.zeros(
                    min(head_dim, fixed_k), device="cpu", dtype=torch.float32
                )
                s_cpu = s_vals[:fixed_k].cpu()
                eigvals[: len(s_cpu)] = s_cpu
                s_sum = float(eigvals.sum().item()) + EPS
                p = eigvals / s_sum
                Hk = -(p * (p + EPS).log()).sum().item()
                erank_k = math.exp(Hk)
                return erank_k

            Q32 = Q_head.float()
            u, s, v = torch.linalg.svd(Q32, full_matrices=False)

            eigvals = s**2
            eigvals = eigvals[:fixed_k].cpu()
            s_sum = float(eigvals.sum().item()) + EPS
            p = eigvals / s_sum
            Hk = -(p * (p + EPS).log()).sum().item()
            erank_k = math.exp(Hk)
            return erank_k

        if seq_len > max_sampled_tokens:
            idx = torch.linspace(
                0,
                seq_len - 1,
                steps=max_sampled_tokens,
                dtype=torch.long,
                device=Q_head.device,
            )
            Qs = Q_head[idx]  # still on original device
        else:
            Qs = Q_head

        Qs_cpu = Qs.detach().float().cpu()
        n = max(1, Qs_cpu.shape[0] - 1)
        cov = (Qs_cpu.t() @ Qs_cpu) / n  # shape (head_dim, head_dim)
        eigvals = _eigvals_from_cov_on_cpu(cov)  # cpu tensor
        eigvals = eigvals[:fixed_k]
        s_sum = float(eigvals.sum().item()) + EPS
        if s_sum <= 0:
            return 1.0
        p = eigvals / s_sum
        Hk = -(p * (p + EPS).log()).sum().item()
        erank_k = math.exp(Hk)

        del Qs, Qs_cpu, cov, eigvals, p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return erank_k

    except RuntimeError as e:
        try:
            if seq_len > max_sampled_tokens:
                idx = torch.linspace(
                    0,
                    seq_len - 1,
                    steps=max_sampled_tokens,
                    dtype=torch.long,
                    device=Q_head.device,
                )
                Qs = Q_head[idx]
            else:
                Qs = Q_head
            Qs_cpu = Qs.detach().float().cpu()
            n = max(1, Qs_cpu.shape[0] - 1)
            cov = (Qs_cpu.t() @ Qs_cpu) / n
            eigvals = _eigvals_from_cov_on_cpu(cov)
            eigvals = eigvals[:fixed_k]
            s_sum = float(eigvals.sum().item()) + EPS
            if s_sum <= 0:
                return 1.0
            p = eigvals / s_sum
            Hk = -(p * (p + EPS).log()).sum().item()
            erank_k = math.exp(Hk)
            del Qs, Qs_cpu, cov, eigvals, p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return erank_k
        except Exception:
            return 1.0


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    try:
        layers = model.model.layers
        embed_dim = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = embed_dim // num_heads
    except Exception as e:
        raise RuntimeError("This example expects a LLaMA-like architecture.") from e

    per_layer_head_eranks = [[[] for _ in range(num_heads)] for _ in range(len(layers))]
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            hidden_states = inputs[0]  # [batch, seq, hidden]

            self_attn = module.self_attn

            q = nn.functional.linear(
                hidden_states, self_attn.q_proj.weight, self_attn.q_proj.bias
            )
            b, s, q_hidden = q.shape

            # Get num_heads from model config (should be correct)
            num_heads = model.config.num_attention_heads

            # SAFETY CHECK: ensure divisibility
            if q_hidden % num_heads != 0:
                raise ValueError(
                    f"Q hidden size {q_hidden} is not divisible by num_heads {num_heads}. "
                    f"This may indicate a GQA/MQA model or config mismatch."
                )
            head_dim = q_hidden // num_heads  # This will be 128 for Qwen3-8B

            q = q.view(b, s, num_heads, head_dim)
            for bi in range(b):
                Qi = q[bi]
                for h in range(num_heads):
                    er = compute_truncated_erank(Qi[:, h, :])
                    per_layer_head_eranks[layer_idx][h].append(er)

        return hook

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    data_args = DataArguments(
        single_seq=True,
        per_device_max_tokens=MAX_LEN,
    )
    training_args = TrainingArguments(seed=42)

    from streaming.base.util import clean_stale_shared_memory

    clean_stale_shared_memory()

    print(f"Loading data from: {DATA_PATH}")
    debug_dataset = build_dataset_for_debug([DATA_PATH], data_args)

    count = 0
    for sample in debug_dataset:
        if count >= NUM_SAMPLES:
            break
        input_ids_raw = sample["input_ids"]

        if hasattr(input_ids_raw, "tolist"):
            input_ids = input_ids_raw.tolist()
        elif isinstance(input_ids_raw, (list, tuple)):
            input_ids = list(input_ids_raw)
        else:
            raise TypeError(f"Unexpected type for input_ids: {type(input_ids_raw)}")

        if len(input_ids) == 0:
            continue

        if len(input_ids) > MAX_LEN:
            input_ids = input_ids[:MAX_LEN]

        batch = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long).to(device),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(
                device
            ),
        }
        with torch.no_grad():
            _ = model(**batch)
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} samples...")

    print(f"Total samples processed: {count}")

    for h in hooks:
        h.remove()

    L = len(layers)
    H = num_heads
    avg_erank = torch.zeros(L, H)
    for i in range(L):
        for h in range(H):
            vals = per_layer_head_eranks[i][h]
            avg_erank[i, h] = float(np.mean(vals)) if vals else 0.0

    def quantile_bins(x, groups):
        qs = torch.quantile(x, torch.linspace(0, 1, groups + 1, device=x.device))
        bins = torch.bucketize(x, qs[1:-1], right=False)
        return bins

    head_groups_per_layer = torch.zeros(L, H, dtype=torch.long)
    for i in range(L):
        head_groups_per_layer[i] = quantile_bins(avg_erank[i], HEAD_GROUPS)
    head_groups_per_layer = (HEAD_GROUPS - 1) - head_groups_per_layer

    E = avg_erank.mean(dim=1)
    Delta = E[:-1] - E[1:]
    epsilon = float(torch.quantile(Delta, 0.75).item()) if len(Delta) > 0 else 0.0
    layer_group_boundaries = (
        (Delta > epsilon).nonzero(as_tuple=False).squeeze(-1).tolist()
    )
    candidate_retrieval_layers = []
    for i in range(1, L - 1):
        if E[i] > E[i - 1] and E[i] > E[i + 1]:
            candidate_retrieval_layers.append(i)

    print("avg_erank per layer shape:", avg_erank.shape)
    print("head_groups_per_layer shape:", head_groups_per_layer.shape)
    print("layer_group_boundaries:", layer_group_boundaries)
    print("candidate retrieval layers:", candidate_retrieval_layers)

    torch.save(
        {
            "avg_erank": avg_erank,
            "head_groups": head_groups_per_layer,
            "layer_boundaries": layer_group_boundaries,
            "retrieval_layers": candidate_retrieval_layers,
        },
        "erank_analysis_qwen_results.pt",
    )
    print("Results saved to erank_analysis_results.pt")


if __name__ == "__main__":
    main()
