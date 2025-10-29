import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------- CONFIG ----------------
RESULT_PATH = "/data/lcm_lab/sora/lab/LOOM-Eval/benchmarks/General/RULER/results/masksonly_Qwen3-4B_bsz16_steps250_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1.0_rlr1.0qwen_streaming_32k_prulong_wfrozen/qa_1.jsonl"
DATA_DIR = "/data/lcm_lab/sora/lab/LOOM-Eval/benchmarks/General/RULER/data"
MODEL_PATH = "/data/lcm_lab/qqt/project/SparseAttn/sparseattn/checkpoints/masksonly_Qwen3-4B_bsz16_steps250_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1.0_rlr1.0qwen_streaming_32k_layer_decay_test_only_layer_sparsity_new_debug_wfrozen"
SAVE_DIR = "/data/lcm_lab/qqt/project/SparseAttn/tests/analysis/attn_vis"
MAX_SAMPLES = 10
MAX_LEN = 32768
# ----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)


def load_sparse_model(model_path):
    config_path = f"{model_path}/config.json"
    with open(config_path, "r") as f:
        config_data = json.load(f)

    arch = config_data.get("architectures", [])
    if not arch:
        raise ValueError("No architecture found in config.json")

    arch_name = arch[0]
    print(f"Detected architecture: {arch_name}")

    if "PawLlama" in arch_name:
        from sparseattn.training.modeling_flash_llama import (
            PawLlamaForCausalLM,
            PawLlamaConfig,
        )

        AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
        model_cls = PawLlamaForCausalLM
    elif "PawQwen" in arch_name:
        from sparseattn.training.modeling_flash_qwen import (
            PawQwen3ForCausalLM,
            PawQwen3Config,
        )

        AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
        model_cls = PawQwen3ForCausalLM
    elif "PawPhi" in arch_name:
        from sparseattn.training.modeling_flash_phi import (
            PawPhi3ForCausalLM,
            PawPhi3Config,
        )

        AutoModelForCausalLM.register(PawPhi3Config, PawPhi3ForCausalLM)
        model_cls = PawPhi3ForCausalLM
    elif "Qwen" in arch_name:
        model_cls = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_attentions=True,
    )
    return model


print(f"✅ Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_sparse_model(MODEL_PATH)
model.eval()


# ---------- 辅助函数 ----------


def find_subsequence(sub, seq):
    """Find subsequence index range (start, end)"""
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i, i + len(sub)
    return None, None


def find_subsequence_by_text(answer_text, input_text, tokenizer):
    input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # 精确匹配
    for i in range(len(input_ids) - len(answer_ids) + 1):
        if input_ids[i : i + len(answer_ids)] == answer_ids:
            return i, i + len(answer_ids)

    # 退化策略：尝试用字符串匹配，找到 approximate span 再 token 对齐
    try:
        start_char = input_text.find(answer_text)
        if start_char == -1:
            return None, None
        prefix_text = input_text[:start_char]
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        return len(prefix_ids), len(prefix_ids) + len(answer_ids)
    except Exception:
        return None, None


def compute_attention_focus(attn_mean, start, end):
    """计算answer区域attention占比"""
    if start is None:
        return 0.0
    attn_last = attn_mean[-1]
    focus = attn_last[start:end].sum() / attn_last.sum()
    return focus.item()


# ---------- 数据加载 ----------
print("📂 Loading prediction results...")
results = [json.loads(l) for l in open(RESULT_PATH)]
error_cases = [r for r in results if r["score"] == 0.0 and r["LEN"] > 12 * 1024]
print(f"共 {len(error_cases)} 条错误case")

focus_scores = []

for case in tqdm(error_cases[:MAX_SAMPLES], desc="Analyzing cases"):
    task_name = case["task"]
    idx = case["index"]
    data_file = Path(DATA_DIR) / f"{task_name}.jsonl"
    if not data_file.exists():
        print(f"⚠️ 数据文件不存在: {data_file}")
        continue

    with open(data_file) as f:
        for l in f:
            ex = json.loads(l)
            if ex["index"] == idx:
                input_text = ex["input"]
                answer_text = ex["answer"]
                break
        else:
            continue

    # tokenizer
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    for i, a in enumerate(outputs.attentions):
        print(f"Layer {i}: {None if a is None else a.shape}")
    attn_mean = outputs.attentions[-1].mean(1).squeeze(0).cpu()  # (seq_len, seq_len)
    answer_spans = []
    input_ids = inputs["input_ids"].squeeze(0).tolist()

    if isinstance(answer_text, str):
        answer_texts = [answer_text]
    else:
        answer_texts = answer_text

    for ans in answer_text:
        answer_tokens = tokenizer(ans, add_special_tokens=False)["input_ids"]
        start, end = find_subsequence_by_text(ans, input_text, tokenizer)
        if start is not None:
            answer_spans.append((start, end))
        if start is None:
            print(f"⚠️ 无法匹配 answer: {ans}")
            print("answer tokens:", tokenizer.convert_ids_to_tokens(answer_ids))
            print(
                "input tokens (片段):", tokenizer.convert_ids_to_tokens(input_ids[:200])
            )

    attn_last = attn_mean[-1]
    if len(answer_spans) == 0:
        focus_score = 0.0
        topk_focus_ratio = 0.0
    else:
        mask = torch.zeros(attn_last.shape[0])
        for s, e in answer_spans:
            mask[s:e] = 1.0

        focus_score = (attn_last * mask).sum() / attn_last.sum()
        focus_score = focus_score.item()

        K = 2048
        seq_len = attn_last.shape[0]
        K = min(K, seq_len)
        topk_vals, topk_idx = torch.topk(attn_last, K)

        in_answer_mask = torch.zeros_like(topk_idx, dtype=torch.bool)
        for s, e in answer_spans:
            in_answer_mask |= (topk_idx >= s) & (topk_idx < e)
        topk_focus_ratio = in_answer_mask.sum().item() / K

    focus_scores.append(
        {
            "task": task_name,
            "index": idx,
            "focus_score": focus_score,
            "topk_focus_ratio": topk_focus_ratio,
        }
    )

    # ---------- 可视化 ----------
    plt.figure(figsize=(10, 3))
    plt.plot(
        attn_last.detach().to(torch.float32).cpu().numpy(), label="Attention weight"
    )
    for s, e in answer_spans:
        plt.axvspan(s, e, color="red", alpha=0.3)
    plt.scatter(
        topk_idx.cpu().to(torch.float32).numpy(),
        topk_vals.cpu().to(torch.float32).numpy(),
        color="blue",
        s=4,
        label=f"Top-{K}",
    )

    plt.legend(["Attention", "Answer spans", f"Top-{K} points"])
    plt.title(
        f"Case {idx} | Focus={focus_score:.4f} | TopK={topk_focus_ratio * 100:.1f}%"
    )
    plt.xlabel("Token position")
    plt.ylabel("Attention weight")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/attn_case_{idx}.png")
    plt.close()

# ---------- 保存结果 ----------
import pandas as pd

df = pd.DataFrame(focus_scores)
csv_path = Path(SAVE_DIR) / "attention_focus_summary.csv"
df.to_csv(csv_path, index=False)
print(f"✅ 分析完成，结果已保存到：{csv_path}")
print(f"📊 平均attention集中度: {df['focus_score'].mean():.4f}")
print(f"📊 平均Top-K重叠率: {df['topk_focus_ratio'].mean():.4f}")
