import json
import torch
import time
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import datetime
from dataclasses import dataclass

from transformers import logging

logging.set_verbosity_error()


# -----------------------------------------------------------------------------
# 1. 统一模型加载器
# -----------------------------------------------------------------------------
def load_model(model_path, is_sparse):
    print(f"\n📥 [System] Loading model from: {model_path} ...")
    config_path = f"{model_path}/config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"❌ Config not found at {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    archs = config_data.get("architectures", [])
    arch_name = archs[0] if archs else "Unknown"
    print(f"🏗️  [System] Detected architecture: {arch_name}")

    is_NSA = False

    if is_sparse:
        # --- 自定义 Sparse 模型注册逻辑 ---
        if "PawLlama" in arch_name:
            from sparseattn.training.eval.modeling_flash_llama import (
                PawLlamaForCausalLM,
                PawLlamaConfig,
            )

            AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
            model_cls = PawLlamaForCausalLM
        elif "PawQwen" in arch_name:
            from sparseattn.efficiency.model.modeling_flash_qwen_xattn import (
                PawQwen3ForCausalLM,
                PawQwen3Config,
            )

            # from sparseattn.efficiency.model.modeling_flash_qwen_prulong import (
            #     PawQwen3ForCausalLM, PawQwen3Config
            # )
            AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
            model_cls = PawQwen3ForCausalLM
        else:
            is_NSA = True
            from transformers import AutoConfig
            from sparseattn.efficiency.model.modeling_llama_nsa import LlamaForCausalLM 
            from sparseattn.efficiency.model.modeling_qwen_nsa import Qwen3ForCausalLM 
            from sparseattn.training.block_sparse_attention_triton.native_sparse_attention.module.llama_nsa import LlamaNSA_prefill
            from sparseattn.training.block_sparse_attention_triton.native_sparse_attention.module.qwen3_nsa import Qwen3NSA_prefill

            # 1. 先加载 Config，并把 NSA 需要的参数注入进去
            #    这步很重要，因为替换后的 Attention 初始化时需要用到这些新参数
            config = AutoConfig.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            config._attn_implementation = "flash_attention_2"
            config.compress_type = "linear"#"avgpool","weightedpool"
            config.kernel_size = 128
            config.kernel_stride = 64
            config.block_size = 64
            config.topk = 128
            config.init_blocks = 1
            config.local_blocks = 2
            config.window_size = 512
            # 2. 【关键步骤】执行 Monkey Patch 替换
            #    把 modeling_qwen3 模块里的 Qwen3Attention 强行变成你的 Qwen3NSA
            print("正在使用NSA_prefill")

            model_cls = Qwen3ForCausalLM
            
    else:
        if "PawLlama" in arch_name:
            from sparseattn.training.eval.modeling_flash_llama import (
                PawLlamaForCausalLM,
                PawLlamaConfig,
            )

            AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
            model_cls = PawLlamaForCausalLM
        elif "PawQwen" in arch_name:
            from sparseattn.efficiency.model.modeling_flash_qwen_full import (
                PawQwen3ForCausalLM,
                PawQwen3Config,
            )

            AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
            model_cls = PawQwen3ForCausalLM

    if is_NSA:
        # 3. 正常加载模型
            #    此时 from_pretrained 内部实例化 Attention 时，实际上实例化的是 Qwen3NSA
            #    并且它会自动尝试加载权重
            model = model_cls.from_pretrained(
                model_path,
                config=config,  # 传入修改后的 config
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            print("NSA loaded successfully.")
            print(f"Model config architectures: {model.config.architectures}")
            print(f"Total parameters in loaded model: {model.num_parameters():,}")
    else:
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, is_sparse


# -----------------------------------------------------------------------------
# 2. 核心评测函数
# -----------------------------------------------------------------------------
import time

def evaluate_efficiency(model, input_ids, gen_len=10, is_sparse=False):
    # -----------------------------------------------------------
    # 辅助函数：强行同步所有 GPU
    # -----------------------------------------------------------
    def sync_all_devices():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

    # --- A. Prefill 阶段 ---
    
    # 1. 先同步所有设备，确保之前的残留任务跑完
    sync_all_devices()
    
    # 2. 记录 CPU 时间
    start_time = time.time()

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)

    # 3. 再次同步所有设备，确保刚才的 Prefill 在所有卡上都彻底跑完
    sync_all_devices()
    
    end_time = time.time()
    prefill_time_ms = (end_time - start_time) * 1000  # 转换为毫秒

    past_key_values = outputs.past_key_values

    # 获取 Sparsity
    current_sparsity = 0.0
    if is_sparse:
        try:
            current_sparsity = getattr(model, "prefill_sparsity", None)
            if isinstance(current_sparsity, torch.Tensor):
                current_sparsity = current_sparsity.item()
        except:
            pass

    # 准备 Decode
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)

    # --- B. Decode 阶段 ---
    sync_all_devices()
    start_time = time.time()

    with torch.inference_mode():
        for _ in range(gen_len):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            # 注意：这里的 next_token 可能在 GPU 1 上，但 model 接受它会自动处理
            next_token = (
                torch.argmax(outputs.logits[:, -1, :], dim=-1)
                .unsqueeze(1)
                .to(model.device) # 保持在模型主设备或自动流转
                .contiguous()
            )

    sync_all_devices()
    end_time = time.time()
    decode_time_ms = (end_time - start_time) * 1000

    return {
        "prefill_ms": prefill_time_ms,
        "decode_ms_total": decode_time_ms,
        "decode_ms_per_token": decode_time_ms / gen_len,
        "sparsity": current_sparsity,
    }

# -----------------------------------------------------------------------------
# 3. 批量测试执行器 (修改：截断逻辑)
# -----------------------------------------------------------------------------
def run_benchmark_suite(
    model_path, samples, tokenizer, gen_len=10, max_len=4096, is_sparse=False
):
    model, is_sparse = load_model(model_path, is_sparse)
    results = []

    # Warmup
    # print("🔥 [System] Warming up GPU...")
    # dummy = tokenizer.encode("Warmup " * 10, return_tensors="pt").to(model.device)
    # evaluate_efficiency(model, dummy, gen_len=2, is_sparse=is_sparse)
    # breakpoint()

    print(
        f"🏃 [System] Running benchmark on {len(samples)} samples (Max Len: {max_len})..."
    )

    for i, item in enumerate(samples):
        gc.collect()
        torch.cuda.empty_cache()
        input_text = item["input"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        seq_len = input_ids.shape[-1]

        note = ""
        # === 截断逻辑 ===
        if seq_len > max_len:
            # 策略：保留头部一半配额，保留尾部一半配额，中间切掉
            # 这样只要 max_input_len >= 200，就绝对能保留前100和后100
            half_len = max_len // 2

            # 1. 取前 half_len (包含前100)
            head_part = input_ids[:, :half_len]

            # 2. 取后 (max - half) (包含后100)
            # 注意：用 (max - half) 而不是 half 是为了处理奇数长度的情况
            tail_part = input_ids[:, -(max_len - half_len) :]

            # 3. 拼接
            input_ids = torch.cat([head_part, tail_part], dim=1)

            note = f"✂️ (Mid-Trunc {seq_len} -> {max_len})"
            seq_len = max_len
        # ===============

        res = evaluate_efficiency(
            model, input_ids, gen_len=gen_len, is_sparse=is_sparse
        )
        # 跳过第一个预热阶段
        if i == 0:
            print("🔥(Skipping warmup sample)")
            del input_ids
            continue

        res["seq_len"] = seq_len
        results.append(res)

        # 实时打印进度
        print(
            f"  Sample {i} {note}: 📏 Len {seq_len} | ⚡ Prefill {res['prefill_ms']:.1f}ms | ⏩ Decode {res['decode_ms_per_token']:.2f}ms/tok \
                | ESR {res['sparsity']:.2f}"
        )

        del input_ids

    # 清理模型释放显存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 [System] Model unloaded & Memory cleared.\n")

    return results


# -----------------------------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------------------------
def main():
    FULL_MODEL_CACHE = {
        # 8192:   (853.27 , 62.37),   
        # 16384:  (1743.36, 	70.45 ),
        # 32768:  (4068.16, 	92.05 ),
        # 65536:  (10977.39, 	135.04 ),
        # 131072: (34370.44, 	225.32 ),
        # 262144: (120104.85, 	429.09),
    }
    
    # ================= 配置区域 =================
    # sparse_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.12steps300_full_streaming_64k_qwen3-8b_end0.7_wfrozen"
    # sparse_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.12steps300_full_streaming_64k_llama3-8b_end0.7_wfrozen"
    # sparse_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.5steps300_full_streaming_64k_qwen3-4b_wfrozen"
    sparse_model_path = "/data2/hf_models/Qwen3-8B"
    # sparse_model_path = ""
    full_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen"
    # full_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.3steps300_full_streaming_64k_llama3.1-8b_wfrozen"
    # full_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-200"

    data_path = "/data1/lcm_lab/sora/loomeval/benchmarks/General/RULER/data/fwe_262144.jsonl"

    num_samples = 3  # 每个长度测试的样本数
    gen_len = 1  # 生成长度

    # target_lengths_k = [8, 16, 32, 64, 128]
    target_lengths_k = [256]
    # target_lengths_k = [256]
    target_lengths = [k * 1024 for k in target_lengths_k]

    # 1. 准备数据
    print(f"📂 [Init] Reading data from {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(sparse_model_path, trust_remote_code=True)

    raw_samples = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if len(raw_samples) >= num_samples + 1:
                break
            try:
                raw_samples.append(json.loads(line))
            except:
                pass

    if not raw_samples:
        print("❌ Error: No data loaded.")
        return

    final_excel_summary = []

    # === 外层循环：遍历不同长度 ===
    for target_len in target_lengths:
        print("\n" + "█" * 80)
        print(f"🎯 TARGET LENGTH: {target_len} ({target_len / 1024:.0f}K)")
        print("█" * 80)

        print(f"🔹 Sparse Model @ {target_len}")
        sparse_results = run_benchmark_suite(
            sparse_model_path, raw_samples, tokenizer, gen_len, target_len, True
        )
        
        # ================== 修改逻辑开始 ==================
        # 检查是否有缓存数据
        if target_len in FULL_MODEL_CACHE:
            print(f"🔄 [System] Using CACHED results for Full Model @ {target_len}")
            cached_prefill, cached_decode = FULL_MODEL_CACHE[target_len]
            
            # 构造假的 full_results 列表，长度与 num_samples 一致，以便后续做对比计算
            full_results = []
            for _ in range(num_samples):
                full_results.append({
                    "prefill_ms": cached_prefill,
                    "decode_ms_per_token": cached_decode,
                    "seq_len": target_len, # 假装长度完全匹配
                    "sparsity": 0.0
                })
        else:
            # 如果缓存里没有这个长度的数据，才去真正跑 Full 模型
            print(f"🔸 Full Model @ {target_len}")
            full_results = run_benchmark_suite(
                full_model_path, raw_samples, tokenizer, gen_len, target_len, False
            )
        # ================== 修改逻辑结束 ==================
        
        # print(f"🔸 Full Model @ {target_len}")
        # full_results = run_benchmark_suite(
        #     full_model_path, raw_samples, tokenizer, gen_len, target_len, False
        # )

        print(
            "\n"
            + "📊" * 15
            + f" COMPARISON REPORT ({target_len / 1024:.0f}K) "
            + "📊" * 15
        )
        print(
            f"{'ID':<4} | {'Len':<6} | {'Sparse (ms)':<18} | {'Full (ms)':<18} | {'🚀 Speedup (Full/Sparse)':<22}"
        )
        print(
            f"{'':<4} | {'':<6} | {'⚡Prefill':<9} {'⏩Decode':<8} | {'⚡Prefill':<9} {'⏩Decode':<8} | {'⚡Prefill':<9} {'⏩Decode':<8}"
        )
        print("-" * 100)

        avg_speedup_prefill = []
        avg_speedup_decode = []

        min_len = min(len(full_results), len(sparse_results))

        for i in range(min_len):
            res_s = sparse_results[i]
            res_f = full_results[i]

            len_tok = res_s["seq_len"]

            # 提取指标
            p_s, d_s = res_s["prefill_ms"], res_s["decode_ms_per_token"]
            p_f, d_f = res_f["prefill_ms"], res_f["decode_ms_per_token"]

            # 计算加速比
            speedup_p = p_f / p_s if p_s > 0 else 0
            speedup_d = d_f / d_s if d_s > 0 else 0

            avg_speedup_prefill.append(speedup_p)
            avg_speedup_decode.append(speedup_d)

            # === 这里保留你的高亮与 Emoji 逻辑 ===
            sp_p_str = (
                f"\033[92m{speedup_p:<8.2f}x\033[0m"
                if speedup_p > 1.0
                else f"{speedup_p:<8.2f}x"
            )
            sp_d_str = (
                f"\033[92m{speedup_d:<8.2f}x\033[0m"
                if speedup_d > 1.0
                else f"{speedup_d:<8.2f}x"
            )

            if speedup_p > 1.0:
                sp_p_str += "🔥"
            if speedup_d > 1.0:
                sp_d_str += "🔥"

            print(
                f"{i + 1:<4} | {len_tok:<6} | {p_s:<9.1f} {d_s:<8.2f} | {p_f:<9.1f} {d_f:<8.2f} | {sp_p_str} {sp_d_str}"
            )

        print("-" * 100)

        if avg_speedup_prefill:
            print(
                f"✨ Average Speedup -> Prefill: \033[1m{sum(avg_speedup_prefill) / len(avg_speedup_prefill):.2f}x\033[0m"
            )
            print(
                f"✨ Average Speedup -> Decode : \033[1m{sum(avg_speedup_decode) / len(avg_speedup_decode):.2f}x\033[0m"
            )

            valid_sp = [r["sparsity"] for r in sparse_results if r]
            if valid_sp:
                avg_spa = sum(valid_sp) / len(valid_sp)
                print(f"📉 Average Sparse Rate: {avg_spa:.4f}")
        else:
            print("⚠️ No valid samples comparing.")

        # 提取数据
        s_prefills = [r["prefill_ms"] for r in sparse_results[:min_len]]
        s_decodes = [r["decode_ms_per_token"] for r in sparse_results[:min_len]]
        f_prefills = [r["prefill_ms"] for r in full_results[:min_len]]
        f_decodes = [r["decode_ms_per_token"] for r in full_results[:min_len]]
        sparsities = [r.get("sparsity", 0) for r in sparse_results[:min_len]]

        # 计算均值
        avg_s_p = sum(s_prefills) / len(s_prefills)
        avg_s_d = sum(s_decodes) / len(s_decodes)
        avg_f_p = sum(f_prefills) / len(f_prefills)
        avg_f_d = sum(f_decodes) / len(f_decodes)
        avg_spar = sum(sparsities) / len(sparsities)

        # 计算加速比
        speedup_p = avg_f_p / avg_s_p if avg_s_p > 0 else 0
        speedup_d = avg_f_d / avg_s_d if avg_s_d > 0 else 0

        # 4. 构建一行 Excel 数据 (严格对齐图片格式)
        row_data = {
            "Length": target_len,
            "Sparse Prefill (ms)": round(avg_s_p, 2),
            "Sparse Decode (ms)": round(avg_s_d, 2),
            "Full Prefill (ms)": round(avg_f_p, 2),
            "Full Decode (ms)": round(avg_f_d, 2),
            "Speedup Prefill": round(speedup_p, 2),
            "Speedup Decode": round(speedup_d, 2),
            "Sparsity": round(avg_spar, 2),
        }

        final_excel_summary.append(row_data)

    if final_excel_summary:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/benchmark_length_{ts}.xlsx"

        print(f"\n💾 Saving summary table to {file_name}...")

        # 创建 DataFrame 并指定列顺序 (防止字典乱序)
        columns_order = [
            "Length",
            "Sparse Prefill (ms)",
            "Sparse Decode (ms)",
            "Full Prefill (ms)",
            "Full Decode (ms)",
            "Speedup Prefill",
            "Speedup Decode",
            "Sparsity",
        ]

        df = pd.DataFrame(final_excel_summary)
        # 重新排序列以确保完全符合图片的视觉顺序
        df = df[columns_order]

        df.to_excel(file_name, index=False)
        print("✅ Excel Saved Successfully (Format matches the image).")
    else:
        print("⚠️ No data collected.")


if __name__ == "__main__":
    main()
