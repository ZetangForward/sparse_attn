import json
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM


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
        from sparseattn.training.eval.modeling_flash_llama_moe import (
            PawLlamaForCausalLM,
            PawLlamaConfig,
        )

        AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
        model_cls = PawLlamaForCausalLM
    elif "PawQwen" in arch_name:
        from sparseattn.training.eval.modeling_flash_qwen_moe import (
            PawQwen3ForCausalLM,
            PawQwen3Config,
        )

        AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
        model_cls = PawQwen3ForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def get_task(metadata_str):
    try:
        if isinstance(metadata_str, str):
            meta_dict = ast.literal_eval(metadata_str)
        elif isinstance(metadata_str, dict):
            meta_dict = metadata_str
        else:
            return None
        return meta_dict.get("task")
    except Exception:
        return None


def main():
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.3steps300_full_streaming_64k_llama3.1-8b_wfrozen_ablation"
    # model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.5steps300_full_streaming_64k_qwen3-4b_wfrozen"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_sparse_model(model_path)
    model.eval()

    sparsity = []

    longbench_prediction = [
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/RULER/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_RULER_128k/fwe_8192.jsonl"
    ]
    longbench = [
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/2wikimqa_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/gov_report_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/hotpotqa_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/lcc_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/multi_news_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/multifieldqa_en_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/passage_count_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/passage_retrieval_en_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/qasper_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/repobench-p_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/samsum_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/trec_e.jsonl",
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/triviaqa_e.jsonl",
    ]
    
    longbench_task = [
        "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/qasper_e.jsonl",
        # "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/2wikimqa_e.jsonl",
        # "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/multi_news_e.jsonl",
        # "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/lcc_e.jsonl",
        # "/data1/lcm_lab/sora/sparseattn_evaluation/results/General/LongBench/prediction/1.2steps300_full_streaming_64k_qwen3-8b_wfrozen_LongBench_128k/triviaqa_e.jsonl",
    ]
    
    for longbench_file in longbench_prediction:
        # 读取jsonl文件
        with open(longbench_file, "r") as f:
            data = [json.loads(line) for line in f]
        for i in range(2):
            input_ids = tokenizer.encode(data[i]["input_text"], return_tensors="pt").to(
                model.device
            )
            attention_mask = torch.ones_like(input_ids).to(model.device)
            actual_len = input_ids.shape[-1]

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=10,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            print(f"sparsity:{model.prefill_sparsity}")
            sparsity.append(model.prefill_sparsity)

            generated_ids = outputs[0][actual_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print("Response:", response)
    print(f"Average Sparsity:{sum(sparsity) / len(sparsity)}")


if __name__ == "__main__":
    main()
