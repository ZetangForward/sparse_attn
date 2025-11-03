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
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def main():
    model_path = "/data/lcm_lab/qqt/project/SparseAttn/sparseattn/checkpoints/masksonly_Meta-Llama-3.1-8B-Instruct_bsz16_steps1000_lr1e-5_warmup0.1_sp0.7_cw2048_mlr1.0_rlr1.0debug_wfrozen"
    # model_path = "/data/lcm_lab/qqt/project/SparseAttn/sparseattn/checkpoints/masksonly_Meta-Llama-3.1-8B-Instruct_bsz16_steps1000_lr1e-5_warmup0.1_sp0.7_cw1024_mlr1.0_rlr1.0qwen_streaming_32k_prulong-sp_0.7_wfrozen"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_sparse_model(model_path)

    config = model.config
    threshold = getattr(config, "suggested_threshold", 0.5)
    # threshold = 0
    print(f"Using threshold: {threshold}")
    # model.set_threshold_for_deterministic(threshold)
    model.eval()

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for i in range(1):
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        after = torch.cuda.memory_allocated()
        print(
            f"Iteration {i}: Δ {(after - before) / 1024**2:.2f} MB, reserved {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        )


if __name__ == "__main__":
    main()
