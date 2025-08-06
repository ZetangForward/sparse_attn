#!/usr/bin/env python3
"""
Model loading and inference example for SparseAttn library.

This example demonstrates how to load a pre-trained LLaMA model with sparse attention
and perform inference on sample text inputs.
"""

import torch
import time
from transformers import AutoTokenizer

def load_model_and_tokenizer(model_path="meta-llama/Llama-2-7b-hf"):
    """
    Load a pre-trained LLaMA model with sparse attention mechanisms.
    
    Args:
        model_path (str): Path to the pre-trained model
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        print(f"Loading model from {model_path}...")
        # Try to import the load_model function
        from sparseattn.src.load_llama import load_model, FastPrefillConfig
        
        # Create a FastPrefillConfig
        config = FastPrefillConfig(
            threshold=0.95,
            print_detail=False,
            stride=16,
            metric="minfer"
        )
        
        # Load model with sparse attention
        model, tokenizer = load_model(
            fastprefillconfig=config,
            name_or_path=model_path
        )
        
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have:")
        print("1. A valid pre-trained LLaMA model")
        print("2. Sufficient GPU memory")
        print("3. Correct model path")
        raise

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generate text using the loaded model.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        tuple: (generated_text, generation_time)
    """
    print(f"Generating text for prompt: '{prompt[:50]}...'")  # 只显示前50个字符
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    print(f"Input length: {input_length} tokens")
    
    # Start timing
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # End timing
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_length = outputs.shape[1]
    
    print(f"Output length: {output_length} tokens (input + generated)")
    print(f"Generated {output_length - input_length} new tokens (max_new_tokens={max_new_tokens})")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Tokens per second: {(output_length - input_length) / generation_time:.2f}")
    
    return generated_text, generation_time

def run_inference_example():
    """Run a complete inference example."""

    print("This example shows how to load a model with sparse attention.")
    print("To run with a real model, you need to:")
    print("1. Have a pre-trained LLaMA model available")
    print("2. Update the model_path in the code")
    print("3. Ensure you have sufficient GPU memory")
    
    # Example usage (commented out as it requires a real model):

    model, tokenizer = load_model_and_tokenizer("/data/hf_models/Meta-Llama-3.1-8B-Instruct")
    
    prompts = [
        "The quick brown fox",
        "Artificial intelligence is",
        "Once upon a time"
    ]
    
    total_time = 0
    for prompt in prompts:
        generated, gen_time = generate_text(model, tokenizer, prompt * 16 * 1024, max_new_tokens=30)
        total_time += gen_time
        print(f"gen_time: {gen_time:.2f} seconds")
        print(f"Generated Length: {len(generated)} characters")
        print("-" * 50)
    
    print(f"Total generation time: {total_time:.2f} seconds")

    
    # Demonstrate attention mechanism usage directly
    print("\nDemonstrating direct attention mechanism usage...")
    from sparseattn import Xattention_prefill
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run attention mechanisms.")
        return
        
    # Test different sequence lengths
    seq_lengths = [16 * 1024 , 32 * 1024, 64 * 1024, 128 * 1024]
    results = []
    
    print("Testing attention mechanism with different sequence lengths:")
    print("=" * 60)
    
    for seq_len in seq_lengths:
        print(f"\nTesting with sequence length: {seq_len}")
        
        # Create sample data (as in basic_usage.py)
        batch_size, num_heads, head_dim = 1, 32, 128
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                            dtype=torch.float16, device='cuda')
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                        dtype=torch.float16, device='cuda')
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                            dtype=torch.float16, device='cuda')
        
        # Warmup run
        _ = Xattention_prefill(
            stride=16,
            query_states=query,
            key_states=key,
            value_states=value,
            block_size=128,
            use_triton=True,
            chunk_size=2048,
            threshold=0.95,
            causal=True
        )
        
        # Actual timed run
        torch.cuda.synchronize()
        start_time = time.time()
        
        output = Xattention_prefill(
            stride=16,
            query_states=query,
            key_states=key,
            value_states=value,
            block_size=128,
            use_triton=True,
            chunk_size=2048,
            threshold=0.95,
            causal=True
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate tokens per second
        total_tokens = batch_size * seq_len
        tokens_per_second = total_tokens / execution_time
        
        results.append((seq_len, execution_time, tokens_per_second))
        
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Throughput: {tokens_per_second:.2f} tokens/second")
        print(f"  Output shape: {output.shape}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Seq Length':<12} {'Time (s)':<12} {'Tokens/sec':<12}")
    print("-" * 60)
    for seq_len, exec_time, tokens_per_sec in results:
        print(f"{seq_len:<12} {exec_time:<12.4f} {tokens_per_sec:<12.2f}")


def main():
    """Main function to run the inference example."""
    print("SparseAttn Model Loading and Inference Example")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a machine with CUDA support.")
        return
    
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name()}")
    
    run_inference_example()
    
    print("\nInference example completed!")

if __name__ == "__main__":
    main()