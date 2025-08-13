#!/usr/bin/env python3
"""
Basic usage examples for SparseAttn library.

This example demonstrates how to use the different sparse attention mechanisms
provided by the SparseAttn library.
"""

import torch
from sparseattn import (
    Xattention_prefill,
    Flexprefill_prefill,
    Minference_prefill,
    Full_prefill,
)


def create_sample_data(batch_size=1, num_heads=32, seq_len=4096, head_dim=128):
    """Create sample query, key, and value tensors."""
    query = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )
    key = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )
    value = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )
    return query, key, value


def example_xattention():
    """Example usage of Xattention."""
    print("Running Xattention example...")
    query, key, value = create_sample_data()

    # Execute sparse attention computation
    output = Xattention_prefill(
        stride=16,  # Stride for the attention mechanism
        query_states=query,
        key_states=key,
        value_states=value,
        block_size=128,  # Block size for the attention
        use_triton=True,
        chunk_size=2048,  # Chunk size for processing
        threshold=0.95,  # Sparsification threshold
        causal=True,  # Whether to use causal mask
    )

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    return output


def example_flexprefill():
    """Example usage of FlexPrefill."""
    print("\nRunning FlexPrefill example...")
    query, key, value = create_sample_data()

    # Block sparse attention computation
    output = Flexprefill_prefill(
        q=query,
        k=key,
        v=value,
        block_size=64,  # Block size
    )

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    return output


def example_minference():
    """Example usage of Minference."""
    print("\nRunning Minference example...")
    query, key, value = create_sample_data()

    # Lightweight inference mode
    output = Minference_prefill(
        query_states=query,
        key_states=key,
        value_states=value,
        vertical_size=1000,  # Vertical sparse size
        slash_size=6096,  # Diagonal sparse size
        adaptive_budget=0.1,  # Adaptive budget
    )

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    return output


def example_fullprefill():
    """Example usage of FullPrefill."""
    print("\nRunning FullPrefill example...")
    query, key, value = create_sample_data()

    # Complete prefill implementation
    output = Full_prefill(
        query_states=query,
        key_states=key,
        value_states=value,
        causal=True,  # Whether to use causal mask
    )

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    return output


def main():
    """Run all examples."""
    print("SparseAttn Basic Usage Examples")
    print("=" * 40)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a machine with CUDA support.")
        return

    try:
        xattn_output = example_xattention()
        flex_output = example_flexprefill()
        minf_output = example_minference()
        full_output = example_fullprefill()

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
