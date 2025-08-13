"""
Unit tests for SparseAttn library.
"""

import torch
import pytest
import math

# Try to import the modules - skip tests if not available
try:
    from sparseattn import (
        Xattention_prefill,
        Flexprefill_prefill,
        Minference_prefill,
        Full_prefill,
    )

    HAS_SPARSEATTN = True
except ImportError:
    HAS_SPARSEATTN = False


@pytest.fixture
def sample_data():
    """Create sample query, key, and value tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size, num_heads, seq_len, head_dim = 1, 4, 256, 64
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


@pytest.mark.skipif(
    not HAS_SPARSEATTN or not torch.cuda.is_available(),
    reason="SparseAttn not available or CUDA not available",
)
class TestXattention:
    """Test Xattention implementation."""

    def test_output_shape(self, sample_data):
        """Test that output has the correct shape."""
        query, key, value = sample_data
        output = Xattention_prefill(
            stride=16,
            query_states=query,
            key_states=key,
            value_states=value,
            block_size=128,
            use_triton=True,
            chunk_size=2048,
            threshold=0.95,
            causal=True,
        )
        assert output.shape == query.shape

    def test_causal_mask(self, sample_data):
        """Test causal mask functionality."""
        query, key, value = sample_data
        output_causal = Xattention_prefill(
            stride=16,
            query_states=query,
            key_states=key,
            value_states=value,
            block_size=128,
            use_triton=True,
            chunk_size=2048,
            threshold=0.95,
            causal=True,
        )
        output_non_causal = Xattention_prefill(
            stride=16,
            query_states=query,
            key_states=key,
            value_states=value,
            block_size=128,
            use_triton=True,
            chunk_size=2048,
            threshold=0.95,
            causal=False,
        )

        # Outputs should be different with and without causal mask
        assert not torch.allclose(output_causal, output_non_causal, atol=1e-6)


@pytest.mark.skipif(
    not HAS_SPARSEATTN or not torch.cuda.is_available(),
    reason="SparseAttn not available or CUDA not available",
)
class TestFlexPrefill:
    """Test FlexPrefill implementation."""

    def test_output_shape(self, sample_data):
        """Test that output has the correct shape."""
        query, key, value = sample_data
        output = Flexprefill_prefill(q=query, k=key, v=value, block_size=64)
        assert output.shape == query.shape


@pytest.mark.skipif(
    not HAS_SPARSEATTN or not torch.cuda.is_available(),
    reason="SparseAttn not available or CUDA not available",
)
class TestMinference:
    """Test Minference implementation."""

    def test_output_shape(self, sample_data):
        """Test that output has the correct shape."""
        query, key, value = sample_data
        output = Minference_prefill(
            query_states=query,
            key_states=key,
            value_states=value,
            vertical_size=1000,
            slash_size=6096,
            adaptive_budget=0.1,
        )
        assert output.shape == query.shape

    def test_adaptive_budget(self, sample_data):
        """Test adaptive budget functionality."""
        query, key, value = sample_data
        output1 = Minference_prefill(
            query_states=query, key_states=key, value_states=value, adaptive_budget=0.1
        )
        output2 = Minference_prefill(
            query_states=query,
            key_states=key,
            value_states=value,
            vertical_size=50,
            slash_size=100,
        )

        # Both methods should produce outputs
        assert output1.shape == query.shape
        assert output2.shape == query.shape


@pytest.mark.skipif(
    not HAS_SPARSEATTN or not torch.cuda.is_available(),
    reason="SparseAttn not available or CUDA not available",
)
class TestFullPrefill:
    """Test FullPrefill implementation."""

    def test_output_shape(self, sample_data):
        """Test that output has the correct shape."""
        query, key, value = sample_data
        output = Full_prefill(
            query_states=query, key_states=key, value_states=value, causal=True
        )
        assert output.shape == query.shape

    def test_causal_mask(self, sample_data):
        """Test causal mask functionality."""
        query, key, value = sample_data
        output_causal = Full_prefill(
            query_states=query, key_states=key, value_states=value, causal=True
        )
        output_non_causal = Full_prefill(
            query_states=query, key_states=key, value_states=value, causal=False
        )

        # Outputs should be different with and without causal mask
        assert not torch.allclose(output_causal, output_non_causal, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
