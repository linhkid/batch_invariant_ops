"""Test for detecting nondeterminism in attention mechanisms."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    batch_size, num_heads, seq_len, d_k = query.shape
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    if attn_mask is not None:
        scores = scores + attn_mask
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, value)
    
    return output


def test_attention_batch_variance(
    seq_len: int = 512,
    d_model: int = 768,
    num_heads: int = 12,
    num_iterations: int = 100,
    device: str = "cuda"
) -> dict:
    """Test if attention produces different results with different batch sizes."""
    
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    
    # Create random input
    torch.manual_seed(42)
    single_input = torch.randn(1, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
    
    # Test configurations
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    print(f"\nTesting attention nondeterminism across {num_iterations} iterations")
    print(f"Sequence length: {seq_len}, Model dim: {d_model}, Heads: {num_heads}")

    
    for batch_size in batch_sizes:
        outputs = []
        
        # Expand input to batch size
        if batch_size == 1:
            batch_input = single_input
        else:
            # Concatenate the same input multiple times
            batch_input = single_input.expand(batch_size, -1, -1, -1).contiguous()
        
        # Run multiple times to detect nondeterminism
        for _ in range(num_iterations):
            with torch.no_grad():
                # Use same input as Q, K, V for simplicity
                output = scaled_dot_product_attention(
                    batch_input, batch_input, batch_input
                )
                # Extract first item from batch
                outputs.append(output[0].cpu().numpy())
        
        # Check variance
        outputs = np.array(outputs)
        variance = np.var(outputs)
        unique_outputs = len(np.unique(outputs.reshape(num_iterations, -1), axis=0))
        
        results[batch_size] = {
            'variance': float(variance),
            'unique_outputs': unique_outputs,
            'max_diff': float(np.max(outputs) - np.min(outputs))
        }
        
        print(f"Batch size {batch_size:2d}: {unique_outputs:3d} unique outputs, "
              f"variance={variance:.2e}, max_diff={results[batch_size]['max_diff']:.2e}")
    
    # Compare outputs across different batch sizes
    print("\nCross-batch comparison:")
    reference = None
    for batch_size in batch_sizes:
        torch.manual_seed(42)
        batch_input = single_input.expand(batch_size, -1, -1, -1).contiguous()
        
        with torch.no_grad():
            output = scaled_dot_product_attention(
                batch_input, batch_input, batch_input
            )[0].cpu().numpy()
        
        if reference is None:
            reference = output
        else:
            diff = np.max(np.abs(output - reference))
            print(f"  Batch size 1 vs {batch_size}: max_diff={diff:.2e}")
    
    return results


def test_torch_sdpa_nondeterminism(
    seq_len: int = 512,
    d_model: int = 768,
    num_heads: int = 12,
    num_iterations: int = 100,
    device: str = "cuda"
) -> dict:
    """Test PyTorch's native scaled_dot_product_attention for nondeterminism."""
    
    if not hasattr(F, 'scaled_dot_product_attention'):
        print("PyTorch scaled_dot_product_attention not available (requires PyTorch 2.0+)")
        return {}
    
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    
    torch.manual_seed(42)
    single_input = torch.randn(1, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    print(f"\nTesting PyTorch's F.scaled_dot_product_attention")

    
    for batch_size in batch_sizes:
        outputs = []
        
        if batch_size == 1:
            batch_input = single_input
        else:
            batch_input = single_input.expand(batch_size, -1, -1, -1).contiguous()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = F.scaled_dot_product_attention(
                    batch_input, batch_input, batch_input,
                    dropout_p=0.0,
                    is_causal=False
                )
                outputs.append(output[0].cpu().numpy())
        
        outputs = np.array(outputs)
        variance = np.var(outputs)
        unique_outputs = len(np.unique(outputs.reshape(num_iterations, -1), axis=0))
        
        results[batch_size] = {
            'variance': float(variance),
            'unique_outputs': unique_outputs,
            'max_diff': float(np.max(outputs) - np.min(outputs))
        }
        
        print(f"Batch size {batch_size:2d}: {unique_outputs:3d} unique outputs, "
              f"variance={variance:.2e}, max_diff={results[batch_size]['max_diff']:.2e}")
    
    return results


def test_attention_with_batch_invariant_ops():
    """Test attention using batch-invariant operations."""
    try:
        from batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode

        print("Testing with Batch-Invariant Mode DISABLED")

        disable_batch_invariant_mode()
        results_normal = test_attention_batch_variance(num_iterations=50)

        print("Testing with Batch-Invariant Mode ENABLED")
        enable_batch_invariant_mode()
        results_invariant = test_attention_batch_variance(num_iterations=50)
        disable_batch_invariant_mode()
        
        # Compare results
        print("SUMMARY\n")
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            normal_unique = results_normal[batch_size]['unique_outputs']
            invariant_unique = results_invariant[batch_size]['unique_outputs']
            
            improvement = ((normal_unique - invariant_unique) / normal_unique * 100) if normal_unique > 1 else 0
            
            print(f"Batch {batch_size:2d}: Normal={normal_unique:3d} unique, "
                  f"Invariant={invariant_unique:3d} unique "
                  f"({improvement:.1f}% reduction)")
        
    except ImportError:
        print("batch_invariant_ops not available, running standard tests only")
        test_attention_batch_variance()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running tests on GPU.")
        
        # Test manual implementation
        test_attention_batch_variance()
        
        # Test PyTorch's native SDPA
        test_torch_sdpa_nondeterminism()
        
        # Test with batch-invariant ops
        test_attention_with_batch_invariant_ops()
    else:
        print("CUDA not available. These tests require a GPU.")