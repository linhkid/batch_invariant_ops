#!/usr/bin/env python3
"""
Comprehensive test demonstrating attention nondeterminism and batch-invariant solution.

This experiment shows:
1. Standard PyTorch attention exhibits batch-size dependent nondeterminism
2. Our batch-invariant implementation produces identical results regardless of batch size
3. Performance comparison between standard and batch-invariant implementations
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple
import argparse


def run_attention_determinism_test(
    implementation: str = "standard",
    seq_len: int = 512,
    d_model: int = 768,
    num_heads: int = 12,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    num_iterations: int = 100,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """
    Test attention implementation for batch-size dependent nondeterminism.
    
    Args:
        implementation: "standard", "batch_invariant", or "both"
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations per batch size
        device: Device to run on
        verbose: Whether to print results
    
    Returns:
        Dictionary with test results
    """
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    
    results = {
        "config": {
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_iterations": num_iterations,
        },
        "implementations": {}
    }
    
    # Test each implementation
    implementations_to_test = []
    
    if implementation in ["standard", "both"]:
        implementations_to_test.append("standard")
    
    if implementation in ["batch_invariant", "both"]:
        try:
            from batch_invariant_ops.batch_invariant_attention import (
                batch_invariant_scaled_dot_product_attention
            )
            implementations_to_test.append("batch_invariant")
        except ImportError:
            print("Warning: batch_invariant_attention not available")
    
    for impl_name in implementations_to_test:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {impl_name.upper()} implementation")
            print(f"{'='*60}")
        
        impl_results = {}
        
        # Create reference input
        torch.manual_seed(42)
        reference_q = torch.randn(1, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
        reference_k = torch.randn(1, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
        reference_v = torch.randn(1, num_heads, seq_len, d_k, device=device, dtype=torch.float32)
        
        # Store reference output (batch_size=1)
        reference_output = None
        
        for batch_size in batch_sizes:
            # Expand inputs to batch size
            if batch_size == 1:
                q, k, v = reference_q, reference_k, reference_v
            else:
                q = reference_q.expand(batch_size, -1, -1, -1).contiguous()
                k = reference_k.expand(batch_size, -1, -1, -1).contiguous()
                v = reference_v.expand(batch_size, -1, -1, -1).contiguous()
            
            outputs = []
            times = []
            
            # Run multiple iterations
            for _ in range(num_iterations):
                torch.cuda.synchronize() if device == "cuda" else None
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    if impl_name == "standard":
                        if hasattr(F, 'scaled_dot_product_attention'):
                            output = F.scaled_dot_product_attention(
                                q, k, v, dropout_p=0.0, is_causal=False
                            )
                        else:
                            # Fallback for older PyTorch versions
                            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                            attn_weights = F.softmax(scores, dim=-1)
                            output = torch.matmul(attn_weights, v)
                    else:  # batch_invariant
                        output = batch_invariant_scaled_dot_product_attention(
                            q, k, v, dropout_p=0.0, is_causal=False
                        )
                
                torch.cuda.synchronize() if device == "cuda" else None
                times.append(time.perf_counter() - start_time)
                
                # Extract first item from batch
                outputs.append(output[0].cpu().numpy())
            
            # Analyze results
            outputs = np.array(outputs)
            
            # Store reference if this is batch_size=1
            if batch_size == 1:
                reference_output = outputs[0]
            
            # Calculate metrics
            variance = np.var(outputs)
            unique_outputs = len(np.unique(outputs.reshape(num_iterations, -1), axis=0))
            max_diff_internal = float(np.max(outputs) - np.min(outputs))
            
            # Compare to reference (batch_size=1)
            if reference_output is not None:
                max_diff_from_ref = float(np.max(np.abs(outputs[0] - reference_output)))
            else:
                max_diff_from_ref = 0.0
            
            avg_time = np.mean(times[1:])  # Skip first iteration (warmup)
            
            impl_results[batch_size] = {
                "variance": float(variance),
                "unique_outputs": unique_outputs,
                "max_diff_internal": max_diff_internal,
                "max_diff_from_ref": max_diff_from_ref,
                "avg_time_ms": avg_time * 1000,
            }
            
            if verbose:
                print(f"Batch {batch_size:2d}: {unique_outputs:3d} unique outputs, "
                      f"var={variance:.2e}, max_diff={max_diff_internal:.2e}, "
                      f"time={avg_time*1000:.2f}ms")
        
        results["implementations"][impl_name] = impl_results
    
    # Compare implementations if both were tested
    if len(implementations_to_test) == 2 and verbose:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        standard_results = results["implementations"]["standard"]
        invariant_results = results["implementations"]["batch_invariant"]
        
        print("\nDeterminism Improvement:")
        for batch_size in batch_sizes:
            std_unique = standard_results[batch_size]["unique_outputs"]
            inv_unique = invariant_results[batch_size]["unique_outputs"]
            improvement = ((std_unique - inv_unique) / std_unique * 100) if std_unique > 1 else 0
            print(f"  Batch {batch_size:2d}: {std_unique:3d} → {inv_unique:3d} unique outputs "
                  f"({improvement:.1f}% reduction)")
        
        print("\nPerformance Overhead:")
        for batch_size in batch_sizes:
            std_time = standard_results[batch_size]["avg_time_ms"]
            inv_time = invariant_results[batch_size]["avg_time_ms"]
            overhead = ((inv_time - std_time) / std_time * 100)
            print(f"  Batch {batch_size:2d}: {std_time:.2f}ms → {inv_time:.2f}ms "
                  f"({overhead:+.1f}% overhead)")
    
    return results


def run_model_test(
    model_name: str = "gpt2",
    num_iterations: int = 10,
    device: str = "cuda"
) -> Dict:
    """
    Test a real transformer model for attention nondeterminism.
    
    Args:
        model_name: Model to test (e.g., "gpt2")
        num_iterations: Number of iterations
        device: Device to run on
    
    Returns:
        Test results
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("transformers not installed. Install with: pip install transformers")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} Model")
    print(f"{'='*60}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare input
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results = {}
    
    for batch_size in batch_sizes:
        # Expand inputs
        batch_inputs = {
            k: v.expand(batch_size, -1) if v.dim() == 2 else v 
            for k, v in inputs.items()
        }
        
        outputs = []
        for _ in range(num_iterations):
            with torch.no_grad():
                output = model(**batch_inputs)
                # Extract first item's last hidden state
                outputs.append(output.last_hidden_state[0].cpu().numpy())
        
        outputs = np.array(outputs)
        unique_outputs = len(np.unique(outputs.reshape(num_iterations, -1), axis=0))
        
        results[batch_size] = {
            "unique_outputs": unique_outputs,
            "variance": float(np.var(outputs))
        }
        
        print(f"Batch {batch_size}: {unique_outputs} unique outputs")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test attention nondeterminism and batch-invariant solution"
    )
    parser.add_argument(
        "--implementation", 
        choices=["standard", "batch_invariant", "both"],
        default="both",
        help="Which implementation to test"
    )
    parser.add_argument(
        "--seq-len", 
        type=int, 
        default=512,
        help="Sequence length"
    )
    parser.add_argument(
        "--d-model", 
        type=int, 
        default=768,
        help="Model dimension"
    )
    parser.add_argument(
        "--num-heads", 
        type=int, 
        default=12,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num-iterations", 
        type=int, 
        default=100,
        help="Number of iterations per test"
    )
    parser.add_argument(
        "--test-model", 
        action="store_true",
        help="Test with a real transformer model"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="gpt2",
        help="Model to test (requires --test-model)"
    )
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Results may differ on CPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Run attention determinism test
    results = run_attention_determinism_test(
        implementation=args.implementation,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_iterations=args.num_iterations,
        device=device
    )
    
    # Optionally test with real model
    if args.test_model:
        model_results = run_model_test(
            model_name=args.model_name,
            num_iterations=min(args.num_iterations, 10),
            device=device
        )
        results["model_test"] = model_results

    print("TEST COMPLETE")

    
    # Print final summary
    if "standard" in results["implementations"] and "batch_invariant" in results["implementations"]:
        std_total_unique = sum(
            r["unique_outputs"] for r in results["implementations"]["standard"].values()
        )
        inv_total_unique = sum(
            r["unique_outputs"] for r in results["implementations"]["batch_invariant"].values()
        )
        
        print(f"\nTotal unique outputs across all batch sizes:")
        print(f"  Standard:        {std_total_unique}")
        print(f"  Batch-Invariant: {inv_total_unique}")
        
        if inv_total_unique == len(results["implementations"]["batch_invariant"]):
            print(f"\nBatch-invariant implementation achieved perfect determinism!")
    
    return results


if __name__ == "__main__":
    main()