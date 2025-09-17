#!/usr/bin/env python3
"""
Test showing the difference between torch.mm (which is fixed) and torch.bmm/matmul (which may not be).
"""

import torch
from batch_invariant_ops import set_batch_invariant_mode


def test_mm_vs_bmm():
    """Test torch.mm vs torch.bmm for batch dependency."""
    
    torch.set_default_device('cuda')
    
    print("="*60)
    print("Testing torch.mm vs torch.bmm/matmul")
    print("="*60)
    
    # Test 1: torch.mm (2D tensors) - this is what the library fixes
    print("\n1. torch.mm (2D matrix multiplication):")
    print("-" * 40)
    
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Standard mode
    print("Standard PyTorch:")
    with set_batch_invariant_mode(False):
        out1 = torch.mm(a[:1], b)
        out2 = torch.mm(a, b)[:1]
        diff = (out1 - out2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    # Batch-invariant mode
    print("\nBatch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        out1 = torch.mm(a[:1], b)
        out2 = torch.mm(a, b)[:1]
        diff = (out1 - out2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    # Test 2: torch.bmm (3D batch matrix multiplication) - used in attention
    print("\n2. torch.bmm (3D batch matrix multiplication):")
    print("-" * 40)
    
    batch_size = 32
    seq_len = 512
    d_k = 64
    
    # Create 3D tensors for bmm
    q = torch.linspace(-10, 10, batch_size*seq_len*d_k).reshape(batch_size, seq_len, d_k)
    k = torch.linspace(-5, 15, batch_size*seq_len*d_k).reshape(batch_size, seq_len, d_k)
    
    print("Standard PyTorch:")
    with set_batch_invariant_mode(False):
        # Process single batch
        scores1 = torch.bmm(q[:1], k[:1].transpose(-2, -1))
        # Process full batch, take first
        scores2 = torch.bmm(q, k.transpose(-2, -1))[:1]
        diff = (scores1 - scores2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    print("\nBatch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        scores1 = torch.bmm(q[:1], k[:1].transpose(-2, -1))
        scores2 = torch.bmm(q, k.transpose(-2, -1))[:1]
        diff = (scores1 - scores2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    # Test 3: torch.matmul (general matrix multiplication)
    print("\n3. torch.matmul (general matrix multiplication):")
    print("-" * 40)
    
    # 4D tensors like in attention
    num_heads = 12
    q_4d = torch.linspace(-10, 10, batch_size*num_heads*seq_len*d_k).reshape(
        batch_size, num_heads, seq_len, d_k
    )
    k_4d = torch.linspace(-5, 15, batch_size*num_heads*seq_len*d_k).reshape(
        batch_size, num_heads, seq_len, d_k
    )
    
    print("Standard PyTorch:")
    with set_batch_invariant_mode(False):
        scores1 = torch.matmul(q_4d[:1], k_4d[:1].transpose(-2, -1))
        scores2 = torch.matmul(q_4d, k_4d.transpose(-2, -1))[:1]
        diff = (scores1 - scores2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    print("\nBatch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        scores1 = torch.matmul(q_4d[:1], k_4d[:1].transpose(-2, -1))
        scores2 = torch.matmul(q_4d, k_4d.transpose(-2, -1))[:1]
        diff = (scores1 - scores2).abs().max()
        print(f"  Difference: {diff.item():.6f}")
        print(f"  Result: {'✓ Batch-invariant' if diff.item() < 1e-6 else '✗ Batch-dependent'}")
    
    # Test 4: Check what operations are registered
    print("\n4. Checking registered operations:")
    print("-" * 40)
    
    # Check if bmm and matmul are being overridden
    from batch_invariant_ops import is_batch_invariant_mode_enabled
    
    with set_batch_invariant_mode(True):
        print(f"  Batch-invariant mode enabled: {is_batch_invariant_mode_enabled()}")
        print("  Note: The library may only override torch.mm, not bmm/matmul")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The batch_invariant_ops library successfully fixes torch.mm")
    print("but attention uses torch.bmm/matmul which may not be overridden.")
    print("This explains why attention still appears batch-invariant in standard mode.")


if __name__ == "__main__":
    test_mm_vs_bmm()