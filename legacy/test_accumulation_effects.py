#!/usr/bin/env python3
"""
Experiment showing how torch.mm batch-dependency accumulates and affects
higher-level operations like mixed precision and quantization.

This demonstrates that small numerical differences in basic operations
can compound into significant differences in complex models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from batch_invariant_ops import set_batch_invariant_mode


class SimpleTransformerLayer(nn.Module):
    """A simple transformer layer that uses multiple torch.mm operations."""
    
    def __init__(self, d_model=768, num_heads=12, d_ff=3072):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers that will use torch.mm
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Feed-forward layers
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        
        # Layer norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Multi-head attention (simplified)
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections (each uses torch.mm internally)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores - this will use torch.matmul/bmm
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_linear(out)  # Another torch.mm
        
        # Residual connection and layer norm
        x = self.ln1(x + out)
        
        # Feed-forward network
        ff_out = self.ff2(F.relu(self.ff1(x)))  # Two more torch.mm operations
        x = self.ln2(x + ff_out)
        
        return x


def test_accumulation_in_transformer():
    """Test how torch.mm differences accumulate through a transformer layer."""
    
    torch.set_default_device('cuda')

    print("ACCUMULATION EXPERIMENT: Large Transformer Layer")
    
    # Create moderately larger model for clear effects without OOM
    model = SimpleTransformerLayer(d_model=1536, num_heads=32).eval()
    
    # Use reasonable batch size with extreme values to amplify differences
    batch_size = 64   # Reasonable batch size
    seq_len = 256     # Moderate sequence length
    d_model = 1536    # Bigger model size
    
    # Create input using EXTREME value range to amplify differences
    total_elements = batch_size * seq_len * d_model
    x = torch.linspace(-1000, 1000, total_elements).reshape(batch_size, seq_len, d_model)  # EXTREME range
    
    print(f"Model: SimpleTransformerLayer (d_model={d_model}, heads=32)")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}, value_range=±1000")
    print(f"Operations per forward pass: ~8 torch.mm calls")
    print(f"Total parameters: ~{(d_model*d_model*4 + d_model*d_model*3)/1e6:.1f}M")
    
    # Test 1: Standard mode
    print("\n1. Standard PyTorch Mode:")

    
    with set_batch_invariant_mode(False):
        with torch.no_grad():
            # Process single input
            out_single = model(x[:1])
            
            # Process full batch, take first
            out_batch = model(x)
            out_batch_first = out_batch[:1]
    
    diff_standard = (out_single - out_batch_first).abs().max().item()
    mean_diff_standard = (out_single - out_batch_first).abs().mean().item()
    
    print(f"  Max difference: {diff_standard:.6f}")
    print(f"  Mean difference: {mean_diff_standard:.6f}")
    print(f"  Result: {'Batch-dependent' if diff_standard > 1e-6 else 'Batch-invariant'}")
    
    # Test 2: Batch-invariant mode
    print("\n2. Batch-Invariant Mode:")

    
    with set_batch_invariant_mode(True):
        with torch.no_grad():
            out_single_inv = model(x[:1])
            out_batch_inv = model(x)
            out_batch_first_inv = out_batch_inv[:1]
    
    diff_invariant = (out_single_inv - out_batch_first_inv).abs().max().item()
    mean_diff_invariant = (out_single_inv - out_batch_first_inv).abs().mean().item()
    
    print(f"  Max difference: {diff_invariant:.6f}")
    print(f"  Mean difference: {mean_diff_invariant:.6f}")
    print(f"  Result: {'Batch-invariant' if diff_invariant < 1e-6 else 'Batch-dependent'}")
    
    return diff_standard, diff_invariant, out_single, out_single_inv


def test_mixed_precision_effects(out_fp32, out_fp32_inv):
    """Test how batch-dependency affects mixed precision inference."""
    

    print("MIXED PRECISION EXPERIMENT\n")

    
    # Convert outputs to different precisions
    precisions = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]
    
    print("Testing how initial FP32 differences affect lower precision:")
    print()
    
    for name, dtype in precisions:
        # Convert standard and invariant outputs to this precision
        out_standard_prec = out_fp32.to(dtype).to(torch.float32)
        out_invariant_prec = out_fp32_inv.to(dtype).to(torch.float32)
        
        # Compare precision loss
        loss_standard = (out_fp32 - out_standard_prec).abs().max().item()
        loss_invariant = (out_fp32_inv - out_invariant_prec).abs().max().item()
        
        # Compare difference between standard and invariant after precision conversion
        final_diff = (out_standard_prec - out_invariant_prec).abs().max().item()
        
        print(f"{name} precision:")
        print(f"  Precision loss (standard):  {loss_standard:.6f}")
        print(f"  Precision loss (invariant): {loss_invariant:.6f}")
        print(f"  Final difference:           {final_diff:.6f}")
        
        if final_diff < 1e-6:
            print(f"  → Differences eliminated by {name} precision")
        else:
            print(f"  → Differences survive {name} precision conversion")
        print()


def test_quantization_effects():
    """Test how batch-dependency affects quantized inference."""

    print("QUANTIZATION EXPERIMENT: Extreme Scale")
    
    torch.set_default_device('cuda')
    
    # Create a reasonable model with extreme input values  
    model = SimpleTransformerLayer(d_model=1536, num_heads=32).eval()
    
    # Create input with moderate size but EXTREME values
    batch_size = 32    # Reasonable batch
    seq_len = 128      # Reasonable sequence  
    d_model = 1536     # Bigger model
    
    total_elements = batch_size * seq_len * d_model
    x = torch.linspace(-10000, 10000, total_elements).reshape(batch_size, seq_len, d_model)  # VERY extreme values
    
    print(f"Model: Quantization transformer (d_model={d_model}, heads=32)")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}, value_range=±10000")
    print(f"Total elements: {total_elements/1e6:.1f}M")
    
    # Get FP32 baseline outputs
    with set_batch_invariant_mode(False):
        with torch.no_grad():
            out_single_fp32 = model(x[:1])
            out_batch_fp32 = model(x)[:1]
    
    with set_batch_invariant_mode(True):
        with torch.no_grad():
            out_single_inv_fp32 = model(x[:1])
    
    fp32_diff = (out_single_fp32 - out_batch_fp32).abs().max().item()
    
    print(f"\nFP32 baseline difference: {fp32_diff:.6f}")
    
    # Test different quantization schemes
    quantization_schemes = [
        ("INT8 symmetric", lambda t: torch.quantize_per_tensor(t, scale=0.1, zero_point=0, dtype=torch.qint8).dequantize()),
        ("INT8 asymmetric", lambda t: torch.quantize_per_tensor(t, scale=0.1, zero_point=64, dtype=torch.qint8).dequantize()),
        ("UINT8", lambda t: torch.quantize_per_tensor(t, scale=0.1, zero_point=128, dtype=torch.quint8).dequantize()),
    ]
    
    print("\nQuantization effects:")

    
    for scheme_name, quantize_fn in quantization_schemes:
        # Quantize the outputs
        out_single_quant = quantize_fn(out_single_fp32)
        out_batch_quant = quantize_fn(out_batch_fp32)
        out_inv_quant = quantize_fn(out_single_inv_fp32)
        
        # Measure differences after quantization
        quant_diff = (out_single_quant - out_batch_quant).abs().max().item()
        quant_vs_inv = (out_single_quant - out_inv_quant).abs().max().item()
        
        # Measure quantization noise
        quant_noise_single = (out_single_fp32 - out_single_quant).abs().max().item()
        quant_noise_batch = (out_batch_fp32 - out_batch_quant).abs().max().item()
        
        print(f"{scheme_name}:")
        print(f"  Quantization noise (single): {quant_noise_single:.6f}")
        print(f"  Quantization noise (batch):  {quant_noise_batch:.6f}")
        print(f"  Difference after quant:      {quant_diff:.6f}")
        print(f"  vs batch-invariant:          {quant_vs_inv:.6f}")
        
        if quant_diff < quant_noise_single:
            print(f"  → Batch differences hidden by quantization noise")
        else:
            print(f"  → Batch differences survive quantization")
        print()


def test_multi_layer_accumulation():
    """Test how differences accumulate across multiple transformer layers."""
    

    print("DEEP ACCUMULATION EXPERIMENT: 12-Layer Transformer\n")

    
    torch.set_default_device('cuda')
    
    # Create a reasonable depth with moderate size but extreme input values
    num_layers = 8   # Reasonable depth  
    layers = nn.ModuleList([
        SimpleTransformerLayer(d_model=768, num_heads=24) 
        for _ in range(num_layers)
    ]).eval()
    
    # Create input with moderate size but VERY extreme values
    batch_size = 32   # Moderate batch
    seq_len = 128     # Moderate sequence
    d_model = 768     # Moderate dimension
    
    total_elements = batch_size * seq_len * d_model
    x = torch.linspace(-5000, 5000, total_elements).reshape(batch_size, seq_len, d_model)  # Very extreme values
    
    print(f"Model: {num_layers} transformer layers (d_model={d_model}, heads=24)")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}, value_range=±5000")
    print(f"Total model size: ~{num_layers * d_model * d_model * 7 / 1e6:.0f}M parameters")
    print()
    
    # Track differences layer by layer
    differences_standard = []
    differences_invariant = []
    
    # Standard mode
    with set_batch_invariant_mode(False):
        with torch.no_grad():
            x_single = x[:1].clone()
            x_batch = x.clone()
            
            for i, layer in enumerate(layers):
                x_single = layer(x_single)
                x_batch = layer(x_batch)
                
                diff = (x_single - x_batch[:1]).abs().max().item()
                differences_standard.append(diff)
                print(f"Layer {i+1} difference (standard): {diff:.6f}")
    
    print()
    
    # Batch-invariant mode
    with set_batch_invariant_mode(True):
        with torch.no_grad():
            x_single_inv = x[:1].clone()
            x_batch_inv = x.clone()
            
            for i, layer in enumerate(layers):
                x_single_inv = layer(x_single_inv)
                x_batch_inv = layer(x_batch_inv)
                
                diff = (x_single_inv - x_batch_inv[:1]).abs().max().item()
                differences_invariant.append(diff)
                print(f"Layer {i+1} difference (invariant): {diff:.6f}")

    print("\nACCUMULATION ANALYSIS:")

    
    initial_diff = differences_standard[0]
    final_diff = differences_standard[-1]
    accumulation_factor = final_diff / (initial_diff + 1e-10)
    
    print(f"Initial difference (layer 1): {initial_diff:.6f}")
    print(f"Final difference (layer {num_layers}):   {final_diff:.6f}")
    print(f"Accumulation factor:          {accumulation_factor:.1f}x")
    
    if accumulation_factor > 2:
        print("Differences ACCUMULATE significantly across layers")
    else:
        print("Differences remain relatively stable")
    
    return differences_standard, differences_invariant


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This experiment requires GPU.")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Run experiments
    diff_std, diff_inv, out_std, out_inv = test_accumulation_in_transformer()
    test_mixed_precision_effects(out_std, out_inv)
    test_quantization_effects()
    differences_std, differences_inv = test_multi_layer_accumulation()


if __name__ == "__main__":
    main()