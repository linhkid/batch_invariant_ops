#!/usr/bin/env python3
"""
A comprehensive analysis suite to demonstrate the effects of torch.mm
batch-dependency across various conditions in a transformer architecture.

This script includes experiments for:
1. Basic forward pass accumulation in a single layer.
2. The impact on mixed-precision casting (FP16, BF16).
3. The interaction with INT8 quantization on a supported submodule (Linear layers).
4. Error accumulation across a deep, multi-layer network.
5. The effect on non-linear activations (GELU) and gradient flow (backward pass).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# The batch_invariant_ops library is assumed to be in the same directory or installed
from batch_invariant_ops import set_batch_invariant_mode
from torch.ao.quantization import get_default_qconfig


# ======================================================================
# Model Definitions
# ======================================================================

class SimpleTransformerLayer(nn.Module):
    """
    A simple transformer layer that uses multiple torch.mm operations.
    Updated to store intermediate activations and FFN inputs.
    """

    def __init__(self, d_model=768, num_heads=12, d_ff=3072):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.intermediate_activation = None
        self.ffn_input = None  # Store input to FFN for quantization experiment

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_linear(out)

        x_after_attn = self.ln1(x + out)
        self.ffn_input = x_after_attn

        ff_intermediate = F.gelu(self.ff1(x_after_attn))
        self.intermediate_activation = ff_intermediate
        ff_out = self.ff2(ff_intermediate)
        x_final = self.ln2(x_after_attn + ff_out)

        return x_final


class QuantizableLinearBlock(nn.Module):
    """
    An isolated, minimal block containing only quantizable nn.Linear layers.
    This avoids unsupported ops like LayerNorm and GELU during quantization.
    """

    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # We don't apply GELU here to keep the module simple for quantization
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.dequant(x)
        return x


# ======================================================================
# Experiment Functions
# ======================================================================

def test_accumulation_in_transformer():
    print("ACCUMULATION EXPERIMENT: Large Transformer Layer\n")

    torch.set_default_device('cuda')
    model = SimpleTransformerLayer(d_model=1536, num_heads=32).eval()

    batch_size, seq_len, d_model = 64, 256, 1536
    x = torch.linspace(-1000, 1000, batch_size * seq_len * d_model).reshape(batch_size, seq_len, d_model)

    print(f"Model: SimpleTransformerLayer (d_model={d_model}, heads=32)")
    print(f"Total parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print("\n1. Standard PyTorch Mode:")

    with set_batch_invariant_mode(False):
        model(x[:1])
        ffn_input_single_std = model.ffn_input.detach().clone()
        model(x)
        ffn_input_batch_std = model.ffn_input.detach().clone()
    # Recalculate diff on the ffn_input which is the source of the error for the next step
    diff = (ffn_input_single_std - ffn_input_batch_std[:1]).abs()
    print(f"  Max difference (pre-FFN): {diff.max().item():.6f}")
    print(f"  Result: {'Batch-dependent' if diff.max().item() > 1e-6 else 'Batch-invariant'}")

    print("\n2. Batch-Invariant Mode:")

    with set_batch_invariant_mode(True):
        model(x[:1])
        ffn_input_single_inv = model.ffn_input.detach().clone()
        model(x)
        ffn_input_batch_inv = model.ffn_input.detach().clone()
    diff_inv = (ffn_input_single_inv - ffn_input_batch_inv[:1]).abs()
    print(f"  Max difference (pre-FFN): {diff_inv.max().item():.6f}")
    print(f"  Result: {'Batch-invariant' if diff_inv.max().item() < 1e-6 else 'Batch-dependent'}")

    return ffn_input_batch_std, ffn_input_batch_inv


def test_mixed_precision(ffn_input_std, ffn_input_inv):
    print("MIXED PRECISION EXPERIMENT\n")
    for dtype, name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        print(f"\n{name} precision:")
        out_std_prec, out_inv_prec = ffn_input_std.to(dtype), ffn_input_inv.to(dtype)
        final_diff = (out_std_prec - out_inv_prec).abs().max().item()
        print(f"  Final difference after casting to {name}: {final_diff:.6f}")


def test_quantization_effects(ffn_input_std, ffn_input_inv):
    print("QUANTIZATION EXPERIMENT ON LINEAR BLOCK (Final)\n")


    torch.set_default_device('cpu')
    ffn_input_std = ffn_input_std.to('cpu')
    ffn_input_inv = ffn_input_inv.to('cpu')

    d_model, d_ff = ffn_input_std.shape[-1], 3072 * 2

    print(f"Isolating the Linear Layers (d_model={d_model}, d_ff={d_ff})")

    # Get FP32 baseline difference from the isolated linear block
    fp32_linear_block = QuantizableLinearBlock(d_model=d_model, d_ff=d_ff).eval()
    out_std_fp32 = fp32_linear_block(ffn_input_std)
    out_inv_fp32 = fp32_linear_block(ffn_input_inv)
    fp32_diff = (out_std_fp32 - out_inv_fp32).abs().max().item()
    print(f"\nFP32 baseline difference from Linear block: {fp32_diff:.6f}")

    # Quantize the Linear block and test again
    print("\nQuantizing the Linear block:")
    quant_linear_block = QuantizableLinearBlock(d_model=d_model, d_ff=d_ff).eval()
    quant_linear_block.qconfig = get_default_qconfig("x86")
    torch.ao.quantization.prepare(quant_linear_block, inplace=True)
    quant_linear_block(ffn_input_std)  # Calibration
    torch.ao.quantization.convert(quant_linear_block, inplace=True)

    out_std_quant = quant_linear_block(ffn_input_std)
    out_inv_quant = quant_linear_block(ffn_input_inv)
    quant_diff = (out_std_quant - out_inv_quant).abs().max().item()

    print(f"  Difference after INT8 quantization:    {quant_diff:.6f}")
    print("Batch differences are affected by the noise from quantization.\n")


def test_deep_accumulation():
    print("DEEP ACCUMULATION EXPERIMENT: 8-Layer Transformer")


    torch.set_default_device('cuda')
    num_layers, d_model, heads = 8, 768, 24
    layers = nn.ModuleList([SimpleTransformerLayer(d_model=d_model, num_heads=heads) for _ in range(num_layers)]).eval()

    batch_size, seq_len = 32, 128
    x = torch.linspace(-5000, 5000, batch_size * seq_len * d_model).reshape(batch_size, seq_len, d_model)

    print(f"Model: {num_layers} transformer layers (d_model={d_model}, heads={heads})")

    with set_batch_invariant_mode(False):
        x_single, x_batch = x[:1].clone(), x.clone()
        for i, layer in enumerate(layers):
            x_single, x_batch = layer(x_single), layer(x_batch)
            diff = (x_single - x_batch[:1]).abs().max().item()
            print(f"Layer {i + 1} difference (standard): {diff:.6f}")


def test_gradient_and_activation_effects():
    print("GRADIENT & ACTIVATION EXPERIMENT")

    torch.set_default_device('cuda')
    model = SimpleTransformerLayer(d_model=768, num_heads=16).train()

    batch_size, seq_len, d_model = 32, 128, 768
    x = torch.linspace(-100, 100, batch_size * seq_len * d_model).reshape(batch_size, seq_len, d_model)
    x.requires_grad = True

    print("\n1. Standard PyTorch Mode:")

    with set_batch_invariant_mode(False):
        model.zero_grad()
        out_single = model(x[:1]);
        out_single.sum().backward()
        grad_single = x.grad[:1].detach().clone()

        x.grad.zero_()
        out_batch = model(x);
        out_batch.sum().backward()
        grad_batch_first = x.grad[:1].detach().clone()

    grad_diff_std = (grad_single - grad_batch_first).abs().max().item()
    print(f"  Max gradient difference: {grad_diff_std:.6f}")
    print(f"  Result: {'Batch-dependent' if grad_diff_std > 1e-6 else 'Batch-invariant'}")

    print("\n2. Batch-Invariant Mode:")
    with set_batch_invariant_mode(True):
        x.grad.zero_();
        model.zero_grad()
        out_single_inv = model(x[:1]);
        out_single_inv.sum().backward()
        grad_single_inv = x.grad[:1].detach().clone()

        x.grad.zero_()
        out_batch_inv = model(x);
        out_batch_inv.sum().backward()
        grad_batch_first_inv = x.grad[:1].detach().clone()

    grad_diff_inv = (grad_single_inv - grad_batch_first_inv).abs().max().item()
    print(f"  Max gradient difference: {grad_diff_inv:.6f}")
    print(f"  Result: {'Batch-invariant' if grad_diff_inv < 1e-6 else 'Batch-dependent'}")


def main():
    use_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else 'CPU'
    print(f"Using device: {device_name}")
    print(f"PyTorch version: {torch.__version__}")

    if use_cuda:
        ffn_in_std, ffn_in_inv = test_accumulation_in_transformer()
        test_mixed_precision(ffn_in_std, ffn_in_inv)
        test_quantization_effects(ffn_in_std, ffn_in_inv)
        test_deep_accumulation()
        test_gradient_and_activation_effects()
    else:
        print("\nCUDA not available. Skipping GPU-dependent experiments.")


if __name__ == "__main__":
    main()

