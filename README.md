# Batch Invariant Ops

**Forked from [Thinking Machines Lab](https://github.com/thinking-machines-lab/batch_invariant_ops)** - a companion library to their excellent research on [defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## Extensions in This Fork

While the original library provides batch-invariant kernels for deterministic inference, this fork adds comprehensive experiments demonstrating:

- **Accumulation analysis** across transformer layers with precise error tracking
- **Mixed precision effects** on batch-dependency amplification  
- **Production quantization** testing with proper INT8 calibration
- **Gradient flow analysis** showing training implications
- **Realistic scale testing** up to 1.5B parameter models

Credit to Thinking Machines Lab for the foundational research and implementation.

## Overview

This library primarily leverages torch.Library to sub out existing PyTorch kernels with "batch-invariant" ones. This allows many existing PyTorch models to use the batch-invariant ops with low overhead and non-intrusive code changes.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from batch_invariant_ops import set_batch_invariant_mode

# Load a model (e.g., GPT-2)
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

# Prepare input
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt")

# Run inference with batch-invariant mode for deterministic results
with set_batch_invariant_mode():
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            temperature=0.0,  # Use greedy decoding for determinism
            do_sample=False
        )
    
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Testing Batch-Invariance

The following example shows how batch size can affect results in standard PyTorch:

```python
import torch
from batch_invariant_ops import set_batch_invariant_mode
torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

```

## Deterministic Inference in vLLM
`deterministic_vllm_inference.py` shows an proof of concept of validating that vLLM can be made deterministic with a minor upstream PR to use this library. Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream PR, there is only one unique sample.

## Comprehensive Accumulation Analysis (NEW)

This experiment provides a complete analysis of how `torch.mm` batch-dependencies propagate through transformer architectures, affecting mixed precision, quantization, and even gradient flow.

### Running the Experiments

```bash
# Comprehensive accumulation analysis with 5 distinct experiments
python test_accumulation_quantization_mp.py

# Basic torch.mm vs bmm/matmul comparison  
python test_mm_and_bmm.py

# Original simple batch invariance test
python test_batch_invariance.py
```

### What the Comprehensive Analysis Shows

1. **Precise Error Tracking**: Monitors intermediate activations to pinpoint exactly where batch-dependencies accumulate
2. **Mixed Precision Analysis**: Tests FP32, FP16, and BF16 precision effects on error propagation
3. **Production Quantization**: Uses PyTorch's official INT8 quantization workflow with proper calibration
4. **Deep Network Effects**: Tracks error accumulation across 8 transformer layers
5. **Training Implications**: Analyzes how batch-dependencies affect gradient flow during training

### Key Experimental Insights

- **Pre-FFN Measurement**: Tracks differences at the most critical accumulation point (before feed-forward networks)
- **Gradient Flow Testing**: Shows batch-dependencies affect both forward AND backward passes
- **Realistic Quantization**: Tests with properly calibrated INT8 models, not toy quantization
- **GELU Activation Effects**: Demonstrates how non-linear activations interact with batch-dependencies
- **Production Scale**: Tests with models up to 1.5B parameters to show real-world impact

### Why This Matters for Production

- **Training Stability**: Batch-dependencies can cause gradient inconsistencies during training
- **Inference Reliability**: Mixed precision deployment amplifies small numerical differences
- **Model Serving**: Different batch sizes in production can yield different results for the same input
- **A/B Testing**: Batch-dependent models make fair comparison impossible

## Supported Operations

### Matrix Operations
- `torch.mm()` - Matrix multiplication
- `torch.addmm()` - Matrix multiplication with bias addition

### Activation Functions
- `torch.log_softmax()` - Log-softmax activation

### Attention Operations (NEW)
- `scaled_dot_product_attention` - Batch-invariant attention mechanism
- Custom Triton kernels for deterministic softmax in attention

### Reduction Operations
- `torch.mean()` - Mean computation along specified dimensions
