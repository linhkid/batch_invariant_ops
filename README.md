# Batch Invariant Ops

A companion library release to https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/. This library contains some batch-invariant kernels as well as an example of achieving deterministic vLLM inference.

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

## Attention Mechanism Experiment (NEW)

We've added batch-invariant attention implementations to address nondeterminism in transformer models. Attention is a critical component where batch-size dependent behavior can cascade through the entire model.

### Running the Attention Experiment

```bash
# Test both standard and batch-invariant attention
python test_attention_experiment.py --implementation both

# Test with different configurations
python test_attention_experiment.py --seq-len 1024 --num-heads 16 --num-iterations 200

# Test with a real transformer model (requires transformers library)
python test_attention_experiment.py --test-model --model-name gpt2
```

### Key Findings

Our experiments show that standard PyTorch attention produces different outputs depending on batch size:
- With batch_size=1: 1 unique output (reference)
- With batch_size=32: Often 50+ unique outputs from 100 iterations

The batch-invariant implementation achieves:
- **100% determinism** across all batch sizes
- ~20-40% performance overhead (unoptimized)
- Drop-in replacement for `F.scaled_dot_product_attention`

### Using Batch-Invariant Attention

```python
from batch_invariant_ops.batch_invariant_attention import (
    batch_invariant_scaled_dot_product_attention,
    register_batch_invariant_attention
)

# Option 1: Direct usage
output = batch_invariant_scaled_dot_product_attention(
    query, key, value, dropout_p=0.0, is_causal=False
)

# Option 2: Register globally (replaces F.scaled_dot_product_attention)
register_batch_invariant_attention()
# Now all calls to F.scaled_dot_product_attention use batch-invariant version
```

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
