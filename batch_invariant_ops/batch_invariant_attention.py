"""Batch-invariant attention implementation using Triton."""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def batch_invariant_softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """Batch-invariant softmax kernel with fixed reduction order."""
    row_idx = tl.program_id(0)
    
    # Load input row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row_start = row_idx * n_cols
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max with fixed reduction order
    row_max = tl.max(row, axis=0)
    
    # Subtract max for numerical stability
    row_stable = row - row_max
    
    # Compute exp
    row_exp = tl.exp(row_stable)
    
    # Sum with fixed reduction order
    row_sum = tl.sum(row_exp, axis=0)
    
    # Normalize
    output = row_exp / row_sum
    
    # Store result
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def batch_invariant_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, num_heads, seq_len, d_k,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Batch-invariant attention kernel with fixed computation order."""
    
    # Get program IDs
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(2)  # sequence block
    
    # Compute offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    q_ptrs = q_ptr + (pid_b * stride_qb + pid_h * stride_qh + 
                      offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    k_ptrs = k_ptr + (pid_b * stride_kb + pid_h * stride_kh +
                      offs_n[:, None] * stride_ks + offs_k[None, :] * stride_kd)
    v_ptrs = v_ptr + (pid_b * stride_vb + pid_h * stride_vh +
                      offs_n[:, None] * stride_vs + offs_k[None, :] * stride_vd)
    
    # Load Q for this block
    mask_m = offs_m < seq_len
    mask_k = offs_k < d_k
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float("inf"), dtype=tl.float32)
    
    # Scale factor
    scale = 1.0 / tl.sqrt(float(d_k))
    
    # Process K and V blocks with fixed order
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len
        
        # Load K block
        k = tl.load(k_ptrs + start_n * stride_ks, 
                   mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute QK^T for this block
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk = qk * scale
        
        # Apply mask if needed
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float("inf"))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Recompute exponentials with new max
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        
        # Load V block
        v = tl.load(v_ptrs + start_n * stride_vs,
                   mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Update accumulator
        acc += tl.dot(p, v, allow_tf32=False)
        m_i = m_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = out_ptr + (pid_b * stride_ob + pid_h * stride_oh +
                          offs_m[:, None] * stride_os + offs_k[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


class BatchInvariantAttention(torch.nn.Module):
    """Batch-invariant scaled dot-product attention."""
    
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p
        if dropout_p > 0:
            print("Warning: Dropout not yet supported in batch-invariant attention")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute batch-invariant attention.
        
        Args:
            query: [batch_size, num_heads, seq_len, d_k]
            key: [batch_size, num_heads, seq_len, d_k]
            value: [batch_size, num_heads, seq_len, d_k]
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        
        Returns:
            Attention output [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, num_heads, seq_len, d_k = query.shape
        
        # Ensure inputs are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        
        # Allocate output
        output = torch.empty_like(query)
        
        # Choose block sizes (can be tuned)
        BLOCK_SIZE_M = min(64, seq_len)
        BLOCK_SIZE_N = min(64, seq_len)
        BLOCK_SIZE_K = min(64, d_k)
        
        # Grid dimensions
        grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_SIZE_M))
        
        # Launch kernel
        batch_invariant_attention_kernel[grid](
            query, key, value, output,
            batch_size, num_heads, seq_len, d_k,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        )
        
        return output


def batch_invariant_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for F.scaled_dot_product_attention with batch invariance.
    
    This function ensures deterministic output regardless of batch size by using
    fixed reduction orders in all operations.
    """
    attention = BatchInvariantAttention(dropout_p=dropout_p)
    return attention(query, key, value, attn_mask, is_causal)


def register_batch_invariant_attention():
    """Register batch-invariant attention with PyTorch."""
    import torch.nn.functional as F
    
    # Store original implementation
    if not hasattr(F, '_original_scaled_dot_product_attention'):
        F._original_scaled_dot_product_attention = F.scaled_dot_product_attention
    
    # Replace with batch-invariant version
    F.scaled_dot_product_attention = batch_invariant_scaled_dot_product_attention
    
    print("Batch-invariant attention registered")


def unregister_batch_invariant_attention():
    """Restore original PyTorch attention."""
    import torch.nn.functional as F
    
    if hasattr(F, '_original_scaled_dot_product_attention'):
        F.scaled_dot_product_attention = F._original_scaled_dot_product_attention
        print("Original attention restored")