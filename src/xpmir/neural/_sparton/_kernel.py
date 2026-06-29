"""Sparton Triton kernels and SpartonHead module.

This file requires CUDA and triton. It is guarded so that it can be
imported (as an empty module) even when triton is not installed, which
prevents the documentation scanner from reporting import errors.
"""

# ruff: noqa: E402

import torch

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if not _HAS_TRITON:
    # Module is importable but empty when triton is not available.
    pass
elif True:
    # All kernel code lives inside this branch so that triton decorators
    # are never evaluated when triton is missing.

    import math
    from typing import Tuple

    import torch.nn as nn
    import triton
    import triton.language as tl

    # -------------------------------------------------------------------
    # Autotuning configs
    # -------------------------------------------------------------------

    def _get_fast_forward_configs():
        return [
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 64}, num_warps=4, num_stages=5),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 64, "BLOCK_S": 512}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 512, "BLOCK_S": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 256, "BLOCK_S": 256}, num_warps=8, num_stages=5),
            triton.Config({"BLOCK_C": 128, "BLOCK_S": 512}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_C": 512, "BLOCK_S": 64}, num_warps=16, num_stages=3),
            triton.Config({"BLOCK_C": 512, "BLOCK_S": 128}, num_warps=16, num_stages=4),
            triton.Config({"BLOCK_C": 1024, "BLOCK_S": 32}, num_warps=16, num_stages=2),
        ]

    def _get_fast_bwd_configs():
        return [
            triton.Config(
                {"BLOCK_B": 16, "BLOCK_V": 16, "BLOCK_D": 32, "GROUP_SIZE": 8},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_B": 32, "BLOCK_V": 16, "BLOCK_D": 32, "GROUP_SIZE": 8},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_B": 32, "BLOCK_V": 32, "BLOCK_D": 64, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_B": 64, "BLOCK_V": 32, "BLOCK_D": 64, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_B": 32, "BLOCK_V": 64, "BLOCK_D": 64, "GROUP_SIZE": 8},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_B": 16, "BLOCK_V": 32, "BLOCK_D": 128, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_B": 32, "BLOCK_V": 32, "BLOCK_D": 128, "GROUP_SIZE": 8},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_B": 32, "BLOCK_V": 32, "BLOCK_D": 64, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=16,
            ),
            triton.Config(
                {"BLOCK_B": 16, "BLOCK_V": 64, "BLOCK_D": 64, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=16,
            ),
            triton.Config(
                {"BLOCK_B": 128, "BLOCK_V": 16, "BLOCK_D": 16, "GROUP_SIZE": 8},
                num_stages=3,
                num_warps=8,
            ),
        ]

    # -------------------------------------------------------------------
    # Forward kernels
    # -------------------------------------------------------------------

    @triton.autotune(
        configs=_get_fast_forward_configs(), key=["S", "C"], cache_results=True
    )
    @triton.jit
    def _reduce_seq_max_log1p_relu_kernel_with_indices(
        logits_ptr,
        mask_ptr,
        out_vals_ptr,
        out_idx_ptr,
        B: tl.constexpr,
        S: tl.constexpr,
        C: tl.constexpr,
        stride_lb,
        stride_ls,
        stride_lc,
        stride_mb,
        stride_ms,
        stride_ob,
        stride_oc,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c_block = tl.program_id(1)
        offs_c = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        NEG_INF = 0.0
        max_vals = tl.full((BLOCK_C,), NEG_INF, dtype=tl.float32)
        max_idx = tl.full((BLOCK_C,), 0, dtype=tl.int64)
        s_range = tl.arange(0, BLOCK_S)
        for s_start in range(0, S, BLOCK_S):
            offs_s = s_start + s_range
            mask_s = offs_s < S
            offs_s_b = offs_s[:, None]
            offs_c_b = offs_c[None, :]
            logits_ptrs = (
                logits_ptr
                + pid_b * stride_lb
                + offs_s_b * stride_ls
                + offs_c_b * stride_lc
            )
            tile_mask = mask_s[:, None] & mask_c[None, :]
            logits_tile = tl.load(logits_ptrs, mask=tile_mask, other=0.0)
            mask_ptrs = mask_ptr + pid_b * stride_mb + offs_s * stride_ms
            mask_vals = tl.load(mask_ptrs, mask=mask_s, other=0)
            logits_tile = logits_tile * mask_vals[:, None]
            logits_tile = tl.where(tile_mask, logits_tile, NEG_INF)
            tile_max_vals = tl.max(logits_tile, axis=0)
            tile_argmax = tl.argmax(logits_tile, axis=0)
            better = tile_max_vals > max_vals
            max_vals = tl.where(better, tile_max_vals, max_vals)
            max_idx = tl.where(better, s_start + tile_argmax, max_idx)
        zero = 0.0
        relu_vals = tl.where(max_vals > zero, max_vals, zero)
        log_vals = tl.log(1.0 + relu_vals)
        out_vals_ptrs = out_vals_ptr + pid_b * stride_ob + offs_c * stride_oc
        out_idx_ptrs = out_idx_ptr + pid_b * stride_ob + offs_c * stride_oc
        tl.store(out_vals_ptrs, log_vals, mask=mask_c)
        tl.store(out_idx_ptrs, max_idx, mask=mask_c)

    @triton.autotune(configs=_get_fast_forward_configs(), key=["S", "C"])
    @triton.jit
    def _reduce_seq_max_log1p_relu_kernel(
        logits_ptr,
        mask_ptr,
        out_vals_ptr,
        B: tl.constexpr,
        S: tl.constexpr,
        C: tl.constexpr,
        stride_lb,
        stride_ls,
        stride_lc,
        stride_mb,
        stride_ms,
        stride_ob,
        stride_oc,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c_block = tl.program_id(1)
        offs_c = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        NEG_INF = 0
        max_vals = tl.full((BLOCK_C,), NEG_INF, dtype=tl.float32)
        s_range = tl.arange(0, BLOCK_S)
        for s_start in range(0, S, BLOCK_S):
            offs_s = s_start + s_range
            mask_s = offs_s < S
            offs_s_b = offs_s[:, None]
            offs_c_b = offs_c[None, :]
            logits_ptrs = (
                logits_ptr
                + pid_b * stride_lb
                + offs_s_b * stride_ls
                + offs_c_b * stride_lc
            )
            tile_mask = mask_s[:, None] & mask_c[None, :]
            logits_tile = tl.load(logits_ptrs, mask=tile_mask, other=0.0)
            mask_ptrs = mask_ptr + pid_b * stride_mb + offs_s * stride_ms
            mask_vals = tl.load(mask_ptrs, mask=mask_s, other=0)
            logits_tile = logits_tile * mask_vals[:, None]
            logits_tile = tl.where(tile_mask, logits_tile, NEG_INF)
            tile_max_vals = tl.max(logits_tile, axis=0)
            better = tile_max_vals > max_vals
            max_vals = tl.where(better, tile_max_vals, max_vals)
        zero = 0.0
        relu_vals = tl.where(max_vals > zero, max_vals, zero)
        log_vals = tl.log(1.0 + relu_vals)
        out_vals_ptrs = out_vals_ptr + pid_b * stride_ob + offs_c * stride_oc
        tl.store(out_vals_ptrs, log_vals, mask=mask_c)

    # -------------------------------------------------------------------
    # Backward kernel
    # -------------------------------------------------------------------

    @triton.autotune(
        configs=_get_fast_bwd_configs(),
        key=["batch_size", "vocab_size", "hidden_dim"],
        reset_to_zero=["hidden_grad_ptr", "embed_grad_ptr", "bias_grad_ptr"],
    )
    @triton.jit
    def _fused_sparton_bwd_kernel_with_bias(
        grad_out_ptr,
        max_scores_ptr,
        max_idx_ptr,
        hidden_ptr,
        embed_ptr,
        hidden_grad_ptr,
        embed_grad_ptr,
        bias_grad_ptr,
        batch_size,
        seq_len,
        hidden_dim: tl.constexpr,
        vocab_size: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        v_block_id = tl.program_id(0)
        b_block_id = tl.program_id(1)
        num_v_blocks = tl.num_programs(0)
        num_b_blocks = tl.num_programs(1)
        b_block_id, v_block_id = tl.swizzle2d(
            b_block_id, v_block_id, num_b_blocks, num_v_blocks, GROUP_SIZE
        )
        start_b = b_block_id * BLOCK_B
        start_v = v_block_id * BLOCK_V
        offs_v = start_v + tl.arange(0, BLOCK_V).to(tl.int64)
        offs_b = start_b + tl.arange(0, BLOCK_B).to(tl.int64)
        offs_v = tl.max_contiguous(tl.multiple_of(offs_v, BLOCK_V), BLOCK_V)
        offs_b = tl.max_contiguous(tl.multiple_of(offs_b, BLOCK_B), BLOCK_B)
        offs_d = tl.arange(0, BLOCK_D)
        b_mask = offs_b < batch_size
        bv_mask = (offs_b[:, None] < batch_size) & (offs_v[None, :] < vocab_size)
        bv_offs = offs_b[:, None] * vocab_size + offs_v[None, :]
        block_max_logits = tl.load(
            max_scores_ptr + bv_offs, mask=bv_mask, other=0.0
        ).to(tl.float32)
        if tl.sum(block_max_logits) == 0:
            return
        grad_out = tl.load(grad_out_ptr + bv_offs, mask=bv_mask, other=0.0).to(
            tl.float32
        )
        block_max_idx = tl.load(max_idx_ptr + bv_offs, mask=bv_mask, other=0)
        relu_log1p_grad = tl.where(
            block_max_logits > 0, grad_out * tl.exp(-block_max_logits), 0.0
        ).to(tl.float32)
        mask_v = offs_v < vocab_size
        tl.atomic_add(
            bias_grad_ptr + offs_v,
            tl.sum(relu_log1p_grad, axis=0),
            mask=mask_v,
            sem="relaxed",
        )
        e_ptrs = embed_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]
        h_ptrs = (
            hidden_ptr
            + offs_b[:, None, None] * seq_len * hidden_dim
            + block_max_idx[:, :, None] * hidden_dim
            + offs_d[None, None, :]
        )
        e_grad_ptrs = embed_grad_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]
        h_grad_ptrs = (
            hidden_grad_ptr
            + offs_b[:, None, None] * seq_len * hidden_dim
            + block_max_idx[:, :, None] * hidden_dim
            + offs_d[None, None, :]
        )
        valid_logit_mask = block_max_logits > 0
        for start_d in range(0, hidden_dim, BLOCK_D):
            mask_d = start_d + offs_d < hidden_dim
            hidden_state = tl.load(
                h_ptrs,
                mask=b_mask[:, None, None]
                & mask_d[None, None, :]
                & valid_logit_mask[:, :, None],
                other=0.0,
            ).to(tl.float32)
            embed_grad_update = tl.sum(
                hidden_state * relu_log1p_grad[:, :, None], axis=0
            )
            tl.atomic_add(
                e_grad_ptrs,
                embed_grad_update,
                mask=mask_v[:, None] & mask_d[None, :],
                sem="relaxed",
            )
            embed = tl.load(
                e_ptrs,
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            hidden_grad_update = (embed[None, :, :] * relu_log1p_grad[:, :, None]).to(
                tl.float32
            )
            tl.atomic_add(
                h_grad_ptrs,
                hidden_grad_update,
                mask=bv_mask[:, :, None]
                & mask_d[None, None, :]
                & valid_logit_mask[:, :, None],
                sem="relaxed",
            )
            e_ptrs += BLOCK_D
            h_ptrs += BLOCK_D
            e_grad_ptrs += BLOCK_D
            h_grad_ptrs += BLOCK_D

    # -------------------------------------------------------------------
    # Python wrappers
    # -------------------------------------------------------------------

    @torch.compile
    def _matmul(a, b):
        return a @ b.T

    @torch.compile
    def _matmul_bias(a, b, c):
        return a @ b.T + c

    def _v_tile_from_bs(B, S, V, temp_mib=64, align=256, min_tiles=512):
        temp_bytes = temp_mib * 1024 * 1024
        C = temp_bytes // (2 * B * S)
        C = max(min_tiles, min(V, (C // align) * align))
        return C

    def _reduce_seq_max_log1p_relu_with_indices(logits, mask):
        assert logits.ndim == 3 and mask.ndim == 2
        B, S, C = logits.shape
        assert mask.shape == (B, S)
        logits = logits.contiguous()
        mask = mask.contiguous()
        vals = torch.empty((B, C), device=logits.device, dtype=logits.dtype)
        idxs = torch.empty((B, C), device=logits.device, dtype=torch.int64)

        def grid(meta):
            return (B, triton.cdiv(C, meta["BLOCK_C"]))

        _reduce_seq_max_log1p_relu_kernel_with_indices[grid](
            logits,
            mask,
            vals,
            idxs,
            B,
            S,
            C,
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            mask.stride(0),
            mask.stride(1),
            vals.stride(0),
            vals.stride(1),
        )
        return vals, idxs

    def _reduce_seq_max_log1p_relu(logits, mask):
        assert logits.ndim == 3 and mask.ndim == 2
        B, S, C = logits.shape
        assert mask.shape == (B, S)
        logits = logits.contiguous()
        mask = mask.contiguous()
        vals = torch.empty((B, C), device=logits.device, dtype=logits.dtype)

        def grid(meta):
            return (B, triton.cdiv(C, meta["BLOCK_C"]))

        _reduce_seq_max_log1p_relu_kernel[grid](
            logits,
            mask,
            vals,
            B,
            S,
            C,
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            mask.stride(0),
            mask.stride(1),
            vals.stride(0),
            vals.stride(1),
        )
        return vals

    def _fused_sparton_fwd_with_indices(hidden, embed, bias, mask):
        B, S, D = hidden.shape
        V, _ = embed.shape
        tile_size = _v_tile_from_bs(B, S, V)
        sparse_reps = torch.empty((B, V), device=hidden.device, dtype=hidden.dtype)
        max_indices = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
        for i in range(0, V, tile_size):
            tile_embed = embed[i : i + tile_size]
            C = tile_embed.shape[0]
            if bias is not None:
                tile_logits = _matmul_bias(hidden, tile_embed, bias[i : i + tile_size])
            else:
                tile_logits = _matmul(hidden, tile_embed)
            tile_vals, tile_idx = _reduce_seq_max_log1p_relu_with_indices(
                tile_logits, mask
            )
            sparse_reps[:, i : i + C] = tile_vals
            max_indices[:, i : i + C] = tile_idx
        return sparse_reps, max_indices

    def _fused_sparton_fwd(hidden, embed, bias, mask):
        B, S, D = hidden.shape
        V, _ = embed.shape
        sparse_reps_tiles = []
        tile_size = _v_tile_from_bs(B, S, V)
        for i in range(0, V, tile_size):
            tile_embed = embed[i : i + tile_size]
            if bias is not None:
                tile_logits = _matmul_bias(hidden, tile_embed, bias[i : i + tile_size])
            else:
                tile_logits = _matmul(hidden, tile_embed)
            tile_vals = _reduce_seq_max_log1p_relu(tile_logits, mask)
            sparse_reps_tiles.append(tile_vals)
        return torch.cat(sparse_reps_tiles, dim=1)

    def _fused_sparton_bwd_with_bias(
        grad_output,
        max_scores,
        max_idx,
        hidden,
        embed,
        hidden_grad,
        embed_grad,
        bias_grad,
    ):
        B, S, D = hidden.shape
        V, D_e = embed.shape
        assert D == D_e
        assert max_scores.shape == (B, V) and max_idx.shape == (B, V)

        def grid(meta):
            return (
                triton.cdiv(V, meta["BLOCK_V"]),
                triton.cdiv(B, meta["BLOCK_B"]),
            )

        _fused_sparton_bwd_kernel_with_bias[grid](
            grad_out_ptr=grad_output,
            max_scores_ptr=max_scores,
            max_idx_ptr=max_idx,
            hidden_ptr=hidden,
            embed_ptr=embed,
            hidden_grad_ptr=hidden_grad,
            embed_grad_ptr=embed_grad,
            bias_grad_ptr=bias_grad,
            batch_size=B,
            seq_len=S,
            hidden_dim=D,
            vocab_size=V,
        )
        return hidden_grad, embed_grad, bias_grad, None

    # -------------------------------------------------------------------
    # Custom op registration (for torch.compile + autograd)
    # -------------------------------------------------------------------

    @torch.library.custom_op("sparton::fused_sparton_fwd", mutates_args=())
    def _fused_sparton_fwd_op(
        hidden: torch.Tensor,
        embed: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hidden.is_cuda, "sparton::fused_sparton_fwd only supports CUDA"
        scores, idx = _fused_sparton_fwd_with_indices(hidden, embed, bias, mask)
        return scores, idx

    @_fused_sparton_fwd_op.register_fake
    def _fused_sparton_fwd_fake(hidden, embed, bias, mask):
        B, S, D = hidden.shape
        V, _ = embed.shape
        return hidden.new_empty((B, V)), torch.empty(
            (B, V), device=hidden.device, dtype=torch.int64
        )

    @torch.library.custom_op("sparton::fused_sparton_bwd", mutates_args=())
    def _fused_sparton_bwd_op(
        grad_out: torch.Tensor,
        max_scores: torch.Tensor,
        max_idx: torch.Tensor,
        hidden: torch.Tensor,
        embed: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert grad_out.is_cuda, "sparton::fused_sparton_bwd only supports CUDA"
        grad_out = grad_out.contiguous()
        hidden_grad = torch.zeros_like(hidden, dtype=torch.float32)
        embed_grad = torch.zeros_like(embed, dtype=torch.float32)
        bias_grad = torch.zeros_like(bias, dtype=torch.float32)
        hidden_grad, embed_grad, bias_grad, _ = _fused_sparton_bwd_with_bias(
            grad_out,
            max_scores,
            max_idx,
            hidden,
            embed,
            hidden_grad,
            embed_grad,
            bias_grad,
        )
        return hidden_grad, embed_grad, bias_grad

    @_fused_sparton_bwd_op.register_fake
    def _fused_sparton_bwd_fake(
        grad_out, max_scores, max_idx, hidden, embed, bias, mask
    ):
        return (
            torch.empty_like(hidden, dtype=torch.float32),
            torch.empty_like(embed, dtype=torch.float32),
            torch.empty_like(bias, dtype=torch.float32),
        )

    def _setup_context(ctx, inputs, output):
        hidden, embed, bias, mask = inputs
        scores, idx = output
        ctx.save_for_backward(scores, idx, hidden, embed, bias, mask)

    def _backward(ctx, grad_scores, grad_idx):
        scores, idx, hidden, embed, bias, mask = ctx.saved_tensors
        hidden_g, embed_g, bias_g = _fused_sparton_bwd_op(
            grad_scores, scores, idx, hidden, embed, bias, mask
        )
        return hidden_g, embed_g, bias_g, None

    _fused_sparton_fwd_op.register_autograd(_backward, setup_context=_setup_context)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    class SpartonHead(nn.Module):
        """Fused SPLADE projection head using Triton kernels.

        Computes ``log1p(relu(max_s(hidden @ weight.T + bias) * mask))``
        in a memory-efficient tiled manner without materializing the full
        ``[B, S, V]`` logits tensor.
        """

        def __init__(self, vocab_size: int, hidden_dim: int, use_bias: bool = False):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_dim))
            if use_bias:
                self.bias = nn.Parameter(torch.empty(vocab_size))
            else:
                self.register_parameter("bias", None)
            self._init_parameters()

        def tie_weights(self, decoder: nn.Linear):
            """Tie weight and bias to an existing decoder layer."""
            self.weight = decoder.weight
            self.bias = decoder.bias

        def _init_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
            scores, _idx = _fused_sparton_fwd_op(
                hidden_states, self.weight, self.bias, attention_mask
            )
            return scores
