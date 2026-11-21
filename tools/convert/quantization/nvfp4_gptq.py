"""G16 NVFP4 GPTQ with represented E4M3 scales and a shared parent divisor.

Error compensation follows the GPTQ algorithm used in mkyt/qwen3-27b-5090's
scripts/quantize_proj.py. Unlike checkpoint import, input must be original BF16
weights. The caller bounds row memory and supplies a measured input Gram matrix.

Portions derived from that implementation are licensed under the MIT License:
Copyright (c) 2026 Masahiro Kiyota
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import struct

import torch

from .groupwise import QuantizedMatrix


def inverse_factor(gram: torch.Tensor, damping: float = 0.01):
    """Return an upper inverse-Cholesky factor and dead-column mask without mutation."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] % 16:
        raise ValueError("GPTQ requires a square Gram matrix with K divisible by 16")
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("GPTQ damping must be positive and finite")
    h = gram.float().clone()
    if not bool(torch.isfinite(h).all()) or not torch.allclose(h, h.T, rtol=1e-5, atol=1e-6):
        raise ValueError("GPTQ Gram matrix must be finite and symmetric")
    diagonal = h.diagonal()
    if bool((diagonal < 0).any()) or not bool((diagonal > 0).any()):
        raise ValueError("GPTQ requires nonnegative, nonempty measured input energy")
    dead = diagonal == 0
    h[dead, dead] = 1
    h.diagonal().add_(damping * h.diagonal().mean())
    factor = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    return factor, dead


def _codes(values: torch.Tensor) -> torch.Tensor:
    # argmin's first hit gives round-to-nearest-even E2M1 magnitude ties.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=values.device)
    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=values.device)
    index = order[(values.abs().unsqueeze(-1) - magnitudes[order]).abs().argmin(-1)]
    return (index | (torch.signbit(values).to(torch.int64) << 3)).to(torch.uint8)


def _decode(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=codes.device)
    return table[(codes & 7).long()] * torch.where(codes & 8 != 0, -1., 1.)


def quantize_gptq(weight: torch.Tensor, factor: torch.Tensor, dead: torch.Tensor,
                  divisor: bytes, *, block_size: int = 128, search: int = 8) -> QuantizedMatrix:
    """Encode complete rows with sequential compensated columns and G16 scale search.

    factor/dead are reusable across row chunks of projections sharing one input.
    Packed codes are captured at quantization time, not re-rounded after feedback.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError("GPTQ source must be an original BF16 [N,K] matrix")
    n, k = weight.shape
    if factor.shape != (k, k) or dead.shape != (k,):
        raise ValueError("GPTQ calibration geometry differs from the weights")
    if block_size <= 0 or block_size % 16 or search < 0:
        raise ValueError("GPTQ block size must be positive G16-aligned; search must be nonnegative")
    d, = struct.unpack("<f", divisor)
    if not math.isfinite(d) or d <= 0:
        raise ValueError("NVFP4 divisor must be positive finite FP32")
    w = weight.float().clone()
    if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(factor).all()):
        raise ValueError("GPTQ inputs must be finite")
    if not bool((factor.diagonal() > 0).all()):
        raise ValueError("GPTQ factor must have positive diagonal")
    w[:, dead] = 0
    codes = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // 16), dtype=torch.uint8, device=w.device)
    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        local = w[:, start:end].clone()
        errors = torch.empty_like(local)
        hi = factor[start:end, start:end]
        for column in range(end - start):
            if column % 16 == 0:
                group = local[:, column:column + 16]
                rounded = (group.abs().amax(1) * d / 6).clamp(max=448).to(torch.float8_e4m3fn)
                center = rounded.view(torch.uint8).to(torch.int16)
                best_error = torch.full((n,), float("inf"), device=w.device)
                best_word = torch.zeros(n, dtype=torch.uint8, device=w.device)
                for delta in range(-search, 2):
                    word = (center + delta).clamp(1, 126).to(torch.uint8)
                    scale = word.view(torch.float8_e4m3fn).float() / d
                    encoded = _codes(group / scale[:, None])
                    error = ((group - _decode(encoded) * scale[:, None]) ** 2).sum(1)
                    better = error < best_error
                    best_error = torch.where(better, error, best_error)
                    best_word = torch.where(better, word, best_word)
                best_word = torch.where(group.abs().amax(1) == 0, 0, best_word)
                scales[:, (start + column) // 16] = best_word
                scale = best_word.view(torch.float8_e4m3fn).float() / d
                denominator = torch.where(scale > 0, scale, 1.)
            current = local[:, column].clone()
            encoded = _codes(current / denominator)
            codes[:, start + column] = torch.where(scale > 0, encoded, 0)
            quantized = _decode(encoded) * scale
            error = (current - quantized) / hi[column, column]
            local[:, column:] -= error[:, None] * hi[column, column:][None, :]
            errors[:, column] = error
        w[:, end:] -= errors @ factor[start:end, end:]
    return QuantizedMatrix(codes[:, ::2] | (codes[:, 1::2] << 4), scales)
