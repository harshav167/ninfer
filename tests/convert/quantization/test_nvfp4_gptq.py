import struct

import pytest
import torch

from tools.convert.quantization.nvfp4_gptq import inverse_factor, quantize_gptq


def decode(encoded, divisor):
    table = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    rows = []
    for packed, scales in zip(encoded.codes.tolist(), encoded.scales.tolist()):
        row = []
        for i, pair in enumerate(packed):
            exponent, mantissa = divmod(scales[i // 8], 8)
            scale = mantissa / 512 if exponent == 0 else (8 + mantissa) * 2. ** (exponent - 10)
            for code in (pair & 15, pair >> 4):
                row.append(table[code & 7] * (-1 if code & 8 else 1) * scale / divisor)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def test_inverse_factor_matches_independent_regularized_inverse():
    x = torch.randn(80, 32, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    h = x.T @ x
    before = h.clone()
    factor, dead = inverse_factor(h, .02)
    regularized = h + torch.eye(32, dtype=h.dtype) * h.diagonal().mean() * .02
    torch.testing.assert_close(factor.double().T @ factor.double(), torch.linalg.inv(regularized),
                               atol=1e-8, rtol=2e-5)
    assert not dead.any()
    assert torch.equal(h, before)


def test_exact_nvfp4_weights_survive_and_dead_columns_are_zero():
    w = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6.,
                       0., -.5, -1., -1.5, -2., -3., -4., -6.] * 2], dtype=torch.bfloat16)
    h = torch.eye(32)
    h[0, 0] = 0
    w[0, 0] = 5
    before = w.clone()
    factor, dead = inverse_factor(h)
    result = quantize_gptq(w, factor, dead, struct.pack('<f', 1.))
    expected = w.double()
    expected[:, 0] = 0
    torch.testing.assert_close(decode(result, 1.), expected, atol=0, rtol=0)
    assert torch.equal(w, before)


def test_feedback_crosses_column_blocks_and_row_chunking_is_invariant():
    generator = torch.Generator().manual_seed(27)
    x = torch.randn(128, 32, generator=generator)
    x[:, 16:] += x[:, :16] * .8
    weight = torch.randn(5, 32, generator=generator).bfloat16()
    factor, dead = inverse_factor(x.T @ x)
    divisor = struct.pack('<f', 64.)
    full = quantize_gptq(weight, factor, dead, divisor, block_size=32)
    tiled = quantize_gptq(weight, factor, dead, divisor, block_size=16)
    torch.testing.assert_close(decode(tiled, 64.), decode(full, 64.), atol=0, rtol=0)
    parts = [quantize_gptq(w, factor, dead, divisor, block_size=32) for w in weight.split(2)]
    assert torch.equal(full.codes, torch.cat([p.codes for p in parts]))
    assert torch.equal(full.scales, torch.cat([p.scales for p in parts]))
    diagonal_factor, diagonal_dead = inverse_factor(torch.diag((x.T @ x).diagonal()))
    independent = quantize_gptq(weight, diagonal_factor, diagonal_dead, divisor)
    error = (decode(full, 64.) - weight.double()) @ x.double().T
    independent_error = (decode(independent, 64.) - weight.double()) @ x.double().T
    assert error.square().sum() < independent_error.square().sum()


@pytest.mark.parametrize('bad', [torch.zeros(16, 16), torch.full((16, 16), float('nan')),
                                -torch.eye(16), torch.ones(16, 15)])
def test_rejects_unusable_calibration(bad):
    with pytest.raises(ValueError):
        inverse_factor(bad)


def test_rejects_requantizing_non_bf16_source():
    factor, dead = inverse_factor(torch.eye(16))
    with pytest.raises(ValueError, match='original BF16'):
        quantize_gptq(torch.ones(2, 16), factor, dead, struct.pack('<f', 1.))
