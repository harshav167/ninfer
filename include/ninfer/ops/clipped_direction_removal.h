#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Removes two directions from every column h = residual[:,m], each only above a reference:
 *
 *   h1 = h  - max(dot(h,  u0) - c0, 0) * u0
 *   h2 = h1 - max(dot(h1, u1) - c1, 0) * u1
 *   residual[:,m] = BF16(h2)
 *
 * with u0/u1 = directions[:,0]/directions[:,1] and c0/c1 = references[0]/references[1]. The
 * second dot product uses the updated h1. Directions need not have unit length; a zero u1
 * makes the second step the identity for every c1.
 *
 * residual is a contiguous BF16 [H,T] tensor (ne[2]=ne[3]=1) updated in place; columns are
 * independent. directions is contiguous FP32 [H,2] and references contiguous FP32 [2]; both are
 * preserved and must not overlap residual. H is a positive multiple of 8 and at most 16384.
 * residual and directions are 16-byte aligned, references 8-byte aligned (so a slice of an
 * [H,2,K] or [2,K] table of per-layer slots is admissible). Operands must be finite.
 *
 * Dot products accumulate in FP32 with unspecified association. h1 and h2 are FP32 values
 * without intermediate BF16 rounding; the result is rounded to nearest-even BF16 once. A step
 * whose clipped coefficient is 0 leaves the column unchanged bit for bit.
 *
 * T=0 does no device work and does not inspect residual storage; the remaining operands are
 * still validated. Submits on execution.stream(); no workspace or retained state. A captured
 * call reads the current contents of all operands at their captured addresses on every replay.
 */
void clipped_direction_removal(const Tensor& directions, const Tensor& references, Tensor& residual,
                               CudaExecutionView execution);

} // namespace ninfer::ops
