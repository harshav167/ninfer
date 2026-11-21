#include "ops/linear/projection/nvfp4/nvfp4_shapes.h"
#include "ops/linear/projection/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
template <int N, int K>
void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using Geometry = Nvfp4Geometry<N, K>;
    if (x.ne[1] <= 8)
        return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>(x, w, y, stream);
    if (x.ne[1] <= 16)
        return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>(x, w, y, stream);
    nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>(x, w, y, stream);
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N5120K25600{5120, 25600, launch_a16<5120, 25600>, nullptr, uses_a4};
const Nvfp4LinearShape kNvfp4N5120K4096{5120, 4096, launch_a16<5120, 4096>, nullptr, uses_a4};
} // namespace ninfer::ops::detail
