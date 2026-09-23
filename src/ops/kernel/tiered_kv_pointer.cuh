#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

// Both addresses are on the inference GPU. Overflow bytes have already been
// staged from secondary VRAM by Program; kernels never dereference remote VRAM.
template <typename T>
struct TieredKVPointer {
    T* resident                    = nullptr;
    T* staged                      = nullptr;
    std::int64_t resident_elements = 0;

    __device__ __forceinline__ T* operator+(std::int64_t index) const {
        return index < resident_elements ? resident + index : staged + (index - resident_elements);
    }

    __device__ __forceinline__ T& operator[](std::int64_t index) const { return *(*this + index); }
};

template <typename T, bool Tiered>
using KVPlanePointer = std::conditional_t<Tiered, TieredKVPointer<T>, T*>;

template <typename T, bool Tiered>
KVPlanePointer<T, Tiered> kv_plane_pointer(const Tensor& resident, const Tensor& staged) {
    if constexpr (Tiered) {
        return {static_cast<T*>(resident.data), static_cast<T*>(staged.data),
                static_cast<std::int64_t>(resident.bytes() / sizeof(T))};
    } else {
        return static_cast<T*>(resident.data);
    }
}

} // namespace ninfer::ops
