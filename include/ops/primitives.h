#pragma once

#include "core/tensor.h"
#include "kernel/functional.h"

#ifdef CUDA
#else  // CUDA
#include "kernel/cpu/primitives.h"
#include <span>
#include <stdexcept>


inline void add(Vector<bf16> &x, const Vector<bf16> &y) 
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("add shape mismatch");
    }

    kernel::add(x.view(), y.view());
}
inline f32 dot(Vector<bf16> &x, const Vector<bf16> &y)  
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("dot shape mismatch");
    }

    return kernel::dot(
        std::span<const bf16>(x.view().data_handle(), x.size()),
        std::span<const bf16>(y.view().data_handle(), y.size())
    );
}

inline void kvCacheUpdate(
    Tensor<bf16, 3>& cache,
    const Vector<bf16>& x,
    size_t layer,
    size_t pos
) {
    if (layer >= cache.shape()[0] ||
        pos >= cache.shape()[1] ||
        cache.shape()[2] != x.size()) {
        throw std::invalid_argument("kv_cache_update shape mismatch");
    }

    kernel::kvCacheUpdate(cache.row(layer, pos), x.view());
}
inline void embedding(
    Vector<bf16>& o,
    const Matrix<bf16>& table,
    size_t token
) {
    const size_t vocab_size = table.shape()[0];
    const size_t dim = table.shape()[1];

    if (token >= vocab_size) {
        throw std::out_of_range("embedding index out of bounds");
    }

    if (o.size() != dim) {
        o.resize(dim);
    }

    kernel::embedding(o.view(), table.view(), token);
}

inline void weightedSum(
    Vector<bf16> &o,
    kernel::KVHeadView x,
    std::span<const f32> w
) {
    if (w.size() != x.extent(0)) {
        throw std::invalid_argument("weighted_sum shape mismatch");
    }
    if (o.size() != x.extent(1)) {
        o.resize(x.extent(1));
    }

    kernel::weightedSum(
        std::span<bf16>(o.view().data_handle(), o.size()),
        x, 
        w);
}
inline void weightedSum(
    Vector<bf16> &o,
    kernel::KVHeadView x,
    f32 w
) {
    if (o.size() != x.extent(1)) {
        o.resize(x.extent(1));
    }

    kernel::weightedSum(
        std::span<bf16>(o.view().data_handle(), o.size()),
        x, 
        w
    );
}

#endif  // CUDA 