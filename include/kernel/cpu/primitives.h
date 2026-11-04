#pragma once
#include "core/tensor.h"
#include "kernel/functional.h"
#include <algorithm>

namespace kernel
{
/*
 * add 
 * dot 
 * kvCacheUpdate 
 * embedding
 * weightedSum 
 */
inline void add(
    Vector<bf16>::ViewType x, 
    Vector<bf16>::ConstViewType y
) noexcept {
    const size_t n = x.extent(0);
    for (size_t i = 0; i < n; ++i) {
        const f32 xi = static_cast<f32>(x[i]),
                  yi = static_cast<f32>(y[i]);
        x[i] = static_cast<bf16>(xi + yi);
    }
}
inline f32 dot(
    std::span<const bf16> x,
    std::span<const bf16> y
) noexcept {
    const size_t n = x.size();
    f32 s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const f32 xi = static_cast<f32>(x[i]),
                  yi = static_cast<f32>(y[i]);
        s += xi * yi;
    }
    return s;
}

inline void kvCacheUpdate(
    Tensor<bf16, 3>::VectorViewType cache_row,
    Vector<bf16>::ConstViewType x
) noexcept {
    const size_t kv_dim = cache_row.extent(0);
    std::copy_n(x.data_handle(), kv_dim, cache_row.data_handle());
}
inline void embedding(
    Vector<bf16>::ViewType out, 
    Matrix<bf16>::ConstViewType table, 
    size_t token
) noexcept {
    const size_t n = table.extent(1);
    std::copy_n(
        table.data_handle() + token * n, 
        n, 
        out.data_handle()
    );
}

inline void weightedSum(
    std::span<bf16> o,
    KVHeadView x,
    std::span<const f32> w
) noexcept {
    const size_t dim = o.size();
    const size_t T = w.size();

    for (size_t i = 0; i < dim; ++i) {
        f32 s = 0.0f;
        for (size_t t = 0; t < T; ++t) {
            s += w[t] * static_cast<f32>(x[t, i]);
        }
        o[i] = static_cast<bf16>(s);
    }
}
inline void weightedSum(
    std::span<bf16> o, 
    KVHeadView x, 
    f32 w
) noexcept {
    const size_t n = o.size(),
                 T = x.extent(0);
    for (size_t i = 0; i < n; ++i) {
        f32 s = 0.0f;
        for (size_t t = 0; t < T; ++t) {
            const f32 xi = static_cast<f32>(x[t, i]);
            s += xi * w;
        }
        o[i] = static_cast<bf16>(s);
    }
}
} 