#pragma once
#include "core/tensor.h"
#include "core/types.h"
#include <utility>

#ifdef CUDA
#else  // CUDA
namespace kernel
{

constexpr f32 eps = 1e-5f;
struct RopeCoefficients {
    size_t head_size;

    std::vector<f32> cos;
    std::vector<f32> sin;

    explicit RopeCoefficients(size_t head_size) : 
        head_size(head_size), cos(head_size / 2), sin(head_size / 2) {}
};

using KVHeadView = std::mdspan<
    const bf16,
    std::dextents<size_t, 2>,
    std::layout_stride
>;

inline kernel::KVHeadView getKVHeadView(
    const Tensor<bf16, 3>& cache,
    size_t layer,
    size_t pos,
    size_t kv_head,
    size_t head_size
) {
    const size_t seq_len = cache.shape()[1];
    const size_t kv_dim = cache.shape()[2];
    if (layer >= cache.shape()[0] ||
        pos >= seq_len ||
        head_size == 0 ||
        kv_dim % head_size != 0 ||
        kv_head >= kv_dim / head_size) {
        throw std::invalid_argument("Invalid key-value head");
    }
    const size_t begin = kv_head * head_size;
    const size_t end = begin + head_size;

    return std::submdspan(
        cache.view(),
        layer,
        std::pair{size_t{0}, pos + 1},
        std::pair{begin, end}
    );
}
}
#endif  // CUDA 