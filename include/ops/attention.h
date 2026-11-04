#pragma once
#include "core/tensor.h"

#ifdef CUDA
#else  // CUDA

#include "kernel/cpu/attention.h"
void attention(
    Vector<bf16> &o,
    const Vector<bf16> &q,
    const Tensor<bf16, 3> &key_cache,
    const Tensor<bf16, 3> &value_cache,
    Matrix<f32> &scores,
    size_t layer,
    size_t pos,
    size_t head_size
);

#endif // CUDA 