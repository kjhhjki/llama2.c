#pragma once
#include "core/tensor.h"
#include "kernel/functional.h"

namespace kernel
{
    void attention(
        Vector<bf16>::ViewType o,
        Vector<bf16>::ConstViewType q,
        Tensor<bf16, 3>::ConstViewType key_cache,
        Tensor<bf16, 3>::ConstViewType value_cache,
        Matrix<f32>::ViewType scores,
        size_t layer,
        size_t pos,
        size_t head_size
    );
} 