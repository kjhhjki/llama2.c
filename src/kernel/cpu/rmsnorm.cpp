#include "kernel/cpu/rmsnorm.h"
#include <cmath>
#include <omp.h>

void kernel::rmsnorm(
    Vector<bf16>::ViewType o, 
    const Vector<bf16>::ConstViewType x, 
    const Vector<bf16>::ConstViewType w
) {
    const size_t n = x.size();
    f32 s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const f32 xi = static_cast<f32>(x[i]);
        s += xi * xi;
    }
    constexpr bf16 eps = 1e-5f;
    s = 1.f / std::sqrt(s / n + eps);
    for (size_t i = 0; i < n; ++i) {
        o[i] = static_cast<bf16>(x[i] * s * w[i]);
    }
} 