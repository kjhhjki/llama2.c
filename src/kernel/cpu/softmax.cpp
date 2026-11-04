#include "kernel/cpu/softmax.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

void kernel::softmax(std::span<f32> x) 
{
    const size_t n = x.size();
    f32 max = *std::ranges::max_element(x),
        s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max);
        s += x[i];
    }
    for (size_t i = 0; i < n; ++i) {
        x[i] /= s;
    }
} 