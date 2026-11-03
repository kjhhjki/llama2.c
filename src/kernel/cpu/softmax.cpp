#include "kernel/cpu/softmax.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

void kernel::softmax(std::span<bf16> x) {
    if (x.empty()) {
        throw std::invalid_argument("softmax input is empty");
    }
    const size_t n = x.size();
    f32 max = *std::max_element(x.begin(), x.end()),
        s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max);
        s += x[i];
    }
    for (size_t i = 0; i < n; ++i) {
        x[i] /= s;
    }
}