#include "ops/rope.h"
#include <cmath>

std::vector<f32> makeRopeFrequencies(size_t head_size)
{
    const size_t half = head_size / 2;
#ifdef CHECK_BOUNDS
    if (head_size == 0 || head_size % 2 != 0) {
        throw std::invalid_argument("Head size must be a positive even number");
    }
#endif
    std::vector<f32> freqs(half);
    for (size_t i = 0; i < half; ++i) {
        const f32 exponent = -2.0f * static_cast<f32>(i) /
                             static_cast<f32>(head_size);
        freqs[i] = std::pow(rope_theta_base, exponent);
    }
    return freqs;
}
void makeRopeCoefficients(
    kernel::RopeCoefficients& coef, 
    size_t pos,
    std::span<const f32> freqs
) {
    const size_t half = coef.head_size / 2;
#ifdef CHECK_BOUNDS
    if (coef.head_size == 0 || coef.head_size % 2 != 0 ||
        freqs.size() != half || coef.cos.size() != half || coef.sin.size() != half) {
        throw std::invalid_argument("Invalid RoPE coefficients");
    }
#endif
    for (size_t i = 0; i < half; ++i) {
        const f32 theta = static_cast<f32>(pos) * freqs[i];
        coef.cos[i] = std::cos(theta);
        coef.sin[i] = std::sin(theta);
    }
}

#ifdef CUDA
#else   // CUDA

void rope(
    Vector<bf16> &q, 
    Vector<bf16> &k, 
    const kernel::RopeCoefficients &coef
) {
    const size_t head_size = coef.head_size;
    if (head_size == 0 || head_size % 2 != 0 ||
        q.size() == 0 || k.size() == 0 ||
        q.size() % head_size != 0 ||
        k.size() % head_size != 0 ||
        k.size() > q.size()) {
        throw std::invalid_argument("Invalid RoPE dimensions");
    }

    kernel::rope(q.view(), k.view(), coef);
}
void rope(
    Vector<bf16>::ViewType q, 
    Vector<bf16>::ViewType k, 
    const kernel::RopeCoefficients &coef
) {
    const size_t head_size = coef.head_size;
    if (head_size == 0 ||
        head_size % 2 != 0 ||
        q.extent(0) == 0 ||
        k.extent(0) == 0 ||
        q.extent(0) % head_size != 0 ||
        k.extent(0) % head_size != 0 ||
        k.extent(0) > q.extent(0)) {
        throw std::invalid_argument("Invalid RoPE dimensions");
    }
    kernel::rope(q, k, coef);
}

#endif  // CUDA 