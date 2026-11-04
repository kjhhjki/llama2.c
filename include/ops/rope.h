#pragma once
#include "core/tensor.h"
#include "kernel/functional.h"

inline constexpr f32 rope_theta_base = 10000.0f;

std::vector<f32> makeRopeFrequencies(size_t head_size);
void makeRopeCoefficients(
    kernel::RopeCoefficients&, 
    size_t, 
    std::span<const f32>
);


#ifdef CUDA
#else   // CUDA

#include "kernel/cpu/rope.h"

void rope(Vector<bf16>&, Vector<bf16>&, const kernel::RopeCoefficients &);
void rope(
    Vector<bf16>::ViewType, 
    Vector<bf16>::ViewType, 
    const kernel::RopeCoefficients&
);

#endif  // CUDA 