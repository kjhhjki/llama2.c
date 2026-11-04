#include "kernel/cpu/swiglu.h"
#include <cmath>
#include <omp.h>

void kernel::swiglu(Vector<bf16>::ViewType gate, Vector<bf16>::ConstViewType up) 
{
    const size_t n = gate.extent(0);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        const f32 g = static_cast<f32>(gate[i]);
        const f32 u = static_cast<f32>(up[i]);

        const f32 silu = g / (1.0f + std::exp(-g));

        gate[i] = static_cast<bf16>(silu * u);
    }
} 