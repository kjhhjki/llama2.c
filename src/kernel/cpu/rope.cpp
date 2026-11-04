#include "kernel/cpu/rope.h"
#include <omp.h>

namespace kernel
{
namespace 
{
    inline void rotate(bf16 &v0, bf16 &v1, f32 c, f32 s) noexcept
    {
        const f32 x0 = static_cast<f32>(v0),
                  x1 = static_cast<f32>(v1);
        const f32 y0 = x0 * c - x1 * s,
                  y1 = x0 * s + x1 * c;
        v0 = static_cast<bf16>(y0);
        v1 = static_cast<bf16>(y1);
    }
}
void rope(
    Vector<bf16>::ViewType q, 
    Vector<bf16>::ViewType k, 
    const RopeCoefficients &coef
) noexcept {
    const size_t dim = q.extent(0), 
                 kv_dim = k.extent(0);
    const size_t half = coef.head_size / 2;
    for (size_t i = 0; (i * 2) < dim; ++i) {
        const size_t pair_idx = i % half, p = i * 2;
        const f32 c = coef.cos[pair_idx],
                  s = coef.sin[pair_idx];
        rotate(q[p], q[p ^ 1], c, s);
        if (p < kv_dim) {
            rotate(k[p], k[p ^ 1], c, s);
        }
    }
}
}
 