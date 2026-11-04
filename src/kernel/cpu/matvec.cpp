#include "kernel/cpu/matvec.h"

void kernel::matvec(
    Vector<bf16>::ViewType o, 
    const Matrix<bf16>::ConstViewType A, 
    const Vector<bf16>::ConstViewType x
) {
    const size_t m = A.extent(0);
    const size_t n = A.extent(1);
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        f32 sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            sum += static_cast<f32>(A[i, j]) * static_cast<f32>(x[j]);
        }
        o[i] = static_cast<bf16>(sum);
    }
} 