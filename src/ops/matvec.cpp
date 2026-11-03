#include "ops/matvec.h"
#include "kernel/cpu/matvec.h"
#include <stdexcept>

#ifdef CUDA

#else  // CUDA
void matvec(Vector<bf16>& o, const Matrix<bf16>& A, const Vector<bf16>& x) {
    const size_t m = A.shape()[0];
    const size_t n = A.shape()[1];
    if (x.size() != n) {
        throw std::invalid_argument("matvec shape mismatch");
    }
    if (std::addressof(o) == std::addressof(x)) {
        throw std::invalid_argument("matvec aliasing");
    }
    if (o.size() != m) {
        o.resize(m);
    }
    kernel::matvec(o.view(), A.view(), x.view());
}
#endif // CUDA