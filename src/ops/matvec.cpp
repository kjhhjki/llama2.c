#include "ops/matvec.h"
#include "kernel/cpu/matvec.h"
#include <stdexcept>

#ifdef CUDA

#else  // CUDA
void matvec(Vector<bf16>& o, const Matrix<bf16>& A, const Vector<bf16>& x) 
{
    if (A.shape()[1] != x.size()) {
        throw std::invalid_argument("matvec shape mismatch");
    }
    if (std::addressof(o) == std::addressof(x)) {
        throw std::invalid_argument("matvec aliasing");
    }
    if (o.size() != A.shape()[0]) {
        o.resize(A.shape()[0]);
    }
    kernel::matvec(o.view(), A.view(), x.view());
}
void matvec(
    Vector<bf16>::ViewType o, 
    const Matrix<bf16> &A, 
    const Vector<bf16> &x
) {
    if (A.shape()[1] != x.size() ||
        A.shape()[0] != o.extent(0)) {
        throw std::invalid_argument("matvec_into shape mismatch");
    }
    kernel::matvec(o, A.view(), x.view());
}

#endif // CUDA 