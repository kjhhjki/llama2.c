#include "ops/rmsnorm.h"
#include "kernel/cpu/rmsnorm.h"
#include <stdexcept>

#ifdef CUDA
#else   // CUDA

void rmsnorm(Vector<bf16>& o, const Vector<bf16>& x, const Vector<bf16>& w) {
    const size_t n = x.size();
    if (n == 0 || w.size() != n) {
        throw std::invalid_argument("rmsnorm shape mismatch.");
    }
    if (std::addressof(o) == std::addressof(w)) {
        throw std::invalid_argument("rmsnorm weight aliasing");
    }
    if (o.size() != n) {
        o.resize(n);
    }
    kernel::rmsnorm(o.view(), x.view(), w.view());
}

#endif  // CUDA