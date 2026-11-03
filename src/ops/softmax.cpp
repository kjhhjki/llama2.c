#include "ops/softmax.h"
#include "kernel/cpu/softmax.h"
#include <stdexcept>

#ifdef CUDA
#else   // CUDA

void softmax(Vector<bf16>& x) {
    kernel::softmax(std::span<bf16>(x.data(), x.size()));
}

#endif  // CUDA