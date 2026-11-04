#include "ops/softmax.h"
#include "kernel/cpu/softmax.h"

#ifdef CUDA
#else   // CUDA

void softmax(Vector<f32>& x)
{
    kernel::softmax(std::span<f32>(x.data(), x.size()));
}

#endif  // CUDA 