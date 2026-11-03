#ifndef LLAMA_KERNEL_CPU_RMSNORM_H
#define LLAMA_KERNEL_CPU_RMSNORM_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

namespace kernel 
{ 
    void rmsnorm(Vector<bf16>::ViewType, Vector<bf16>::ConstViewType, Vector<bf16>::ConstViewType); 
}

#endif  // CUDA
#endif  // LLAMA_KERNEL_CPU_RMSNORM_H
