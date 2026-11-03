#ifndef LLAMA_KERNEL_CPU_SOFTMAX_H
#define LLAMA_KERNEL_CPU_SOFTMAX_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

namespace kernel 
{ 
    void softmax(std::span<bf16>); 
}

#endif  // CUDA
#endif  // LLAMA_KERNEL_CPU_SOFTMAX_H
