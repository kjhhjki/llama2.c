#ifndef LLAMA_KERNEL_CPU_ROPE_H
#define LLAMA_KERNEL_CPU_ROPE_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

namespace kernel 
{ 
    void rope(Vector<bf16>::ViewType); 
}

#endif  // CUDA
#endif
