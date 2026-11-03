#ifndef LLAMA_KERNEL_CPU_MATVEC_H
#define LLAMA_KERNEL_CPU_MATVEC_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

namespace kernel 
{ 
    void matvec(Vector<bf16>::ViewType, Matrix<bf16>::ConstViewType, 
                                        Vector<bf16>::ConstViewType); 
}

#endif  // CUDA
#endif  // LLAMA_KERNEL_CPU_MATVEC_H
