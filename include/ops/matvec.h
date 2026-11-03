#ifndef LLAMA_OPS_MATVEC_H
#define LLAMA_OPS_MATVEC_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA
void matvec(Vector<bf16>&, const Matrix<bf16>&, const Vector<bf16>&);


#endif  // CUDA
#endif  // LLAMA_OPS_MATVEC_H