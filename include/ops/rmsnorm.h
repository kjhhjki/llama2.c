#ifndef LLAMA_OPS_RMSNORM_H
#define LLAMA_OPS_RMSNORM_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA
void rmsnorm(Vector<bf16>&, const Vector<bf16>&, const Vector<bf16>&);

#endif  // CUDA
#endif  // LLAMA_OPS_RMSNORM_H
