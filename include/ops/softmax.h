#ifndef LLAMA_OPS_SOFTMAX_H
#define LLAMA_OPS_SOFTMAX_H

#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA
void softmax(Vector<bf16>&);

#endif  // CUDA
#endif  // LLAMA_OPS_SOFTMAX_H
