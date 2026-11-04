#pragma once
#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

void rmsnorm(Vector<bf16>&, const Vector<bf16>&, const Vector<bf16>&);

#endif  // CUDA
