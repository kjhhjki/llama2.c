#pragma once
#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

void softmax(Vector<f32>&);

#endif  // CUDA
 