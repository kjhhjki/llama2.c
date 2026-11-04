#pragma once
#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

#include "kernel/cpu/swiglu.h"
void swiglu(Vector<bf16>&, Vector<bf16>&);

#endif  // CUDA 