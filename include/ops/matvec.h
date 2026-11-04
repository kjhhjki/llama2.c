#pragma once
#include "core/tensor.h"

#ifdef CUDA
#else   // CUDA

void matvec(Vector<bf16>&, const Matrix<bf16>&, const Vector<bf16>&);
void matvec(
    Vector<bf16>::ViewType, 
    const Matrix<bf16>&, 
    const Vector<bf16>&
);

#endif  // CUDA 