#include "ops/swiglu.h"

#ifdef CUDA
#else   // CUDA

void swiglu(Vector<bf16> &gate, Vector<bf16> &up) 
{
    if (gate.size() != up.size()) {
        throw std::invalid_argument("Gate and Up vectors must have the same size");
    }
    kernel::swiglu(gate.view(), up.view());
}

#endif  // CUDA