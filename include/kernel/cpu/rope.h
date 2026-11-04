#pragma once
#include "core/tensor.h"
#include "kernel/functional.h"

namespace kernel 
{ 
    void rope(
        Vector<bf16>::ViewType, 
        Vector<bf16>::ViewType,
        const RopeCoefficients&
    ) noexcept;
} 