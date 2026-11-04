#pragma once
#include "core/tensor.h"

namespace kernel
{
    void swiglu(
        Vector<bf16>::ViewType, 
        Vector<bf16>::ConstViewType
    );
} 