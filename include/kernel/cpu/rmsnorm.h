#pragma once
#include "core/tensor.h"

namespace kernel 
{ 
    void rmsnorm(
        Vector<bf16>::ViewType, 
        Vector<bf16>::ConstViewType, 
        Vector<bf16>::ConstViewType
    ); 
}
 