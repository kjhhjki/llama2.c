#pragma once
#include "core/tensor.h"

namespace kernel 
{ 
    void matvec(
        Vector<bf16>::ViewType, 
        Matrix<bf16>::ConstViewType, 
        Vector<bf16>::ConstViewType
    ); 
}
 