#pragma once
#include "core/tensor.h"

namespace kernel 
{ 
    void softmax(std::span<f32>); 
} 