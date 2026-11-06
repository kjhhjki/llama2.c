#pragma once
#ifdef CUDA

#else  // CUDA
#include <cstdint>

using i8 = std::int8_t;
using u8 = std::uint8_t;
using f32 = float;
using bf16 = __bf16;
using TokenID = std::int32_t;

#endif  // CUDA 