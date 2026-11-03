#ifndef LLAMA_CORE_TYPES_H
#define LLAMA_CORE_TYPES_H

#ifdef CUDA

#else  // CUDA
#include <cstdint>

using i8 = std::int8_t;
using u8 = std::uint8_t;
using f32 = float;
using bf16 = float;

#endif  // CUDA
#endif  // LLAMA_CORE_TYPES_H
