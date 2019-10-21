/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdint.h>

namespace cudf {
namespace io {

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v)        __shfl_sync(~0, v, 0)
#define SHFL(v, t)      __shfl_sync(~0, v, t)
#define SHFL_XOR(v, m)  __shfl_xor_sync(~0, v, m)
#define SYNCWARP()      __syncwarp()
#define BALLOT(v)       __ballot_sync(~0, v)
#else
#define SHFL0(v)        __shfl(v, 0)
#define SHFL(v, t)      __shfl(v, t)
#define SHFL_XOR(v, m)  __shfl_xor(v, m)
#define SYNCWARP()
#define BALLOT(v)       __ballot(v)
#endif

#if (__CUDA_ARCH__ >= 700)
#define NANOSLEEP(d)  __nanosleep(d)
#else
#define NANOSLEEP(d)  clock()
#endif

// Warp reduction helpers
template <typename T> inline __device__ T WarpReduceSum2(T acc)     { return acc + SHFL_XOR(acc, 1); }
template <typename T> inline __device__ T WarpReduceSum4(T acc)     { acc = WarpReduceSum2(acc); return acc + SHFL_XOR(acc, 2); }
template <typename T> inline __device__ T WarpReduceSum8(T acc)     { acc = WarpReduceSum4(acc); return acc + SHFL_XOR(acc, 4); }
template <typename T> inline __device__ T WarpReduceSum16(T acc)    { acc = WarpReduceSum8(acc); return acc + SHFL_XOR(acc, 8); }
template <typename T> inline __device__ T WarpReduceSum32(T acc)    { acc = WarpReduceSum16(acc); return acc + SHFL_XOR(acc, 16); }

template <typename T> inline __device__ T WarpReducePos2(T pos, uint32_t t) { T tmp = SHFL(pos, t & 0x1e); pos += (t & 1) ? tmp : 0; return pos; }
template <typename T> inline __device__ T WarpReducePos4(T pos, uint32_t t) { T tmp; pos = WarpReducePos2(pos, t); tmp = SHFL(pos, (t & 0x1c) | 1); pos += (t & 2) ? tmp : 0; return pos; }
template <typename T> inline __device__ T WarpReducePos8(T pos, uint32_t t) { T tmp; pos = WarpReducePos4(pos, t); tmp = SHFL(pos, (t & 0x18) | 3); pos += (t & 4) ? tmp : 0; return pos; }
template <typename T> inline __device__ T WarpReducePos16(T pos, uint32_t t) { T tmp; pos = WarpReducePos8(pos, t); tmp = SHFL(pos, (t & 0x10) | 7); pos += (t & 8) ? tmp : 0; return pos; }
template <typename T> inline __device__ T WarpReducePos32(T pos, uint32_t t) { T tmp; pos = WarpReducePos16(pos, t); tmp = SHFL(pos, 0xf); pos += (t & 16) ? tmp : 0; return pos; }

inline __device__ double Int128ToDouble_rn(uint64_t lo, int64_t hi)
{
    double sign;
    if (hi < 0) {
        sign = -1.0;
        lo = (~lo) + 1;
        hi = (~hi) + (lo == 0);
    } else {
        sign = 1.0;
    }
    return sign * __fma_rn(__ll2double_rn(hi), 4294967296.0 * 4294967296.0, __ull2double_rn(lo));
}

} // namespace io
} // namespace cudf

