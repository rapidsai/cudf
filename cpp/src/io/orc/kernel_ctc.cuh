/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef __ORC_KERNEL_CTC_H__
#define __ORC_KERNEL_CTC_H__

#include "kernel_private_common.cuh"


class CudaThreadControl {
public:
    __device__ CudaThreadControl() {
        tid = threadIdx.x;

        warp_bitshift = (blockDim.x == 32) ? 5 : (blockDim.x == 16) ? 4 : 3;
        warp_bitmask = blockDim.x - 1;
    }
    __device__ ~CudaThreadControl() {};

    __device__ int getFullCount(int length) const { return (length >> warp_bitshift); };
    __device__ int getRestCount(int length) const { return (length & warp_bitmask); };

    __device__ int getBitSize(int bit_width) const {    //  (1, 2, 4) => (1, 2, 3)
        return (bit_width == 4) ? 3 : bit_width;
    };

    __device__ int getBitIndex(int bit_width) const     // (1, 2, 4) => (3, 2, 1)
    {
        return 4 - getBitSize(bit_width);
    };

    __device__ int bitmask(int width) const {    // (1, 2, 4) => tid & (0x01, 0x03, 0x0f)
        return 0x0f >> (4 - width);
    };

    __device__ int tid_per(int width) const {    // (1, 2, 4) => tid >> (3, 2, 1)
        return tid >> getBitIndex(width);
    };

    __device__ int tid_mod(int width) const {    // (1, 2, 4) => & (0x07, 0x03, 0x01)
        return (tid & (0x0f >> getBitSize(width)));
    };

    __device__ int tid_rev_shift(int width) const {    // (1, 2, 4) => 7 - tid_mod(witdh)
        return (8 - (tid_mod(width) + 1)*width);
    };

    __device__ int block_per(int width) const {    // (1, 2, 4) => blockDim.x >> (3, 2, 1)
        return blockDim.x >> getBitIndex(width);
    };

public:
    int tid;        // thread ID

    int warp_bitshift;
    int warp_bitmask;
};



#endif // __ORC_KERNEL_CTC_H__
