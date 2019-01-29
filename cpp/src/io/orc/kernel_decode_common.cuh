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

#ifndef __ORC_KERNEL_DECODE_COMMON_H__
#define __ORC_KERNEL_DECODE_COMMON_H__

#include "kernel_private.cuh"

#define RLE_Repeat        0
#define RLE_Literal       1

template <class T, class T_writer = data_writer<T>, 
    class T_reader = stream_reader<T>>
class ORCdecodeCommon
{
public:
    __device__ ORCdecodeCommon<T, T_writer, T_reader>(KernelParamCommon* param)
        : reader(param)
        , writer(param)
        , stat(param->stat)
    {};

//    __device__ ~ORCdecodeCommon() {};
//    __device__ void decode() {};

public:
    T_reader                reader;
    T_writer                writer;

    CudaThreadControl        ctc;
    CudaOrcKernelRetStats   stat;
};


#endif // __ORC_KERNEL_DECODE_COMMON_H__
