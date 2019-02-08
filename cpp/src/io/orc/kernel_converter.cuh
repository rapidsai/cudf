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

#ifndef __ORC_KERNEL_CONVERTER_H__
#define __ORC_KERNEL_CONVERTER_H__

#include "kernel_private_common.cuh"

namespace cudf {
namespace orc {

//! the base class of converter
template <class T>
class ORCConverterBase
{
public:
    __device__ ORCConverterBase(KernelParamCommon* param)
    {};

    __device__ void Convert(T& value) {};
};

//< int32_t days since the UNIX epoch
template<class T>
using ORCConverterDate32 = ORCConverterBase<T>;


//< int64_t milliseconds since the UNIX epoch
template <class T>
class ORCConverterGdfDate64 : public ORCConverterBase<T>
{
public:
    __device__ ORCConverterGdfDate64(KernelParamCommon* param)
        : ORCConverterBase<T>(param)
    {};

    __device__ __host__ void Convert(T& value) {
        value *= (24 * 60 * 60 * 1000); // date to milli-seconds
    };
};

template <class T>
__device__ __host__ inline 
void NanoConvert(T& value) 
{
    const int scale[8] = { 1, 100, 1000, 10000, 100000, 10000000, 10000000, 100000000 };
    int zero_count = (value & 0x07);
    value >>= 3;
    value *= scale[zero_count];
}


template <class T>
class ORCConverterTimestamp : public ORCConverterBase<T>
{
public:
    __device__ ORCConverterTimestamp(KernelParamCommon* param)
        : ORCConverterBase<T>(param)
    {};

    __device__ __host__ void Convert(T& value) {
        NanoConvert(value);
    };
};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_CONVERTER_H__
