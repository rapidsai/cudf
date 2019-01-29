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

#ifndef __ORC_KERNEL_WRITER_H__
#define __ORC_KERNEL_WRITER_H__

#include "kernel_private_common.cuh"

template <class T, class T_converter= ORCConverterBase<T>>
class data_writer {
public:
    __device__ data_writer(KernelParamCommon* param)
        : top(reinterpret_cast<T*>(param->output))
        , size(param->output_count)
        , local_offset(0)
        , converter(param)
    {};

    __device__ void write(T data, size_t offset) {
        converter.Convert(data);
        top[offset] = data;
    }

    __device__ void add_value(T data, size_t offset) {
        converter.Convert(data);
        top[offset] += data;
    }

    __device__ void write_local(T data, size_t offset) {
        write(data, local_offset + offset);
    };

    __device__ void add_offset(int count) {
        local_offset += count;
    };

    __device__ bool end() {
        return (local_offset >= size);
    };

    __device__ int get_decoded_count() {
        return local_offset;
    };

    __device__ int get_written_range() {
        return local_offset;
    };

    __device__ void expect(int count) {};

protected:
    T* top;
    size_t size;

    T_converter converter;

    int local_offset;    //< offset for the index.
};



#endif // __ORC_KERNEL_WRITER_H__
