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

#ifndef __ORC_KERNEL_READER_BUFFERS_H__
#define __ORC_KERNEL_READER_BUFFERS_H__

#include "kernel_private_common.cuh"


template <class  T>
class byte_reader_buffers {
public:
    __device__ byte_reader_buffers(const T* input, size_t size_)
        : top(input), size(size_), local_offset(0)
    {};

    __device__ ~byte_reader_buffers() {};

    __device__ T getLocal(size_t offset) {
        const T *val = top + offset + local_offset;
        return *val;
    };

    __device__ const T* getLocalAddress(size_t offset) {
        return top + offset + local_offset;
    };

    __device__ bool end() {
        return (local_offset >= size);
    };

    __device__ void add_offset(int count) {
        local_offset += count;
    };

    __device__ int get_read_count() {
        return local_offset;
    };

protected:
    const T* top;
    size_t size;

    size_t local_offset;
};


#endif // __ORC_KERNEL_READER_BUFFERS_H__
