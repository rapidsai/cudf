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

#ifndef __ORC_KERNEL_WRITER_BITMAP_H__
#define __ORC_KERNEL_WRITER_BITMAP_H__

#include "kernel_writer.cuh"

namespace cudf {
namespace orc {

class bitmap_writer {
public:
    __device__ bitmap_writer(KernelParamBitmap* param) :
        top(reinterpret_cast<orc_bitmap*>(param->output)), start_offset(param->start_id), local_offset(0)
    {
        size = param->output_count;
    };

    __device__ ~bitmap_writer() {};

    __device__ void write_safe(orc_bitmap byte, size_t index) {
        if (index == 0 && start_offset != 0) {
            top[index] += byte;
        }
        else {
            top[index] = byte;
        }
    }

    __device__ void write_local_single(orc_bitmap byte ){
        if (threadIdx.x == 0) write_safe(byte, local_offset);
        local_offset++;
    }

    __device__ void write(orc_bitmap byte, size_t index) {
        top[index] = byte;
    }

    __device__ void write_local(orc_bitmap byte, size_t offset) {
        write(byte, local_offset + offset);
    }

    __device__ void add_offset(int count) {
        local_offset += count;
    }

    __device__ bool end() {
        return (local_offset >= size);
    }

    __device__ int current_bit_offset() {
        return start_offset;
    }

    __device__ void fill_rest(orc_bitmap value)
    {

    }

private:
    orc_bitmap* top;
    size_t size;

    int start_offset;
    int end_offset;
    int local_offset;    // offset for the index.

};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_WRITER_BITMAP_H__
