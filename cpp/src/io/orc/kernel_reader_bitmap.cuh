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

#ifndef __ORC_KERNEL_READER_BITMAP_H__
#define __ORC_KERNEL_READER_BITMAP_H__

#include "kernel_reader.cuh"


template <class  T>
class byte_reader_bitmap : public byte_reader<T> {
public:
    __device__ byte_reader_bitmap(const T* bitmap, size_t size_, int start_offset_, int end_count_)
        : byte_reader<T>(bitmap, size_), local_input_offset(0), local_output_offset(0)
    {
        byte_reader<T>::size = (end_count_ >> 8);
    };

    __device__ ~byte_reader_bitmap() {};

protected:
    int start_offset;
    int end_offset;
    int local_input_offset;        // offset for the input index.
    int local_output_offset;    // offset for the output index.
};



#endif // __ORC_KERNEL_READER_BITMAP_H__
