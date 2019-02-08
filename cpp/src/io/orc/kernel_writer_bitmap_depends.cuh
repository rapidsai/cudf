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

#ifndef __ORC_KERNEL_WRITER_BITMAP_DEPENDS_H__
#define __ORC_KERNEL_WRITER_BITMAP_DEPENDS_H__

#include "kernel_writer_bitmap.cuh"
#include "kernel_reader_bitmap.cuh"

namespace cudf {
namespace orc {

template <class  T>
class bitmap_writer_depends : public bitmap_writer {
public:
    __device__ bitmap_writer_depends(KernelParamBitmap* param)
        : bitmap_writer(param), input(param->parent, param->input_size, param->start_id, param->input_size)
    {};

    __device__ ~bitmap_writer_depends() {};

protected:
    byte_reader_bitmap<T> input;

};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_WRITER_BITMAP_DEPENDS_H__
