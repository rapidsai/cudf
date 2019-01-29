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

#ifndef __ORC_KERNEL_WRITER_DEPENDS_H__
#define __ORC_KERNEL_WRITER_DEPENDS_H__

#include "kernel_writer.cuh"


template <class T, class T_present_reader = byte_reader_present_warp, class T_converter = ORCConverterBase<T>>
class data_writer_depends : public data_writer<T, T_converter> {
public:
    __device__ data_writer_depends(KernelParamCommon* param)
        : data_writer<T, T_converter>(param)
        , present(reinterpret_cast<const orc_bitmap*>(param->present), param->output_count, param->start_id, param->output_count)
        , written_count(0)
    {};

    //! this function must be called by all thread in the warp
    //! count is the total number (count of threads) to call write_local() after expect()
    __device__ void expect(int count) {
        next_index = present.expect(count); // next index is the absolute index
    };

    __device__ void write_local(T data, size_t offset) {
        // the offset value is ignored at here
        // next index must be calculated by expect()
        write(data, this->local_offset + next_index);
    };

    __device__ void add_offset(int count) {
        written_count += count;
        this->local_offset += present.getExpectRange();
    };

    __device__ int get_decoded_count() {
        return written_count;
    };

protected:
    T_present_reader     present;    //< present stream input
    size_t               next_index;
    size_t               written_count; 
};

// type alias for single threaded byte_reader_present_warp
template<class T, class T_converter = ORCConverterBase<T>>
using data_writer_depends_single = data_writer_depends<T, byte_reader_present_single, T_converter>;

template<class T, class T_converter = ORCConverterBase<T>>
using data_writer_depends_warp = data_writer_depends<T, byte_reader_present_warp, T_converter>;


#endif // __ORC_KERNEL_WRITER_DEPENDS_H__
