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

#ifndef __ORC_KERNEL_READER_ELEMENT_H__
#define __ORC_KERNEL_READER_ELEMENT_H__

#include "kernel_private_common.cuh"

namespace cudf {
namespace orc {

template <class TU, class  T_reader_input, int element_size>
class stream_reader_element : public stream_reader<T_reader_input> {
public:
    __device__ stream_reader_element(const KernelParamCommon* kernParam)
        : stream_reader<T_reader_input>(reinterpret_cast<const KernelParamBase*>(kernParam))
    {};

    __device__ ~stream_reader_element() {};

    __device__ TU getLocalElement(size_t offset) {
        TU ret = 0;
        int top_offset = offset * element_size;
        for (int i = 0; i < element_size; i++) {
            ret <<= 8;
            // becase the top_offset can be not aligned by the alignmnet of TU, 
            // the stream accesses each byte of the element, or hit alignment violation
            ret += this->getLocal(top_offset + element_size - 1 - i);
        }
        return ret;
    };

    __device__ void add_offset(int count) {
        this->reader.add_offset(count * element_size);
    };

protected:
};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_READER_ELEMENT_H__
