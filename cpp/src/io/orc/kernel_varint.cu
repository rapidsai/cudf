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

#include "kernel_private.cuh"
#include "kernel_reader_int.cuh"

namespace cudf {
namespace orc {

#define USE_DEPEND

template <class T, class T_writer, class T_reader>
class ORCdecodeVarintSingle : public ORCdecodeCommon<T, T_writer, T_reader>
{
public:
    __device__ ORCdecodeVarintSingle(KernelParamCommon* param)
        : ORCdecodeCommon<T, T_writer, T_reader>(param)
    {};

    __device__ ~ORCdecodeVarintSingle() {};

    __device__ void decode();

};

template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeVarintSingle<T, T_writer, T_reader>::decode()
{
    while (!this->reader.end() && !this->writer.end()) {
        T BaseValue = 0;    // Base 128 Varint
        int count = this->reader.getVarint128(&BaseValue, 0);
        //    DP0_INT(value);

        this->writer.expect(1);
        this->writer.write_local(BaseValue, 0);
        this->writer.add_offset(1);
        this->reader.add_offset(count);
    }
}


template <class T_decoder>
__global__ void kernel_base128_varint(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_base128_varint_entry(KernelParamCommon* param)
{
    const int num_CTA = 1;
    kernel_base128_varint<ORCdecodeVarintSingle<T, T_writer, T_reader>> << <1, num_CTA, 0, param->stream >> > (*param);
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_base128_varint_reader_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader_int<T, data_reader_single_buffer>;
        cuda_base128_varint_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader_int<T, data_reader_multi_buffer>;
        cuda_base128_varint_entry<T, T_writer, reader>(param);
    }
}

template <typename T>
void cuda_base128_varint_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_base128_varint_reader_select<T, data_writer_depends_single<T>>(param);
    }
    else {
        cuda_base128_varint_reader_select<T, data_writer<T>>(param);
    }
}

void cudaDecodeVarint(KernelParamCommon* param)
{
    switch (param->elementType) {
    case OrcElementType::Uint64:
        cuda_base128_varint_writer_select<orc_uint64>(param);
        break;
    case OrcElementType::Sint64:
        cuda_base128_varint_writer_select<orc_sint64>(param);
        break;
    default:
        EXIT("unhandled type");
    }
}

}   // namespace orc
}   // namespace cudf