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

//#define _DEBUG
//#define _INVESTIGATION  // debug use

#include "kernel_private.cuh"
#include "kernel_reader_element.cuh"

namespace cudf {
namespace orc {

template <class T, class T_writer, class T_reader>
class ORCdecodeRawDataWarp
{
public:
    __device__ ORCdecodeRawDataWarp(KernelParamCommon* param)
        : reader(param)
        , writer(param)
        , stat(param->stat)
    {
        length = param->input_size / param->elementSize;
    };

    __device__ ~ORCdecodeRawDataWarp() {};

    __device__ void decode();

protected:
    T_reader                reader;
    T_writer                writer;

    CudaThreadControl       ctc;
    CudaOrcKernelRetStats   stat;

    // for raw data stream, the length used for decoding is the count of the elements of input stream
    // input_count is not always output_count of KernelParamCommon if present stream is provided.
    size_t length;  
};

template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeRawDataWarp<T, T_writer, T_reader>::decode()
{
    T value;
    for (int i = 0; i < this->ctc.getFullCount(length); i++) {
        value = this->reader.getLocalElement(this->ctc.tid);
        this->writer.expect(blockDim.x);
        this->writer.write_local(value, this->ctc.tid);

        this->writer.add_offset(blockDim.x);
        this->reader.add_offset(blockDim.x);
    }

    this->writer.expect(this->ctc.getRestCount(length));
    if (this->ctc.tid < this->ctc.getRestCount(length)) {
        value = this->reader.getLocalElement(this->ctc.tid);
        this->writer.write_local(value, this->ctc.tid);
    }

    this->writer.add_offset(this->ctc.getRestCount(length));
    this->reader.add_offset(this->ctc.getRestCount(length));
}

template <class T_decoder>
__global__ void kernel_raw_data_depends(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_raw_data_entry(KernelParamCommon* param)
{
    const int num_CTA = 32;
    kernel_raw_data_depends<ORCdecodeRawDataWarp<T, T_writer, T_reader>> << <1, num_CTA, 0, param->stream >> > (*param);
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer, int elemenet_size>
void cuda_raw_data_reader_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader_element<T, data_reader_single_buffer, elemenet_size>;
        cuda_raw_data_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader_element<T, data_reader_multi_buffer, elemenet_size>;
        cuda_raw_data_entry<T, T_writer, reader>(param);
    }
}

template <typename T, int elemenet_size>
void cuda_raw_data_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_raw_data_reader_select<T, data_writer_depends<T>, elemenet_size>(param);
    }
    else {
        cuda_raw_data_reader_select<T, data_writer<T>, elemenet_size>(param);
    }
}

void cudaDecodeRawData(KernelParamCommon* param)
{
    switch (param->elementType) {
    case OrcElementType::Float32:
        param->elementSize = 4;
        // the element type is actually orc_float32, 
        // unsinged element type since the class requires since the class treat byte stream using bit shift
        cuda_raw_data_writer_select<orc_uint32, 4>(param);   
        break;
    case OrcElementType::Float64:
        param->elementSize = 8;
        // the element type is set to orc_uint64 as well as Float32.
        cuda_raw_data_writer_select<orc_uint64, 8>(param);
        break;
    default:
        param->elementSize = 1;
        cuda_raw_data_writer_select<orc_byte, 1>(param);
        break;
    }
}

}   // namespace orc
}   // namespace cudf