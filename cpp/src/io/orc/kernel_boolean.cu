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

namespace cudf {
namespace orc {

namespace kernel_booleanRLE_CudaOrcKernelRetStats {

    enum ORCCodePath {
        RepeatFull = 0,
        RepeatRest,
        LiteralFull,
        LiteralRest,
        ORCCodePath_Max,
    };
};

template <class T, class T_writer, class T_reader>
class ORCdecode_Boolean_warp : public ORCdecodeCommon<T, T_writer, T_reader>
{
public:
    __device__ ORCdecode_Boolean_warp(KernelParamCommon* param)
        : ORCdecodeCommon<T, T_writer, T_reader>(param)
    {};

    __device__ ~ORCdecode_Boolean_warp() {};

    __device__ void decode();

protected:
    __device__ void decodeRun(int length);
    __device__ void decodeLiteral(int length);
};



template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Boolean_warp<T, T_writer, T_reader>::decode()
{
    while (!this->reader.end() && !this->writer.end()) {
        orc_byte h = this->reader.getLocal(0);
        bool is_literals = h & 0x80;
        int length = (is_literals) ? 256 - (h) : int(h) + 3;

        if (!is_literals) {    // Run - a sequence of [3, 127+3] identical values
            decodeRun(length);
            this->stat.IncrementModeCount(RLE_Repeat);
        }
        else {                // Literals - a sequence of non-identical values
            decodeLiteral(length);
            this->stat.IncrementModeCount(RLE_Literal);
        }
    }

    this->stat.SetDecodeCount(this->writer.get_decoded_count());
    this->stat.SetReadCount(this->reader.get_read_count());
    this->stat.SetReturnCode(0);
    this->stat.Output();
}

__device__
orc_byte GetBooleanFromBitmap(orc_byte origin, int tid)
{
    orc_byte value = ((origin >> (7 - (tid & 0x07))) & 0x01);
    return value;
}


template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Boolean_warp<T, T_writer, T_reader>::decodeRun(int length)
{
    orc_byte origin = this->reader.getLocal(1);
    this->reader.add_offset(2);

    orc_byte value = GetBooleanFromBitmap(origin, this->ctc.tid);

    for (int i = 0; i < this->ctc.getFullCount(length*8); i++) {
        this->writer.expect(blockDim.x);
        this->writer.write_local(value, this->ctc.tid);
        this->writer.add_offset(blockDim.x);
    }

    int the_rest = this->ctc.getRestCount(length * 8);
    this->writer.expect(the_rest);
    if (this->ctc.tid < the_rest) {
        this->writer.write_local(value, this->ctc.tid);
    }

    this->writer.add_offset(the_rest);
}

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Boolean_warp<T, T_writer, T_reader>::decodeLiteral(int length)
{
    this->reader.add_offset(1);

    for (int i = 0; i < this->ctc.getFullCount(length * 8); i++) {
        orc_byte origin = this->reader.getLocal(this->ctc.tid >> 3);
        orc_byte value = GetBooleanFromBitmap(origin, this->ctc.tid);

        this->writer.expect(blockDim.x);
        this->writer.write_local(value, this->ctc.tid);
        this->writer.add_offset(blockDim.x);
        this->reader.add_offset(blockDim.x >> 3);
    }

    int the_rest = this->ctc.getRestCount(length * 8);
    this->writer.expect(the_rest);
    if (this->ctc.tid < the_rest) {
        orc_byte origin = this->reader.getLocal(this->ctc.tid >> 3);
        orc_byte value = GetBooleanFromBitmap(origin, this->ctc.tid);

        this->writer.write_local(value, this->ctc.tid);
    }

    this->writer.add_offset(the_rest);
    this->reader.add_offset(the_rest >> 3);

}

template <class T_decoder>
__global__ void kernel_booleanRLE(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_booleanRLE_entry(KernelParamCommon* param)
{
    const int num_threads = 32;
    kernel_booleanRLE<ORCdecode_Boolean_warp<T, T_writer, T_reader>> << <1, num_threads, 0, param->stream >> > (*param);

    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_booleanRLE_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader<data_reader_single_buffer>;
        cuda_booleanRLE_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader<data_reader_multi_buffer>;
        cuda_booleanRLE_entry<T, T_writer, reader>(param);
    }
}

template <typename T>
void cuda_booleanRLE_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_booleanRLE_select<T, data_writer_depends<T>>(param);
    }
    else {
        cuda_booleanRLE_select<T, data_writer<T>>(param);
    }
}

void cudaDecodeBooleanRLE(KernelParamCommon* param)
{
    cuda_booleanRLE_writer_select<orc_byte>(param);
}

}   // namespace orc
}   // namespace cudf

