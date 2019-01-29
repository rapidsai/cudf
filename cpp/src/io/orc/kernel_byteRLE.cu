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

#include "kernel_decode_common.cuh"

template <class T, class T_writer, class T_reader>
class ORCdecode_Byte_warp : public ORCdecodeCommon<T, T_writer, T_reader>
{
public:
    __device__ ORCdecode_Byte_warp(KernelParamCommon* param)
        : ORCdecodeCommon<T, T_writer, T_reader>(param)
    {};

    __device__ ~ORCdecode_Byte_warp() {};

    __device__ void decode();

protected:
    __device__ void decodeRun(int length);
    __device__ void decodeLiteral(int length);
};

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Byte_warp<T, T_writer, T_reader>::decode()
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

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Byte_warp<T, T_writer, T_reader>::decodeRun(int length)
{
    orc_byte value = this->reader.getLocal(1);
    this->reader.add_offset(2);

    for (int i = 0; i < this->ctc.getFullCount(length); i++) {
        this->writer.expect(blockDim.x);
        this->writer.write_local(value, this->ctc.tid);
        this->writer.add_offset(blockDim.x);
    }

    int the_rest = this->ctc.getRestCount(length);
    this->writer.expect(the_rest);
    if (this->ctc.tid < the_rest) {
        this->writer.write_local(value, this->ctc.tid);
    }

    this->writer.add_offset(the_rest);
}

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_Byte_warp<T, T_writer, T_reader>::decodeLiteral(int length)
{
    this->reader.add_offset(1);

    for (int i = 0; i < this->ctc.getFullCount(length); i++) {
        orc_byte value = this->reader.getLocal(this->ctc.tid);

        this->writer.expect(blockDim.x);
        this->writer.write_local(value, this->ctc.tid);
        this->writer.add_offset(blockDim.x);
        this->reader.add_offset(blockDim.x);
    }

    int the_rest = this->ctc.getRestCount(length);
    this->writer.expect(the_rest);
    if (this->ctc.tid < the_rest) {
        orc_byte value = this->reader.getLocal(this->ctc.tid);

        this->writer.write_local(value, this->ctc.tid);
    }

    this->writer.add_offset(the_rest);
    this->reader.add_offset(the_rest);

}

template <class T_decoder>
__global__ void kernel_byteRLE(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_byteRLE_entry(KernelParamCommon* param)
{
    const int num_threads = 32;
    kernel_byteRLE<ORCdecode_Byte_warp<T, T_writer, T_reader>> << <1, num_threads >> > (*param);
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_byteRLE_reader_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader<data_reader_single_buffer>;
        cuda_byteRLE_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader<data_reader_multi_buffer>;
        cuda_byteRLE_entry<T, T_writer, reader>(param);
    }
}

template <typename T>
void cuda_byteRLE_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_byteRLE_reader_select<T, data_writer_depends<T>>(param);
    }
    else {
        cuda_byteRLE_reader_select<T, data_writer<T>>(param);
    }
}

void cuda_ByteRLEDepends(KernelParamCommon* param)
{
    cuda_byteRLE_writer_select<orc_byte>(param);
}

