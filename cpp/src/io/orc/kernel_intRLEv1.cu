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
#include "kernel_reader_int.cuh"

template <class T, class T_writer, class T_reader>
class ORCdecode_IntRLEv1_single : public ORCdecodeCommon<T, T_writer, T_reader>
{
public:
    __device__ ORCdecode_IntRLEv1_single(KernelParamCommon* param)
        : ORCdecodeCommon<T, T_writer, T_reader>(param)
    {};

    __device__ ~ORCdecode_IntRLEv1_single() {};

    __device__ void decode();

protected:
    __device__ void decodeRun(int length);
    __device__ void decodeLiteral(int length);
};


template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_IntRLEv1_single<T, T_writer, T_reader>::decode()
{
    while (!this->reader.end() && !this->writer.end()) {
        orc_byte h = this->reader.getLocal(0);
        bool is_literals = h & 0x80;
        int length = (is_literals) ? 256 - (h) : int(h) + 3;

        if (!is_literals) {    // Run - a sequence of at least 3 identical values
            decodeRun(length);
        }
        else {                // Literals - a sequence of non-identical values
            decodeLiteral(length);
        }
    }
}

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_IntRLEv1_single<T, T_writer, T_reader>::decodeRun(int length)
{
    orc_sint8 delta = orc_sint8(this->reader.getLocal(1));    // single byte delta in range [-128, 127].
    T BaseValue = 0;    // Base 128 Varint
    int count = this->reader.getVarint128(&BaseValue, 2);
    this->reader.add_offset(2 + count);

    for (int i = 0; i < length; i++) {
        this->writer.expect(1);
        this->writer.write_local(BaseValue + delta * i, 0);
        this->writer.add_offset(1);
    }
}

template <class T, class T_writer, class T_reader>
__device__ void ORCdecode_IntRLEv1_single<T, T_writer, T_reader>::decodeLiteral(int length)
{
    this->reader.add_offset(1);

    while (length--) {
        T BaseValue = 0;    // Base 128 Varint
        int count = this->reader.getVarint128(&BaseValue, 0);
        this->reader.add_offset(count);

        //                DP0_INT(value);

        this->writer.expect(1);
        this->writer.write_local(BaseValue, 0);
        this->writer.add_offset(1);
    }
}

template <class T_decoder>
__global__ void kernel_integerRLEv1(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_integerRLEv1_entry(KernelParamCommon* param)
{
    const int num_threads = 1;

    kernel_integerRLEv1<ORCdecode_IntRLEv1_single<T, T_writer, T_reader>> << <1, num_threads >> > (*param);
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_integerRLEv1_reader_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader_int<T, data_reader_single_buffer>;
        cuda_integerRLEv1_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader_int<T, data_reader_multi_buffer>;
        cuda_integerRLEv1_entry<T, T_writer, reader>(param);
    }
}

template <typename T, class T_converter = ORCConverterBase<T>>
void cuda_integerRLEv1_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_integerRLEv1_reader_select<T, data_writer_depends_single<T, T_converter>>(param);
    }
    else {
        cuda_integerRLEv1_reader_select<T, data_writer<T, T_converter>>(param);
    }
}

template <typename T>
void cuda_integerRLEv1_converter_select(KernelParamCommon* param)
{
    switch (param->convertType) {
    case OrcKernelConvertionType::GdfDate64:
        cuda_integerRLEv1_writer_select<T, ORCConverterGdfDate64<T>>(param);
        break;
    case OrcKernelConvertionType::GdfConvertNone:
        cuda_integerRLEv1_writer_select<T, ORCConverterBase<T>>(param);
        break;
    default:
        EXIT("unhandled convert type");
        break;
    }

}

void cuda_integerRLEv1_Depends(KernelParamCommon* param)
{
    switch (param->elementType) {
    case OrcElementType::Uint64:
        cuda_integerRLEv1_converter_select<orc_uint64>(param);
        break;
    case OrcElementType::Sint64:
        cuda_integerRLEv1_converter_select<orc_sint64>(param);
        break;
    case OrcElementType::Uint32:
        cuda_integerRLEv1_converter_select<orc_uint32>(param);
        break;
    case OrcElementType::Sint32:
        cuda_integerRLEv1_converter_select<orc_sint32>(param);
        break;
    case OrcElementType::Uint16:
        cuda_integerRLEv1_converter_select<orc_uint16>(param);
        break;
    case OrcElementType::Sint16:
        cuda_integerRLEv1_converter_select<orc_sint16>(param);
        break;
    default:
        EXIT("unhandled type");
        break;
    }

}

