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

namespace kernel_booleanRLE_CudaOrcKernelRetStats
{
    enum ORCCodePath {
        RepeatFull = 0,
        RepeatRest,
        LiteralFull,
        LiteralRest,
        ORCCodePath_Max,
    };
};

using namespace kernel_booleanRLE_CudaOrcKernelRetStats;

__device__ orc_bitmap applymask(orc_bitmap& bitmap, orc_bitmap mask)
{
    orc_bitmap result = 0;
    int count = 0;
    for (int i = 0; i < 8; i++) {
        int flag = ((mask >> i) & 0x01);
        if (flag) {
            result += (((bitmap >> count) & 0x01) << i);
            count++;
        }
    }
    return result;
}


// ------------------------------------------------------------------------------
// this is worked as 130 threads kernel.
// required no parent, start_id
// this is the most standard case to decoding.
template <class T_reader>
class ORCdecodePresentFullthrottle
{
public:
    __device__ ORCdecodePresentFullthrottle(KernelParamBitmap* param);
    __device__ ~ORCdecodePresentFullthrottle() {};

    __device__ virtual void decode();

protected:
    __device__ virtual void decodeRun(int length);
    __device__ virtual void decodeLiteral(int length);

protected:
    T_reader                reader;
    bitmap_writer           writer;
    CudaOrcKernelRetStats   stat;
};

template <class T_reader>
__device__ ORCdecodePresentFullthrottle<T_reader>::ORCdecodePresentFullthrottle(KernelParamBitmap* param)
    : reader(param)
    , writer(param)
    , stat(param->stat)
{
}

template <class T_reader>
__device__ void ORCdecodePresentFullthrottle<T_reader>::decode()
{
    while (!reader.end() && !writer.end()) {
        orc_byte h = reader.getLocal(0);
        bool is_literals = h & 0x80;
        int length = (is_literals) ? 256 - (h) : int(h) + 3;

        if (!is_literals) {    // Run - a sequence of at least 3 identical values
            decodeRun(length);
            DP0("Present, run: %d\n", length);
            stat.IncrementModeCount(RepeatFull);
        }
        else {                // Literals - a sequence of non-identical values
            decodeLiteral(length);
            DP0("Present, literal: %d\n", length);
            stat.IncrementModeCount(LiteralRest);
        }
    }

    // todo: 
//    writer.fill_rest(0x00);
}

template <class T_reader>
__device__ void ORCdecodePresentFullthrottle<T_reader>::decodeRun(int length)
{
    orc_byte v = reader.getLocal(1);
    v = getBitOrderFlip(v);
    reader.add_offset(2);

    if (threadIdx.x < length) {
        writer.write_local(v, threadIdx.x);
    }

    writer.add_offset(length);
}

template <class T_reader>
__device__ void ORCdecodePresentFullthrottle<T_reader>::decodeLiteral(int length)
{
    reader.add_offset(1);

    if (threadIdx.x < length) {
        orc_byte v = reader.getLocal(threadIdx.x);
        v = getBitOrderFlip(v);
        writer.write_local(v, threadIdx.x);
    }
    writer.add_offset(length);
    reader.add_offset(length);
}


// ------------------------------------------------------------------------------
template <class T_reader>
class ORCdecodePresentCarriedSingle : public ORCdecodePresentFullthrottle<T_reader>
{
public:
    __device__ ORCdecodePresentCarriedSingle(KernelParamBitmap* param);
    __device__ ~ORCdecodePresentCarriedSingle() {};

    __device__ virtual void decode();

protected:
    __device__ virtual void decodeRun(int length);
    __device__ virtual void decodeLiteral(int length);

protected:
    orc_bitmap bitmap;            //< the bitmap can be separated into 3. if id_offset == 0, bitmap is 1.
    orc_bitmap carriedValue;    
    orc_bitmap firstValue;

    int carried_bit_count;
};

template <class T_reader>
__device__ ORCdecodePresentCarriedSingle<T_reader>::ORCdecodePresentCarriedSingle(KernelParamBitmap* param)
    : ORCdecodePresentFullthrottle<T_reader>(param)
    , carriedValue(0)
    , carried_bit_count(param->start_id)
{
}

template <class T_reader>
__device__ void ORCdecodePresentCarriedSingle<T_reader>::decode()
{
    while (!this->reader.end() && !this->writer.end()) {
        orc_byte h = this->reader.getLocal(0);
        bool is_literals = h & 0x80;
        int length = (is_literals) ? 256 - (h) : int(h) + 3;

        if (!is_literals) {    // Run - a sequence of at least 3 identical values
            decodeRun(length);
            this->stat.IncrementModeCount(RepeatFull);
        }
        else {                // Literals - a sequence of non-identical values
            decodeLiteral(length);
            this->stat.IncrementModeCount(LiteralRest);
        }
    }

    if (carried_bit_count) {
        this->writer.write_local(carriedValue, 0);
        this->writer.add_offset(1);
    }

    this->writer.fill_rest(0x00);
}

template <class T_reader>
__device__ void ORCdecodePresentCarriedSingle<T_reader>::decodeRun(int length)
{
    orc_byte v = this->reader.getLocal(1);
    v = getBitOrderFlip(v);
//    DP0("Present, run: %d, %d\n", length, v);

    bitmap = ( v << carried_bit_count);
    firstValue = carriedValue | bitmap;
    carriedValue = ( v >> (8 - carried_bit_count) );
    bitmap |= carriedValue;
    this->reader.add_offset(2);

    if (carried_bit_count) {
        this->writer.write_local_single(firstValue);
        length--;
    }

    while (length--) {
        this->writer.write_local(bitmap, 0);
        this->writer.add_offset(1);
    }
}

template <class T_reader>
__device__ void ORCdecodePresentCarriedSingle<T_reader>::decodeLiteral(int length)
{
    this->reader.add_offset(1);
    if (carried_bit_count) {
        orc_byte v = this->reader.getLocal(0);
        v = getBitOrderFlip(v);
        bitmap = (v << carried_bit_count);
        firstValue = carriedValue | bitmap;
        carriedValue = (v >> (8 - carried_bit_count));

        this->writer.write_local_single(firstValue);
        length--;
        this->reader.add_offset(1);
    }


    while (length--) {
        orc_byte v = this->reader.getLocal(0);
        v = getBitOrderFlip(v);

        bitmap =  (v << carried_bit_count) | carriedValue;
        carriedValue = (v >> (8 - carried_bit_count));

        this->writer.write_local(bitmap, 0);
        this->writer.add_offset(1);
        this->reader.add_offset(1);
    }
}

// ------------------------------------------------------------------------------------------------------------------------
template <class T_reader>
class ORCdecodePresentDependsSingle : public ORCdecodePresentCarriedSingle<T_reader>
{
public:
    __device__ ORCdecodePresentDependsSingle(KernelParamBitmap* param)
        : ORCdecodePresentCarriedSingle<T_reader>(param)
        , parent(param->parent, param->output_count, param->start_id, 0)
    {};

protected:
    __device__ virtual void decodeRun(int length);
    __device__ virtual void decodeLiteral(int length);

protected:
    byte_reader_bitmap<orc_bitmap>    parent;
};

template <class T_reader>
__device__ void ORCdecodePresentDependsSingle<T_reader>::decodeRun(int length)
{
    orc_byte v = this->reader.getLocal(1);    // the value to be repeated
    this->reader.add_offset(2);
    v = getBitOrderFlip(v);

    do {
        orc_bitmap mask = this->parent.getLocal(0);
        int bit_count = getBitCount(mask);

        if (bit_count == 0) {
            this->writer.write_local(0x00, 0);
            this->writer.add_offset(1);
            this->parent.add_offset(1);
            continue;
        }

        orc_bitmap bitmap = this->carriedValue | (v << this->carried_bit_count);
        if (bit_count < 8) bitmap = applymask(bitmap, mask);

        if (bit_count > this->carried_bit_count) {
            length--;
            this->carried_bit_count = 8 + this->carried_bit_count - bit_count;    // carried_bit_count must be [0,7]
        }
        else {
            this->carried_bit_count = this->carried_bit_count - bit_count;
        }
        this->carriedValue = (v >> (8 - this->carried_bit_count));

        this->writer.write_local(bitmap, 0);
        this->writer.add_offset(1);
        this->parent.add_offset(1);
    } while (length);

}

template <class T_reader>
__device__ void ORCdecodePresentDependsSingle<T_reader>::decodeLiteral(int length)
{
    this->reader.add_offset(1);

    do {
        orc_bitmap mask = this->parent.getLocal(0);
        int bit_count = getBitCount(mask);

        if (bit_count == 0) {
            this->writer.write_local(0x00, 0);
            this->writer.add_offset(1);
            this->parent.add_offset(1);
            continue;
        }

        orc_bitmap bitmap;
        if (bit_count <= this->carried_bit_count) {
            bitmap = this->carriedValue;

            this->carried_bit_count -= bit_count;
            this->carriedValue >>= bit_count;

        }else{
            orc_byte v = this->reader.getLocal(0);
            v = getBitOrderFlip(v);
            this->reader.add_offset(1);
            length--;

            bitmap = this->carriedValue | (v << this->carried_bit_count);

            this->carried_bit_count = 8 + this->carried_bit_count - bit_count;    // carried_bit_count must be [0,7]
            this->carriedValue = (v >> (8 - this->carried_bit_count));
        }

        bitmap = applymask(bitmap, mask);
        this->writer.write_local(bitmap, 0);
        this->writer.add_offset(1);
        this->parent.add_offset(1);
    } while (length);
}


// ------------------------------------------------------------------------------------------------------------------------

template <class T_decoder>
__global__ void kernel_Present(KernelParamBitmap param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_present_entry(KernelParamBitmap* param)
{
    int numThreads = 1;

    if (param->parent) {
        using decoder = ORCdecodePresentDependsSingle<T_reader>;
        kernel_Present<decoder> << <1, numThreads >> > (*param);
    }
    else {
        if (param->start_id == 0) {
            numThreads = 130;   // := 127 + 3, the maximum length of run or literal encodings, 
            using decoder = ORCdecodePresentFullthrottle<T_reader>;
            kernel_Present<decoder> << <1, numThreads >> > (*param);
        }
        else {
            using decoder = ORCdecodePresentCarriedSingle<T_reader>;
            kernel_Present<decoder> << <1, numThreads >> > (*param);
        }
    }
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_present_reader_select(KernelParamBitmap* param)
{
    if (param->input) {
        using reader = stream_reader<data_reader_single_buffer>;
        cuda_present_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader<data_reader_multi_buffer>;
        cuda_present_entry<T, T_writer, reader>(param);
    }
}

template <typename T>
void cuda_present_writer_select(KernelParamBitmap* param)
{
    if (param->parent) {
        cuda_present_reader_select<T, data_writer_depends_single<T>>(param);
    }
    else {
        cuda_present_reader_select<T, data_writer<T>>(param);
    }
}

void cuda_booleanRLEbitmapDepends(KernelParamBitmap* param)
{
    cuda_present_writer_select<orc_bitmap>(param);
}

