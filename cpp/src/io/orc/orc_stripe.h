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

#ifndef __ORC_STRIPE_HEADER__
#define __ORC_STRIPE_HEADER__

#include "orc_read.h"
#include "orc_debug.h"
#include "orc_util.hpp"
#include "kernel_orc.cuh"
#include "orc_memory.h"
#include <vector>

class OrcStripeArguemnts;
class OrcStreamArguemnts;
struct OrcColumnArguemnts;

/** ---------------------------------------------------------------------------*
* @brief parameters for a column in a stripe
* ---------------------------------------------------------------------------**/
struct OrcColumnArguemnts {
    OrcColumnArguemnts()
        : stream_present(NULL), stream_data(NULL), stream_second(NULL)
        , valid_parent_present(NULL)
        , temporary_buffer(NULL)
    {};

    ~OrcColumnArguemnts()
    {
        releaseTemporaryBuffer();
    };

    void releaseTemporaryBuffer()
    {
        if (temporary_buffer) {
            ReleaseTemporaryBufffer(temporary_buffer, 0);
            temporary_buffer = NULL;
        }
    };

    OrcTypeInfo* type;                          //< pointer for the type info, type is constant value over all stripes

    // these values will be different at each stripe
    ORCColumnEncodingKind encoding;             //< kind of column encoding for the column of the stripe.
    orc_uint32 dictionary_size;                 //< size of dictionary for the column of the stripe.
    orc_uint32 bloom_encoding;                  //< encoding of bloom filter for the column of the stripe.

    OrcStreamArguemnts* stream_present;         //< present stream.
    OrcStreamArguemnts* stream_data;            //< data stream.
    OrcStreamArguemnts* stream_second;          //< second stream.
    OrcStreamArguemnts* stream_length;          //< length stream.
    OrcStreamArguemnts* stream_dictionary_data; //< dictionary data stream.

    OrcStreamArguemnts* valid_parent_present;   //< pointer to the valid parent's present stream.
    int valid_parent_id;                        //< id fo valid parent

    void* temporary_buffer;

    OrcBufferHolder holder;
};

/** ---------------------------------------------------------------------------*
* @brief parameters for decode a stream, this value will be kept until the stripe is discarded.
* ---------------------------------------------------------------------------**/
class OrcStreamArguemnts {
public:
    OrcStreamArguemnts(){};
    ~OrcStreamArguemnts() {};

    void SetArgs(const OrcStripeArguemnts* stripe, ORCStreamKind kind, const OrcColumnArguemnts* column, int colunm_id)
    {
        _stripe = stripe;
        _kind = kind;
        _column = column;
        _colunm_id = colunm_id;
    };

    void SetSource(const orc_byte* cpu_addr, const orc_byte* gpu_addr, size_t length)
    {
        src_addr_cpu = cpu_addr;
        src_addr_gpu = gpu_addr;
        src_length = length;
    }

    void SetTarget(orc_byte* gpu_addr, size_t start_index, size_t count, int element_size)
    {
        dst_addr_gpu_top = gpu_addr;
        dst_addr_gpu_stream = gpu_addr + start_index * element_size;
        dst_length = count * element_size;
        dst_element_count = count;

        dst_start_count = start_index;

        present_start_index = start_index & 0x07;
    };

    // set target stream for present stream
    void SetPresentTarget(orc_byte* gpu_addr, size_t start_offset, size_t count)
    {
        dst_addr_gpu_top = gpu_addr;
        dst_addr_gpu_stream = gpu_addr + (start_offset >> 3);

        present_start_index = start_offset & 0x07;
        present_end_index = (start_offset + count) & 0x07;

        dst_element_count = count;
        dst_length = ((present_start_index + count + 7) >> 3);
    };

    void GetKernelParamBitmap(KernelParamBitmap* param)
    {
        param->output = dst_addr_gpu_stream;
        param->input = src_addr_gpu;
        param->parent = (_column->valid_parent_present) ? _column->valid_parent_present->target() : NULL;
        param->input_size = src_length;
        param->start_id = present_start_index;
        param->output_count = dst_length;
        param->stat = NULL;
    }

    void GetKernelParamCommon(KernelParamCommon* param)
    {
        param->output = dst_addr_gpu_stream;
        param->output_count = dst_length;

        param->input = src_addr_gpu;
        param->input_size = src_length;
        param->present = (_column->stream_present) ? _column->stream_present->target() : NULL;
        param->start_id = present_start_index;

        param->elementType = _column->type->elementType;
        param->stat = NULL;
    }

public: // accessors
    // the app have to ensure atomic write for the start byte if presentStartIndex != 0
    int presentStartIndex() { return present_start_index; };
    int presentEndIndex() { return present_end_index; };

    const OrcStripeArguemnts* stripe() { return _stripe; };
    const OrcColumnArguemnts* column() { return _column; };
    ORCStreamKind kind() { return _kind; };

    orc_byte* target() { return dst_addr_gpu_stream; };
    const orc_byte* source() { return src_addr_gpu; };
    const orc_byte* sourceCpuAddr() { return src_addr_cpu; };

    size_t sourceSize() { return src_length; };
    size_t targetSize() { return dst_length; };
    size_t targetCount() { return dst_element_count; };
    size_t targetStartCount() { return dst_start_count; };

protected:
    const orc_byte* src_addr_cpu;    //< cpu address of source address
    const orc_byte* src_addr_gpu;    //< gpu address of source address
    size_t src_length;               //< lenght of source address

    orc_byte* dst_addr_gpu_top;      //< gpu address of target stream
    orc_byte* dst_addr_gpu_stream;   //< gpu address of target stream for this stream 

    size_t dst_length;               //< length of target stream
    size_t dst_element_count;        //< the count of elements for the target stream for this stream
    size_t dst_element_size;         //< the bytes of the element of thr target stream
    size_t dst_start_count;          //< the start element count from dst_addr_gpu_top
    size_t present_start_index;      //< start index of present stream
    size_t present_end_index;        //< end index of present stream


    const OrcStripeArguemnts* _stripe;      //< pointer for stripeArgs[stripe_id]
    const OrcColumnArguemnts* _column;      //< pointer for columnArgs[_colunm_id]
    int _colunm_id;                         //< id of the column for this stream, same as type id.
    ORCStreamKind _kind;                    //< Stream Kind


    //    CudaOrcRetStats retStats;
};

/** ---------------------------------------------------------------------------*
* @brief parameters to decode a stripe
* ---------------------------------------------------------------------------**/
class OrcStripeArguemnts {
public:
    OrcStripeArguemnts() : stride_addr_gpu(NULL), gmtoffset(0) {};
    ~OrcStripeArguemnts() {};

    CudaOrcError_t allocateAndCopyStripe(const orc_byte* src, size_t length);
    void releaseStripeBuffer();

    void SetCompressionKind(OrcCompressionKind kind) { compKind = kind; };
    void SetStripeInfo(OrcStripeInfo* info, int id) { stripe_info = info; stripe_id = id; };
    void SetDeviceArray(std::vector<ORCstream>*    _deviceArray) { deviceArray = _deviceArray; };

public:
    void DecodePresentStreams();
    void DecodeDataStreams();

    bool isFinished();    //< return true if the executions of all streams are completed.

protected:
    void DecodeCompressedStream(KernelParamBase *param, OrcStreamArguemnts *stream);


public:
    void setStreamRange(ORCStreamKind kind, int stream_id, int column_id, size_t offset, size_t length) {
        OrcStreamArguemnts& arg = getStreamArg(stream_id);
        arg.SetArgs(this, kind, &getColumnArg(column_id), column_id);
        arg.SetSource(getCpuAddress(offset), getGpuAddress(offset), length);
    };

    void setNumOfStream(int num) { streamArgs.resize(num); }
    void setNumOfColumn(int num) { columnArgs.resize(num); }

    OrcStreamArguemnts& getStreamArg(int stream_id) { return streamArgs[stream_id]; }
    OrcColumnArguemnts& getColumnArg(int column_id) { return columnArgs[column_id]; }

    const orc_byte* getCpuAddress(size_t offset) { return stride_addr_cpu + offset; };
    const orc_byte* getGpuAddress(size_t offset) { return stride_addr_gpu + offset; };

    int StripeId() const { return stripe_id; };

    void SetGMToffset(orc_uint32 gmtoff) { gmtoffset = gmtoff; };
    orc_uint32 GetGMToffset() { return gmtoffset; };

protected:
    int FindValidParant(int id);


protected:
    int stripe_id;                                      //< id of stripe
    const OrcStripeInfo* stripe_info;                   //< stripe info of the stripe_id
    const orc_byte* stride_addr_cpu;                    //< cpu accessible memory for the stride, usually mapped file memory
    const orc_byte* stride_addr_gpu;                    //< gpu accessible memory for the stride, usually cuda device memory
    size_t          stride_length;                      //< the effective byte length of the stride
    orc_uint32      gmtoffset;                          //< GMT offset[sec] for the writer's local timezone
    OrcCompressionKind      compKind;

    std::vector<OrcStreamArguemnts> streamArgs;         //< arguments for streams to be decoded
    std::vector<OrcColumnArguemnts> columnArgs;         //< arguments for culumns columnArgs.input_size == types.input_size

    std::vector<ORCstream>*         deviceArray;        //< pointer of device array

    OrcBufferInfoHolder holder;                         //< OrcBuffer holder for the stripe when ORC file is compressed
};


#endif // __ORC_STRIPE_HEADER__
