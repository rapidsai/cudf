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

#ifndef __ORC_CUDA_CUH__
#define __ORC_CUDA_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>

#include "orc_types.h"
#include "orc_debug.h"


#ifdef _DEBUG
#define CHECK_ORC_PERF_STATS    
#endif

#ifdef WIN32
#define DEBUG_BREAK()       __debugbreak()
#else
#define DEBUG_BREAK()
#endif

#define CudaFuncCall(call)                            \
{                                                     \
    const cudaError_t error = (call);                 \
    if (error != cudaSuccess)                         \
    {                                                 \
       PRINTF("Error: %s:%d,  ", __FILE__, __LINE__); \
       PRINTF("code:%d, reason: %s\n", error,         \
            cudaGetErrorString(error));               \
       DEBUG_BREAK(); \
    }                                                 \
}

namespace cudf {
namespace orc {

struct CudaOrcKernelRetStatsValue {
    // these parametes are required value to be returned from cuda kernel
    int ret;                     //< return code of the kernel 
    orc_uint32 decoded_count;    //< decoded count of the executed kernel
    orc_uint32 reader_count;     //< the count that the reader load.

    // this is used for debug and perf stats usage
    // these parameters are optional. available if CHECK_ORC_PERF_STATS is defined.
#ifdef CHECK_ORC_PERF_STATS
    //! the range of writer class wrote. same as decoded_count at data_writer, but different at data_writer_depends
    orc_uint32 writen_range;

    union {
        struct {
            orc_uint32 mode_count[4];
            orc_uint32 code_path_count[20];
        };
    };

    union {
        orc_uint32 null_count;
    };
#endif
};

enum OrcKernelConvertionType {
    GdfConvertNone = 0,
    GdfDate64,
    GdfTimestampUnit_s,
    GdfTimestampUnit_ms,
    GdfTimestampUnit_us,
    GdfTimestampUnit_ns,
    GdfString_direct,
    GdfString_dictionary,
};

struct OrcBuffer {
    size_t bufferSize;   //< byte size of the buffer
    orc_byte* buffer;    //< pointer to the start address of the buffer
};

struct OrcBufferArray {
    int numBuffers;      //< the count of the buffers. 0 if there is no decompressed buffers.
    OrcBuffer* buffers;  //< pointer to OrcBuffer

    OrcBufferArray() : numBuffers(0) {};
};


//! cuda kernel's parameter only for decoding present streams.
struct KernelParamBase {
    orc_byte* output;                   //< the start address of output buffer
    const orc_byte* input;              //< the start address of input buffer
    size_t input_size;                  //< input_size of input buffer
    size_t start_id;                    //< the start offset bits of output buffer [0-7]
    size_t output_count;                //< the element count of output buffer (for present stream, byte length)
    OrcBufferArray bufferArray;         //< buffer array for cpu decompressed buffers
    cudaStream_t stream;                //< cuda stream
    CudaOrcKernelRetStatsValue *stat;   //< return value and kernel statistics (stats are debug only) 

    KernelParamBase()
        : output(NULL)
        , output_count(0)
        , input(NULL)
        , input_size(0)
        , start_id(0)
        , stream(NULL)
    {
        bufferArray.numBuffers = 0;
        bufferArray.buffers = NULL;
    };
};

//! cuda kernel's parameter only for decoding present streams.
struct KernelParamBitmap : KernelParamBase {
    const orc_bitmap* parent;               //< the start address of parent stream, NULL if no parent stream.
    size_t end_id;                          //< the end offset bits of output buffer [0-7]
};

//! cuda kernel's parameter for decoding data streams.
struct KernelParamCommon : KernelParamBase {
    const orc_bitmap* present;              //< the start address of present stream, NULL if no present stream.
    OrcElementType    elementType;          //< the element type of output
    int    elementSize;                     //< the size of the element
    OrcKernelConvertionType convertType;    //< the type of date conversion

    KernelParamCommon()
        : KernelParamBase()
        , present(NULL)
        , elementType(OrcElementType::None)
        , elementSize(1)
        , convertType(OrcKernelConvertionType::GdfConvertNone)
    {};
};

// this parameter is used for data stream conversion between ORC and GDF
struct KernelParamCoversion {
    OrcKernelConvertionType convertType;    //< the type of data conversion, it also defines element types of output/input/secondary streams

    void* output;                           //< the start address of output stream
    const void* input;                      //< the start address of data stream
    const void* secondary;                  //< the start address of secondary stream

    const orc_bitmap* null_bitmap;          //< the start address of null bitmap stream
    int start_id;                           //< the start offset of null bitmap stream

    orc_sint64 adjustClock;                 //< adjustment clock value from ORC epock (local time) into unix epock time (GMT).

    size_t data_count;                      //< the element count of output buffer
    size_t dict_count;                      //< the dictionary count for dictionary data conversion.

    cudaStream_t stream;                    //< cuda stream

    KernelParamCoversion()
        : stream(NULL)
    {};
};

// this parameter is designed to get the status of kernel execution to debug, perf tuning in future
class CudaOrcRetStats {
public:
    CudaOrcRetStats();
    ~CudaOrcRetStats();

    const CudaOrcKernelRetStatsValue& getRetStats() { return value; };
    const orc_byte* getGpuAddr() { return gpu_addr; };

    void copyToHost();

    void dump(int mode, int codepath);

    void lazy_dump(int mode, int codepath);


protected:
    CudaOrcKernelRetStatsValue value;
    orc_byte* gpu_addr;
};

// -----------------------------------------------------------------------------------------
// declaration of entry functions invoking cuda kernel
// -----------------------------------------------------------------------------------------
void cudaDecodePresent(KernelParamBitmap* param);
void cudaDecodeIntRLEv2(KernelParamCommon* param);
void cudaDecodeIntRLEv1(KernelParamCommon* param);
void cudaDecodeVarint(KernelParamCommon* param);
void cudaDecodeBooleanRLE(KernelParamCommon* param);
void cudaDecodeByteRLE(KernelParamCommon* param);
void cudaDecodeRawData(KernelParamCommon* param);

void cudaConvertData(KernelParamCoversion* param);
void cudaClearPresent(KernelParamBitmap* param);

// -----------------------------------------------------------------------------------------
}   // namespace orc
}   // namespace cudf

#endif //  __ORC_CUDA_CUH__
