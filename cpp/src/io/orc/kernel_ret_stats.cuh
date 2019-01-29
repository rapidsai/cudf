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

#ifndef __ORC_KERNEL_RET_STATS_H__
#define __ORC_KERNEL_RET_STATS_H__

#include "kernel_private_common.cuh"


#define ORC_PERF_RETS(expr) {expr; }


#ifdef CHECK_ORC_PERF_STATS
#define ORC_PERF_STATS(expr) {expr; }
#else
#define ORC_PERF_STATS(expr) NOOP
#endif

//! This class is used for debugging/investigation/performance tuning
struct CudaOrcKernelRetStats {
public:
    __device__ CudaOrcKernelRetStats(CudaOrcKernelRetStatsValue* value) : ret_val(value){
        SetReturnCode(0);
    }
    __device__ CudaOrcKernelRetStats() {
    }
    __device__ ~CudaOrcKernelRetStats() {};

    __device__ void SetReturnCode(int ret) { ORC_PERF_RETS(val.ret = ret; ) };
    __device__ void SetDecodeCount(orc_uint32 count) { ORC_PERF_RETS( val.decoded_count = count; ) };
    __device__ void SetReadCount(int count) { ORC_PERF_RETS(val.reader_count = count; ) };

    __device__ void SetNullCount(int count) { ORC_PERF_STATS(val.null_count = count; ) };

    __device__ void Init(int _max_mode, int _max_codepath) { 
        ORC_PERF_STATS(max_mode = _max_mode);
        ORC_PERF_STATS(max_codepath = _max_codepath);
        ORC_PERF_STATS(for (int i = 0; i < max_mode; i++) val.mode_count[i] = 0; );
        ORC_PERF_STATS(for (int i = 0; i < max_codepath; i++) val.code_path_count[i] = 0; );
    };

    // -- these functions are used for Stats of Kernel execution.
    __device__ void IncrementModeCount(int mode) {
        ORC_PERF_STATS(val.mode_count[mode] ++; );
    }
    __device__ void IncrementCodePathCount(int codePathID) {
        ORC_PERF_STATS(val.code_path_count[codePathID] ++;);
    }
    // --

    // only this fuction call writes the values into managed memory
    __device__ void Output() {
        if (ret_val && threadIdx.x == 0)
        {
#define GDF_ORC_WRITE(x)  ret_val->x = val.x;
            GDF_ORC_WRITE(ret);
            GDF_ORC_WRITE(decoded_count);
            GDF_ORC_WRITE(reader_count);

#ifdef CHECK_ORC_PERF_STATS
            GDF_ORC_WRITE(null_count);
#undef GDF_ORC_WRITE
#define GDF_ORC_WRITE(x, count)     for(int i=0; i<count; i++){ret_val->x[i] = val.x[i];}
            GDF_ORC_WRITE(mode_count, 4);
            GDF_ORC_WRITE(code_path_count, 20);
#undef GDF_ORC_WRITE
#endif
        }
    }

protected:
    CudaOrcKernelRetStatsValue val;
    CudaOrcKernelRetStatsValue* ret_val;
    int max_mode;
    int max_codepath;

};


#endif // __ORC_KERNEL_RET_STATS_H__
