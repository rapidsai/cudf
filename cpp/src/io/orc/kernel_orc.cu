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

#include "device_launch_parameters.h"

#include "kernel_orc.cuh"
#include <vector>

namespace cudf {
namespace orc {

CudaOrcRetStats::CudaOrcRetStats()
{ 
    CudaFuncCall(cudaMallocManaged(&gpu_addr, sizeof(CudaOrcKernelRetStatsValue))); 
};

CudaOrcRetStats::~CudaOrcRetStats() 
{
    CudaFuncCall(cudaFree(gpu_addr)) 
};

void CudaOrcRetStats::copyToHost()
{
//    CudaFuncCall(cudaMemcpy(&value, gpu_addr, sizeof(CudaOrcKernelRetStatsValue), cudaMemcpyDeviceToHost));
}

void CudaOrcRetStats::dump(int mode, int codepath)
{
    PRINTF("Decode count: %d", value.decoded_count);

#ifdef  CHECK_ORC_PERF_STATS
    PRINTF("Null count: %d", value.null_count);
    PRINTF("Mode: %d", mode);
    for (int i = 0; i < mode; i++) {
        PRINTF(" [%d]: %d", i, value.mode_count[i]);
    }
    
    for (int i = 0; i < codepath; i++) {
        PRINTF(" [%d]: %d", i, value.code_path_count[i]);
    }
#endif

    PRINTF(" ");
}

void CudaOrcRetStats::lazy_dump(int mode, int codepath)
{
//    CudaFuncCall(cudaGetLastError());
//    CudaFuncCall(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    copyToHost();

    dump(mode, codepath);
}

}   // namespace orc
}   // namespace cudf
