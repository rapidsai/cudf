/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#ifndef __ERROR_CUH
#define __ERROR_CUH

#include <cstdio>
#include <cstdlib>
#include <rmm/rmm.h>
#include <cudf/cudf.h>


#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                         \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if (cudaSuccess != cudaStatus)                                                                 \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}
#endif


#ifndef RMM_CALL
#define RMM_CALL(call)                                                                             \
{                                                                                                  \
    rmmError_t rmmStatus = call;                                                                   \
    if (RMM_SUCCESS != rmmStatus)                                                                  \
        fprintf(stderr, "\"%s\" in line %d of file %s failed with %s (%d).\n",                     \
                        #call, __LINE__, __FILE__, rmmGetErrorString(rmmStatus), rmmStatus);       \
}
#endif


#ifndef GDF_CALL
#define GDF_CALL(call)                                                                             \
{                                                                                                  \
    gdf_error status = call;                                                                       \
    if (GDF_SUCCESS != status)                                                                     \
        fprintf(stderr, "\"%s\" in line %d of file %s failed with %s (%d).\n",                     \
                        #call, __LINE__, __FILE__, gdf_error_get_name(status), status);            \
}
#endif


#endif // __ERROR_CUH
