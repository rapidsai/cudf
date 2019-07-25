/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <cuda_runtime.h>
#include <string>

class NVStrings;

// csv parser flags
#define CSV_SORT_LENGTH    1
#define CSV_SORT_NAME      2
#define CSV_NULL_IS_EMPTY  8
// this has become a thing
NVStrings* createFromCSV(std::string csvfile, unsigned int column, unsigned int lines=0, unsigned int flags=0);

__host__ __device__ inline unsigned int u2u8( unsigned int unchr );
__host__ __device__ inline unsigned int u82u( unsigned int utf8 );

// copies and moves dest pointer
__device__ inline char* copy_and_incr( char*& dest, char* src, unsigned int bytes );
__device__ inline char* copy_and_incr_both( char*& dest, char*& src, unsigned int bytes );

template<typename T>
T* device_alloc(size_t count, cudaStream_t sid);

// adapted from cudf/cpp/src/utilities/error_utils.hpp
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
        std::ostringstream message;                               \
        message << "error " << status << " from cuda call";       \
        throw std::runtime_error(message.str());                  \
    }                                                             \
  } while (0);

//
#include "util.inl"
