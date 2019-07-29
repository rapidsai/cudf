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

namespace custr
{
    // convert string with numerical characters to number
    __device__ inline int stoi( const char* str, unsigned int bytes );
    __device__ inline long stol( const char* str, unsigned int bytes );
    __device__ inline unsigned long stoul( const char* str, unsigned int bytes );
    __device__ inline float stof( const char* str, unsigned int bytes );
    __device__ inline double stod( const char* str, unsigned int bytes );
    __device__ inline unsigned int hash( const char* str, unsigned int bytes );

    //
    __device__ inline int compare(const char* src, unsigned int sbytes, const char* tgt, unsigned int tbytes );
    __device__ inline int find( const char* src, unsigned int sbytes, const char* tgt, unsigned int tbytes );
    __device__ inline int rfind( const char* src, unsigned int sbytes, const char* tgt, unsigned int tbytes );
    //
    __device__ inline void copy( char* dst, unsigned int bytes, const char* src );

}

#include "custring.inl"