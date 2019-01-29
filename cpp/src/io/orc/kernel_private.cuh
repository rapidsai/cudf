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

// file: kernel_private.cuh
// commmon private header for .cu files

#ifndef __ORC_KERNEL_PRIVATE_H__
#define __ORC_KERNEL_PRIVATE_H__

#include <assert.h>
#include <algorithm>


#if defined(_DEBUG) || 1
#define ORC_DEBUG_KERNEL_CALL_CHECK()  CudaFuncCall(cudaGetLastError()); CudaFuncCall(cudaDeviceSynchronize());
#else
#define ORC_DEBUG_KERNEL_CALL_CHECK()  ((void)0);
#endif

#include "kernel_private_common.cuh"

#include "kernel_util.cuh"
#include "kernel_ctc.cuh"
#include "kernel_ret_stats.cuh"
#include "kernel_converter.cuh"

#include "kernel_reader.cuh"
#include "kernel_reader_bitmap.cuh"
#include "kernel_reader_present.cuh"

#include "kernel_writer.cuh"
#include "kernel_writer_depends.cuh"
#include "kernel_writer_bitmap.cuh"
#include "kernel_writer_bitmap_depends.cuh"

#include "kernel_decode_common.cuh"


#endif // __ORC_KERNEL_PRIVATE_H__
