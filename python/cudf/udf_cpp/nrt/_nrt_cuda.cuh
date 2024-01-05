/*
* SPDX-FileCopyrightText: Copyright (c) <2023> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
Copyright (c) 2012, Anaconda, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _NRT_CUDA_H
#define _NRT_CUDA_H

#include <cuda/atomic>
#include "_nrt.cuh"

typedef struct MemInfo NRT_MemInfo;

/* The Memory System object */
__device__ NRT_MemSys* TheMSys;

void setGlobalMemSys(NRT_MemSys* allocated_memsys) {
  TheMSys = allocated_memsys;
}

extern "C" __device__ void* NRT_Allocate(size_t size)
{
  void* ptr = NULL;
  ptr       = malloc(size);
  if (TheMSys->stats.enabled) { TheMSys->stats.alloc++; }
  return ptr;
}

extern "C" __device__ void NRT_MemInfo_init(
  NRT_MemInfo* mi, void* data, size_t size, NRT_dtor_function dtor, void* dtor_info)
{
  mi->refct     = 1; /* starts with 1 refct */
  mi->dtor      = dtor;
  mi->dtor_info = dtor_info;
  mi->data      = data;
  mi->size      = size;
  if (TheMSys->stats.enabled) { TheMSys->stats.mi_alloc++; }
}

__device__ NRT_MemInfo* NRT_MemInfo_new(void* data,
                                        size_t size,
                                        NRT_dtor_function dtor,
                                        void* dtor_info)
{
  NRT_MemInfo* mi = (NRT_MemInfo*)NRT_Allocate(sizeof(NRT_MemInfo));
  if (mi != NULL) { NRT_MemInfo_init(mi, data, size, dtor, dtor_info); }
  return mi;
}

extern "C" __device__ void NRT_Free(void* ptr)
{
  free(ptr);
  if (TheMSys->stats.enabled) { TheMSys->stats.free++; }
}

extern "C" __device__ void NRT_dealloc(NRT_MemInfo* mi) { NRT_Free(mi); }

extern "C" __device__ void NRT_MemInfo_destroy(NRT_MemInfo* mi)
{
  NRT_dealloc(mi);
  if (TheMSys->stats.enabled) { TheMSys->stats.mi_free++; }
}
extern "C" __device__ void NRT_MemInfo_call_dtor(NRT_MemInfo* mi)
{
  if (mi->dtor) /* We have a destructor */
    mi->dtor(mi->data, mi->size, NULL);
  /* Clear and release MemInfo */
  NRT_MemInfo_destroy(mi);
}
#endif
