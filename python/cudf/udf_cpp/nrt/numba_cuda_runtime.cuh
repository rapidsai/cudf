/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#ifndef _NRT_H
#define _NRT_H

#include <cuda/atomic>

typedef __device__ void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef __device__ void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);

typedef struct MemInfo NRT_MemInfo;

extern "C" {
struct MemInfo {
  cuda::atomic<size_t, cuda::thread_scope_device> refct;
  NRT_dtor_function dtor;
  void* dtor_info;
  void* data;
  size_t size;
};
}

// Globally needed variables
struct NRT_MemSys {
  struct {
    bool enabled;
    cuda::atomic<size_t, cuda::thread_scope_device> alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> free;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_free;
  } stats;
};

/* The Memory System object */
__device__ NRT_MemSys* TheMSys;

extern "C" __device__ void* NRT_Allocate(size_t size, NRT_MemSys* TheMSys)
{
  void* ptr = NULL;
  ptr       = malloc(size);
  if (TheMSys->stats.enabled) { TheMSys->stats.alloc++; }
  return ptr;
}

extern "C" __device__ void NRT_MemInfo_init(NRT_MemInfo* mi,
                                            void* data,
                                            size_t size,
                                            NRT_dtor_function dtor,
                                            void* dtor_info,
                                            NRT_MemSys* TheMSys)
{
  mi->refct     = 1; /* starts with 1 refct */
  mi->dtor      = dtor;
  mi->dtor_info = dtor_info;
  mi->data      = data;
  mi->size      = size;
  if (TheMSys->stats.enabled) { TheMSys->stats.mi_alloc++; }
}

__device__ NRT_MemInfo* NRT_MemInfo_new(
  void* data, size_t size, NRT_dtor_function dtor, void* dtor_info, NRT_MemSys* TheMSys)
{
  NRT_MemInfo* mi = (NRT_MemInfo*)NRT_Allocate(sizeof(NRT_MemInfo), TheMSys);
  if (mi != NULL) { NRT_MemInfo_init(mi, data, size, dtor, dtor_info, TheMSys); }
  return mi;
}

extern "C" __device__ void NRT_Free(void* ptr, NRT_MemSys* TheMSys)
{
  free(ptr);
  if (TheMSys->stats.enabled) { TheMSys->stats.free++; }
}

extern "C" __device__ void NRT_dealloc(NRT_MemInfo* mi, NRT_MemSys* TheMSys)
{
  NRT_Free(mi, TheMSys);
}

extern "C" __device__ void NRT_MemInfo_destroy(NRT_MemInfo* mi, NRT_MemSys* TheMSys)
{
  NRT_dealloc(mi, TheMSys);
  if (TheMSys->stats.enabled) { TheMSys->stats.mi_free++; }
}
extern "C" __device__ void NRT_MemInfo_call_dtor(NRT_MemInfo* mi, NRT_MemSys* TheMSys)
{
  if (mi->dtor) /* We have a destructor */
    mi->dtor(mi->data, mi->size, NULL);
  /* Clear and release MemInfo */
  NRT_MemInfo_destroy(mi, TheMSys);
}

/*
  c++ version of the NRT_decref function that usually is added to
  the final kernel link in PTX form by numba. This version may be
  used by c++ APIs that accept ownership of live objects and must
  manage them going forward.
*/
extern "C" __device__ void NRT_internal_decref(NRT_MemInfo* mi, NRT_MemSys* TheMSys)
{
  mi->refct--;
  if (mi->refct == 0) { NRT_MemInfo_call_dtor(mi, TheMSys); }
}

#endif
