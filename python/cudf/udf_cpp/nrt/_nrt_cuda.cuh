/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuda/atomic>

typedef __device__ void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef __device__ void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);

typedef __device__ void* (*NRT_malloc_func)(size_t size);
typedef __device__ void* (*NRT_realloc_func)(void* ptr, size_t new_size);
typedef __device__ void (*NRT_free_func)(void* ptr);

extern "C" {
struct MemInfo {
  cuda::atomic<size_t, cuda::thread_scope_device> refct;
  NRT_dtor_function dtor;
  void* dtor_info;
  void* data;
  size_t size;
};
}

typedef struct MemInfo NRT_MemInfo;

// Globally needed variables
struct NRT_MemSys {
  /* Shutdown flag */
  int shutting;
  /* System allocation functions */
  struct {
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
  } allocator;
};

/* The Memory System object */
__device__ NRT_MemSys TheMSys;

extern "C" __device__ void* NRT_Allocate(size_t size) { return malloc(size); }

extern "C" __device__ void NRT_MemInfo_init(
  NRT_MemInfo* mi, void* data, size_t size, NRT_dtor_function dtor, void* dtor_info)
{
  mi->refct     = 1; /* starts with 1 refct */
  mi->dtor      = dtor;
  mi->dtor_info = dtor_info;
  mi->data      = data;
  mi->size      = size;
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

extern "C" __device__ void NRT_Free(void* ptr) { free(ptr); }

extern "C" __device__ void NRT_dealloc(NRT_MemInfo* mi) { NRT_Free(mi); }

extern "C" __device__ void NRT_MemInfo_destroy(NRT_MemInfo* mi) { NRT_dealloc(mi); }

extern "C" __device__ void NRT_MemInfo_call_dtor(NRT_MemInfo* mi)
{
  if (mi->dtor && !TheMSys.shutting) /* We have a destructor and the system is not shutting down */
    mi->dtor(mi->data, mi->size, NULL);
  /* Clear and release MemInfo */
  NRT_MemInfo_destroy(mi);
}
