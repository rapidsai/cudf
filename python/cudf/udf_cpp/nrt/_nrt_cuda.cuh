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
 
 typedef __device__ void* (*NRT_external_malloc_func)(size_t size, void* opaque_data);
 typedef __device__ void* (*NRT_external_realloc_func)(void* ptr, size_t new_size, void* opaque_data);
 typedef __device__ void (*NRT_external_free_func)(void* ptr, void* opaque_data);
 
 
 struct ExternalMemAllocator {
   NRT_external_malloc_func malloc;
   NRT_external_realloc_func realloc;
   NRT_external_free_func free;
   void* opaque_data;
 };
 
 typedef struct ExternalMemAllocator NRT_ExternalAllocator;
 
 extern "C" {
 struct MemInfo {
   cuda::atomic<size_t, cuda::thread_scope_device> refct;
   NRT_dtor_function dtor;
   void* dtor_info;
   void* data;
   size_t size;
   NRT_ExternalAllocator* external_allocator;
 };
 }
 
 typedef struct MemInfo NRT_MemInfo;
 
 // Globally needed variables
 struct NRT_MemSys {
   /* Shutdown flag */
   int shutting;
   struct {
     bool enabled;
     std::atomic_size_t alloc;
     std::atomic_size_t free;
     std::atomic_size_t mi_alloc;
     std::atomic_size_t mi_free;
   } stats;
   /* System allocation functions */
   struct {
     NRT_malloc_func malloc;
     NRT_realloc_func realloc;
     NRT_free_func free;
   } allocator;
 };
 
 /* The Memory System object */
 __device__ NRT_MemSys TheMSys;
 
 // for creating a new MemInfo object from an existing pointer
 // create a new MemInfo structure
 
 // allocate from the heap using whatever the global allocator is set to
 
 extern "C" __device__ void* NRT_Allocate_External(size_t size, NRT_ExternalAllocator* allocator)
 {
   
   void* ptr = NULL;
   if (allocator) {
     ptr = allocator->malloc(size, allocator->opaque_data);
   } else {
     ptr = malloc(size);
   }
   return ptr;
 }
 
 extern "C" __device__ void* NRT_Allocate(size_t size) { return NRT_Allocate_External(size, NULL); }
 
 extern "C" __device__ void NRT_MemInfo_init(NRT_MemInfo* mi,
                                             void* data,
                                             size_t size,
                                             NRT_dtor_function dtor,
                                             void* dtor_info,
                                             NRT_ExternalAllocator* external_allocator)
 {
   mi->refct              = 1; /* starts with 1 refct */
   mi->dtor               = dtor;
   mi->dtor_info          = dtor_info;
   mi->data               = data;
   mi->size               = size;
   mi->external_allocator = external_allocator;
 }
 
 __device__ NRT_MemInfo* NRT_MemInfo_new(void* data,
                                         size_t size,
                                         NRT_dtor_function dtor,
                                         void* dtor_info)
 {
   NRT_MemInfo* mi = (NRT_MemInfo*)NRT_Allocate(sizeof(NRT_MemInfo));
   if (mi != NULL) {
     NRT_MemInfo_init(mi, data, size, dtor, dtor_info, NULL);
   }
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
 
 extern "C" __device__ void NRT_MemInfo_acquire(NRT_MemInfo* mi)
 {
   assert(mi->refct > 0 && "RefCt cannot be zero");
   mi->refct++;
 }
 
 extern "C" __device__ void NRT_MemInfo_release(NRT_MemInfo* mi)
 {
   assert(mi->refct > 0 && "RefCt cannot be 0");
   /* RefCt drop to zero */
   if ((--(mi->refct)) == 0) { NRT_MemInfo_call_dtor(mi); }
 }
 
 __device__ void* nrt_allocate_meminfo_and_data(size_t size,
                                                NRT_MemInfo** mi_out,
                                                NRT_ExternalAllocator* allocator)
 {
   NRT_MemInfo* mi = NULL;
   char* base = (char*)NRT_Allocate_External(sizeof(NRT_MemInfo) + size, allocator);
   if (base == NULL) {
     *mi_out = NULL; /* set meminfo to NULL as allocation failed */
     return NULL;    /* return early as allocation failed */
   }
   mi      = (NRT_MemInfo*)base;
   *mi_out = mi;
   return (void*)((char*)base + sizeof(NRT_MemInfo));
 }
 
 __device__ void* nrt_allocate_meminfo_and_data_align(size_t size,
                                                      unsigned align,
                                                      NRT_MemInfo** mi,
                                                      NRT_ExternalAllocator* allocator)
 {
   size_t offset = 0, intptr = 0, remainder = 0;
   char* base = (char*)nrt_allocate_meminfo_and_data(size + 2 * align, mi, allocator);
   if (base == NULL) { return NULL; /* return early as allocation failed */ }
   intptr = (size_t)base;
   /*
    * See if the allocation is aligned already...
    * Check if align is a power of 2, if so the modulo can be avoided.
    */
   if ((align & (align - 1)) == 0) {
     remainder = intptr & (align - 1);
   } else {
     remainder = intptr % align;
   }
   if (remainder == 0) { /* Yes */
     offset = 0;
   } else { /* No, move forward `offset` bytes */
     offset = align - remainder;
   }
   return (void*)((char*)base + offset);
 }
 
 extern "C" __device__ NRT_MemInfo* NRT_MemInfo_alloc_aligned(size_t size, unsigned align)
 {
   NRT_MemInfo* mi = NULL;
   void* data      = nrt_allocate_meminfo_and_data_align(size, align, &mi, NULL);
   if (data == NULL) { return NULL; /* return early as allocation failed */ }
   NRT_MemInfo_init(mi, data, size, NULL, NULL, NULL);
   return mi;
 }
