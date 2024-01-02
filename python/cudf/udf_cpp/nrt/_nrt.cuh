#ifndef _NRT_H
#define _NRT_H


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

// Globally needed variables
struct NRT_MemSys {
  /* System allocation functions */
  struct {
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
  } allocator;
  struct {
    bool enabled;
    cuda::atomic<size_t, cuda::thread_scope_device> alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> free;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_free;
  } stats;
};

#endif
