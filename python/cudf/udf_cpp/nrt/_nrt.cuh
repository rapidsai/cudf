#ifndef _NRT_H
#define _NRT_H


#include <cuda/atomic>

typedef __device__ void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
typedef __device__ void (*NRT_dealloc_func)(void* ptr, void* dealloc_info);


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
// TODO: add the rest of the declarations
extern "C" __device__ void NRT_Free(void* ptr);

extern "C" __device__ void NRT_decref(void* ptr);

void setGlobalMemSys(NRT_MemSys* allocated_memsys);

#endif
