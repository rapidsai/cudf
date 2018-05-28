// Copyright 2017 NVIDIA Corporation.  All rights reserved.
#ifndef MANAGED_CUH
#define MANAGED_CUH

#include <new>

struct managed {
    static void *operator new(size_t n) {
        void* ptr = 0;
        cudaError_t result = cudaMallocManaged( &ptr, n );
        if( cudaSuccess != result || 0 == ptr ) throw std::bad_alloc();
        return ptr;
    }

    static void operator delete(void *ptr) noexcept {
        cudaFree(ptr);
    }
};

#endif //MANAGED_CUH
