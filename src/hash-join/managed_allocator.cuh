// Copyright 2017 NVIDIA Corporation.  All rights reserved.

#ifndef MANAGED_ALLOCATOR_CUH
#define MANAGED_ALLOCATOR_CUH

#include <new>

template <class T>
struct managed_allocator {
      typedef T value_type;
      
      managed_allocator() = default;
      
      template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
      
      T* allocate(std::size_t n) const {
          T* ptr = 0;
          cudaError_t result = cudaMallocManaged( &ptr, n*sizeof(T) );
          if( cudaSuccess != result || 0 == ptr ) throw std::bad_alloc();
          return ptr;
      }
      void deallocate(T* p, std::size_t) const noexcept {
        cudaFree(p);
      }
};

template <class T, class U>
bool operator==(const managed_allocator<T>&, const managed_allocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const managed_allocator<T>&, const managed_allocator<U>&) { return false; }

template <class T>
struct legacy_allocator {
      typedef T value_type;

      legacy_allocator() = default;

      template <class U> constexpr legacy_allocator(const legacy_allocator<U>&) noexcept {}

      T* allocate(std::size_t n) const {
          T* ptr = 0;
          cudaError_t result = cudaMalloc( &ptr, n*sizeof(T) );
          if( cudaSuccess != result || 0 == ptr ) throw std::bad_alloc();
          return ptr;
      }
      void deallocate(T* p, std::size_t) const noexcept {
        cudaFree(p);
      }
};

template <class T, class U>
bool operator==(const legacy_allocator<T>&, const legacy_allocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const legacy_allocator<T>&, const legacy_allocator<U>&) { return false; }

#endif
