/*
 *  Copyright 2008-2022 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/cuda/experimental/pinned_allocator.h
 *  \brief An allocator which creates new elements in "pinned" memory with \p cudaMallocHost
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <stdexcept>
#include <limits>
#include <string>
#include <thrust/system/system_error.h>
#include <thrust/system/cuda/error.h>


namespace cudf
{

/*! \p pinned_allocator is a CUDA-specific host memory allocator
 *  that employs \c cudaMallocHost for allocation.
 *
 * This implementation is ported from the experimental/pinned_allocator
 * that Thrust used to provide.
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template<typename T> class pinned_allocator;

template<>
  class pinned_allocator<void>
{
  public:
    typedef void           value_type;
    typedef void       *   pointer;
    typedef const void *   const_pointer;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // convert a pinned_allocator<void> to pinned_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef pinned_allocator<U> other;
    }; // end rebind
}; // end pinned_allocator


template<typename T>
  class pinned_allocator
{
  public:
    //! \{
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;
    //! \}

    // convert a pinned_allocator<T> to pinned_allocator<U>
    template<typename U>
      struct rebind
    {
      typedef pinned_allocator<U> other;
    }; // end rebind

    /*! \p pinned_allocator's null constructor does nothing.
     */
    __host__ __device__
    inline pinned_allocator() {}

    /*! \p pinned_allocator's null destructor does nothing.
     */
    __host__ __device__
    inline ~pinned_allocator() {}

    /*! \p pinned_allocator's copy constructor does nothing.
     */
    __host__ __device__
    inline pinned_allocator(pinned_allocator const &) {}

    /*! This version of \p pinned_allocator's copy constructor
     *  is templated on the \c value_type of the \p pinned_allocator
     *  to copy from.  It is provided merely for convenience; it
     *  does nothing.
     */
    template<typename U>
    __host__ __device__
    inline pinned_allocator(pinned_allocator<U> const &) {}

    /*! This method returns the address of a \c reference of
     *  interest.
     *
     *  \p r The \c reference of interest.
     *  \return \c r's address.
     */
    __host__ __device__
    inline pointer address(reference r) { return &r; }

    /*! This method returns the address of a \c const_reference
     *  of interest.
     *
     *  \p r The \c const_reference of interest.
     *  \return \c r's address.
     */
    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    /*! This method allocates storage for objects in pinned host
     *  memory.
     *
     *  \p cnt The number of objects to allocate.
     *  \return a \c pointer to the newly allocated objects.
     *  \note This method does not invoke \p value_type's constructor.
     *        It is the responsibility of the caller to initialize the
     *        objects at the returned \c pointer.
     */
    __host__
    inline pointer allocate(size_type cnt,
                            const_pointer = 0)
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      pointer result(0);
      cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type));

      if(error)
      {
        cudaGetLastError(); // Clear global CUDA error state.
        throw std::bad_alloc();
      } // end if

      return result;
    } // end allocate()

    /*! This method deallocates pinned host memory previously allocated
     *  with this \c pinned_allocator.
     *
     *  \p p A \c pointer to the previously allocated memory.
     *  \p cnt The number of objects previously allocated at
     *         \p p.
     *  \note This method does not invoke \p value_type's destructor.
     *        It is the responsibility of the caller to destroy
     *        the objects stored at \p p.
     */
    __host__
    inline void deallocate(pointer p, size_type /*cnt*/)
    {
      cudaError_t error = cudaFreeHost(p);

      cudaGetLastError(); // Clear global CUDA error state.

      if(error)
      {
        cudaGetLastError(); // Clear global CUDA error state.
        throw thrust::system_error(error, thrust::cuda_category());
      } // end if
    } // end deallocate()

    /*! This method returns the maximum size of the \c cnt parameter
     *  accepted by the \p allocate() method.
     *
     *  \return The maximum number of objects that may be allocated
     *          by a single call to \p allocate().
     */
    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    } // end max_size()

    /*! This method tests this \p pinned_allocator for equality to
     *  another.
     *
     *  \param x The other \p pinned_allocator of interest.
     *  \return This method always returns \c true.
     */
    __host__ __device__
    inline bool operator==(pinned_allocator const& x) const { return true; }

    /*! This method tests this \p pinned_allocator for inequality
     *  to another.
     *
     *  \param x The other \p pinned_allocator of interest.
     *  \return This method always returns \c false.
     */
    __host__ __device__
    inline bool operator!=(pinned_allocator const &x) const { return !operator==(x); }
}; // end pinned_allocator

/*! \}
 */

} // end cudf
