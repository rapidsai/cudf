/*
 *  Copyright (c) 2008-2024, NVIDIA CORPORATION
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

#pragma once

#include <cudf/utilities/error.hpp>

#include <thrust/host_vector.h>

#include <cstddef>
#include <limits>
#include <new>  // for bad_alloc

namespace cudf::detail {

/*! \p pinned_allocator is a CUDA-specific host memory allocator
 *  that employs \c cudaMallocHost for allocation.
 *
 * This implementation is ported from the experimental/pinned_allocator
 * that Thrust used to provide.
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <typename T>
class pinned_allocator;

/*! \p pinned_allocator is a CUDA-specific host memory allocator
 *  that employs \c cudaMallocHost for allocation.
 *
 * This implementation is ported from the experimental/pinned_allocator
 * that Thrust used to provide.
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <>
class pinned_allocator<void> {
 public:
  using value_type      = void;            ///< The type of the elements in the allocator
  using pointer         = void*;           ///< The type returned by address() / allocate()
  using const_pointer   = void const*;     ///< The type returned by address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  /**
   * @brief converts a `pinned_allocator<void>` to `pinned_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = pinned_allocator<U>;  ///< The rebound type
  };
};

/*! \p pinned_allocator is a CUDA-specific host memory allocator
 *  that employs \c cudaMallocHost for allocation.
 *
 * This implementation is ported from the experimental/pinned_allocator
 * that Thrust used to provide.
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <typename T>
class pinned_allocator {
 public:
  using value_type      = T;               ///< The type of the elements in the allocator
  using pointer         = T*;              ///< The type returned by address() / allocate()
  using const_pointer   = T const*;        ///< The type returned by address()
  using reference       = T&;              ///< The parameter type for address()
  using const_reference = T const&;        ///< The parameter type for address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  /**
   * @brief converts a `pinned_allocator<T>` to `pinned_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = pinned_allocator<U>;  ///< The rebound type
  };

  /**
   * @brief pinned_allocator's null constructor does nothing.
   */
  __host__ __device__ inline pinned_allocator() {}

  /**
   * @brief pinned_allocator's null destructor does nothing.
   */
  __host__ __device__ inline ~pinned_allocator() {}

  /**
   * @brief pinned_allocator's copy constructor does nothing.
   */
  __host__ __device__ inline pinned_allocator(pinned_allocator const&) {}

  /**
   * @brief  pinned_allocator's copy constructor does nothing.
   *
   *  This version of pinned_allocator's copy constructor
   *  is templated on the \c value_type of the pinned_allocator
   *  to copy from.  It is provided merely for convenience; it
   *  does nothing.
   */
  template <typename U>
  __host__ __device__ inline pinned_allocator(pinned_allocator<U> const&)
  {
  }

  /**
   * @brief This method returns the address of a \c reference of
   *  interest.
   *
   *  @param r The \c reference of interest.
   *  @return \c r's address.
   */
  __host__ __device__ inline pointer address(reference r) { return &r; }

  /**
   * @brief This method returns the address of a \c const_reference
   *  of interest.
   *
   *  @param r The \c const_reference of interest.
   *  @return \c r's address.
   */
  __host__ __device__ inline const_pointer address(const_reference r) { return &r; }

  /**
   * @brief This method allocates storage for objects in pinned host
   *  memory.
   *
   *  @param cnt The number of objects to allocate.
   *  @return a \c pointer to the newly allocated objects.
   *  @note The second parameter to this function is meant as a
   *        hint pointer to a nearby memory location, but is
   *        not used by this allocator.
   *  @note This method does not invoke \p value_type's constructor.
   *        It is the responsibility of the caller to initialize the
   *        objects at the returned \c pointer.
   */
  __host__ inline pointer allocate(size_type cnt, const_pointer /*hint*/ = 0)
  {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if

    pointer result(0);
    CUDF_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  /**
   * @brief This method deallocates pinned host memory previously allocated
   *  with this \c pinned_allocator.
   *
   *  @param p A \c pointer to the previously allocated memory.
   *  @note The second parameter is the number of objects previously allocated
   *        but is ignored by this allocator.
   *  @note This method does not invoke \p value_type's destructor.
   *        It is the responsibility of the caller to destroy
   *        the objects stored at \p p.
   */
  __host__ inline void deallocate(pointer p, size_type /*cnt*/)
  {
    auto dealloc_worked = cudaFreeHost(p);
    (void)dealloc_worked;
    assert(dealloc_worked == cudaSuccess);
  }

  /**
   * @brief This method returns the maximum size of the \c cnt parameter
   *  accepted by the \p allocate() method.
   *
   *  @return The maximum number of objects that may be allocated
   *          by a single call to \p allocate().
   */
  inline size_type max_size() const { return (std::numeric_limits<size_type>::max)() / sizeof(T); }

  /**
   * @brief This method tests this \p pinned_allocator for equality to
   *  another.
   *
   *  @param x The other \p pinned_allocator of interest.
   *  @return This method always returns \c true.
   */
  __host__ __device__ inline bool operator==(pinned_allocator const& x) const { return true; }

  /**
   * @brief This method tests this \p pinned_allocator for inequality
   *  to another.
   *
   *  @param x The other \p pinned_allocator of interest.
   *  @return This method always returns \c false.
   */
  __host__ __device__ inline bool operator!=(pinned_allocator const& x) const
  {
    return !operator==(x);
  }
};

/**
 * @brief A vector class with pinned host memory allocator
 */
template <typename T>
using pinned_host_vector = thrust::host_vector<T, pinned_allocator<T>>;

}  // namespace cudf::detail
