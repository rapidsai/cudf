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

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

namespace cudf::detail {

/// Sentinel value for `cudf::size_type`
static cudf::size_type constexpr CUDF_SIZE_TYPE_SENTINEL = -1;

/// Default load factor for cuco data structures
static double constexpr CUCO_DESIRED_LOAD_FACTOR = 0.5;

/**
 * @brief Stream-ordered allocator adaptor used for cuco data structures
 *
 * The stream-ordered `rmm::mr::polymorphic_allocator` cannot be used in `cuco` directly since the
 * later expects a standard C++ `Allocator` interface. This allocator helper provides a simple way
 * to handle cuco memory allocation/deallocation with the given `stream` and the rmm default memory
 * resource.
 */
template <typename T>
class cuco_allocator_base
  : public rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<T>> {
  /// Default stream-ordered allocator type
  using default_allocator = rmm::mr::polymorphic_allocator<T>;
  /// The base allocator adaptor type
  using base_type = rmm::mr::stream_allocator_adaptor<default_allocator>;

 public:
  template <typename U>
  cuco_allocator_base(cuco_allocator_base<U> const& other) : cuco_allocator_base{other.stream()}
  {
  }

  /**
   * @brief Constructs the allocator adaptor with the given `stream`
   */
  cuco_allocator_base(rmm::cuda_stream_view stream)
    : base_type{default_allocator{}, stream}, stream_{stream}
  {
  }

  // TODO: If we move forward with prefetching, rather than having to implement
  // rebinding and override allocation we should just make the underlying
  // polymorphic_allocator wrap a custom prefetching allocator. That should be simpler.
  template <typename U>
  struct rebind {
    using other = cuco_allocator_base<U>;
  };

  typename base_type::value_type* allocate(std::size_t num)
  {
    auto ptr = base_type::allocate(num);
    cudf::experimental::prefetch::detail::prefetch(
      "cuco_allocator_base::allocate", ptr, num * sizeof(T));
    return ptr;
  }

 private:
  rmm::cuda_stream_view stream_;  ///< Stream on which (de)allocations are performed
};

using cuco_allocator = cuco_allocator_base<char>;

}  // namespace cudf::detail
