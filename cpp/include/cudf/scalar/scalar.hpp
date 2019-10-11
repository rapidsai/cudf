/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include "column_view.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device_memory_resource.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

class numeric_scalar_device_view;
class string_scalar_device_view;
class timestamp_scalar_device_view;

namespace cudf {

class scalar {
 public:
  scalar() = default;
  ~scalar() = default;
  scalar& operator=(scalar const& other) = delete;
  scalar& operator=(scalar&& other) = delete;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar by deep copying the contents of `other`.
   *
   * All device memory allocation and copying is done using the
   * `device_memory_resource` and `stream` from `other`.
   *
   * @param other The scalar to copy
   *---------------------------------------------------------------------------**/
  scalar(scalar const& other);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar object by deep copying the contents of
   *`other`.
   *
   * Uses the specified `stream` and device_memory_resource for all allocations
   * and copies.
   *
   * @param other The `scalar` to copy
   * @param stream The stream on which to execute all allocations and copies
   * @param mr The resource to use for all allocations
   *---------------------------------------------------------------------------**/
  scalar(scalar const& other, cudaStream_t stream,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  /**---------------------------------------------------------------------------*
   * @brief Move the contents from `other` to create a new scalar.
   *
   * After the move, `other.type() = {EMPTY}`
   *
   * @param other The scalar whose contents will be moved into the new scalar
   *---------------------------------------------------------------------------**/
  scalar(scalar&& other);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar from existing device memory.
   *
   * @note This constructor is primarily intended for use in scalar factory
   * functions. 
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the scalar
   * @param[in] data The scalar's data
   * @param[in] is_valid Optional, scalar's null value indicator bitmask. May
   * be empty if `null_count` is 0 or `UNKNOWN_NULL_COUNT`.
   *---------------------------------------------------------------------------**/
  template <typename B1, typename B2 = rmm::device_scalar<bool>>
  scalar(data_type dtype, size_type size, B1&& data, B2&& is_valid = {})
      : _type{dtype},
        _size{size},
        _data{std::forward<B1>(data)},
        _is_valid{std::forward<B2>(is_valid)} {}

  /**---------------------------------------------------------------------------*
   * @brief Returns the scalar's logical element type
   *---------------------------------------------------------------------------**/
  data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Sets this scalar to null
   *---------------------------------------------------------------------------**/
  void set_null();

  /**---------------------------------------------------------------------------*
   * @brief Indicates whether the scalar contains a valid value
   *
   * @note Using the value when `is_valid() == false` is undefined behaviour
   * 
   * @return true Value is valid
   * @return false Value is invalid
   *---------------------------------------------------------------------------**/
  bool is_valid() const;

 protected:
  data_type _type{EMPTY};      ///< Logical type of elements in the scalar
  rmm::device_scalar<bool> _is_valid{};  ///< Device bool signifying validity
};

template <typename T>
class numeric_scalar {
  // TODO: prevent construction using anything other than arithmetic types

  // TODO: store value_type
  T value() { return _data.value(); }

  // TODO: implement
  numeric_scalar_device_view<T> device_view();

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing numeric value
};

class string_scalar {

  // TODO: store value_type
  std::string value() {} // TODO: implement

  // TODO: implement
  string_scalar_device_view device_view();

 protected:
  rmm::device_vector<char> _data{};  ///< device memory containing the string
};

template <typename T>
class timestamp_scalar {
  // TODO: prevent construction using anything other than timestamp types

  // TODO: store value_type
  T value() { return _data.value(); }

  // TODO: implement
  timestamp_scalar_device_view<T> device_view();

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing timestamp value
};

}  // namespace cudf
