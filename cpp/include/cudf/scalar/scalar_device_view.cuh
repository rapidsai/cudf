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

#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.cuh>

namespace cudf {

namespace detail {

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of scalar from device that is trivially 
 * copyable and usable in CUDA device code.
 *---------------------------------------------------------------------------**/
class alignas(16) scalar_device_view_base {
 public:
  ~scalar_device_view_base() = default;

  /**---------------------------------------------------------------------------*
   * @brief Returns the element type
   *---------------------------------------------------------------------------**/
  __host__ __device__ data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Returns whether the scalar holds a valid value (i.e., not null).
   *
   * @return true The element is valid
   * @return false The element is null
   *---------------------------------------------------------------------------**/
  __device__ bool is_valid() const noexcept {
    return *_is_valid;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns whether the value is null.
   *
   * @return true The element is null
   * @return false The element is valid
   *---------------------------------------------------------------------------**/
  __device__ bool is_null() const noexcept {
    return not is_valid();
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the null mask to indicate that the value is valid
   *---------------------------------------------------------------------------**/
  __device__ void set_valid() noexcept {
    *_is_valid = true;
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the null mask to indicate that the value is null
   *---------------------------------------------------------------------------**/
  __device__ void set_null() noexcept {
    *_is_valid = false;
  }

 protected:
  data_type _type{EMPTY};   ///< Element type
  void * _data{};      ///< Pointer to device memory containing the value
  bool * _is_valid{};  ///< Pointer to device memory containing
                       ///< boolean representing validity of the value.

  scalar_device_view_base() = default;
};

template <typename T>
class alignas(16) primitive_scalar_device_view
    : public detail::scalar_device_view_base {
 public:

  /**---------------------------------------------------------------------------*
   * @brief Returns reference to element at the specified index.
   *---------------------------------------------------------------------------**/
  __device__ T& value() noexcept {
    return *get();
  }

  __device__ T const& value() const noexcept {
    return *get();
  }

  __device__ T* get() noexcept {
    return static_cast<T*>(_data);
  }

  __device__ T const* get() const noexcept {
    return static_cast<T const*>(_data);
  }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Construct's a `primitive_scalar_device_view` from a
   *`mutable_column_view` populating all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`primitive_scalar_device_view` from a `column_view`, the
   *`primitive_scalar_device_view::create()` function should be used.
   *---------------------------------------------------------------------------**/
  primitive_scalar_device_view(mutable_column_view source);
};

}  // namespace detail


template <typename T>
class alignas(16) numeric_scalar_device_view
    : public detail::primitive_scalar_device_view<T> {
  using ValueType = T;
};

class alignas(16) string_scalar_device_view
    : public detail::scalar_device_view_base {
 public:
  using ValueType = cudf::string_view;

  /**---------------------------------------------------------------------------*
   * @brief Returns string_view of the value of this scalar.
   *---------------------------------------------------------------------------**/
  __device__ string_view value() const noexcept {
    return string_view(this->get(), _size);
  }

  __device__ char const* get() const noexcept {
    return static_cast<char const*>(_data);
  }

 private:
  size_type _size;  ///< Length of the string
  /**---------------------------------------------------------------------------*
   * @brief Construct's a `string_scalar_device_view` from a
   *`mutable_column_view` populating all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`string_scalar_device_view` from a `column_view`, the
   *`string_scalar_device_view::create()` function should be used.
   *---------------------------------------------------------------------------**/
  string_scalar_device_view(mutable_column_view source);
};

template <typename T>
class alignas(16) timestamp_scalar_device_view
    : public detail::primitive_scalar_device_view {
  using ValueType = T;
};


}  // namespace cudf
