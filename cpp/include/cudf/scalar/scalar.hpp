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
#include <cudf/utilities/type_dispatcher.hpp>
// #include "scalar_device_view.cuh"

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device_memory_resource.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// Forward declarations
template <typename T>
class numeric_scalar_device_view;

class string_scalar_device_view;

template <typename T>
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
   * @brief Move the contents from `other` to create a new scalar.
   *
   * After the move, `other.type() = {EMPTY}`
   *
   * @param other The scalar whose contents will be moved into the new scalar
   *---------------------------------------------------------------------------**/
  scalar(scalar&& other);

  /**---------------------------------------------------------------------------*
   * @brief Returns the scalar's logical element type
   *---------------------------------------------------------------------------**/
  data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Sets this scalar to null
   *---------------------------------------------------------------------------**/
  void set_null() { _is_valid.set_value(false); }

  /**---------------------------------------------------------------------------*
   * @brief Sets this scalar to valid
   *---------------------------------------------------------------------------**/
  void set_valid() { _is_valid.set_value(true); }

  /**---------------------------------------------------------------------------*
   * @brief Indicates whether the scalar contains a valid value
   *
   * @note Using the value when `is_valid() == false` is undefined behaviour
   * 
   * @return true Value is valid
   * @return false Value is invalid/null
   *---------------------------------------------------------------------------**/
  bool is_valid() const { return _is_valid.value(); }

 protected:
  data_type _type{EMPTY};      ///< Logical type of elements in the scalar
  rmm::device_scalar<bool> _is_valid{};  ///< Device bool signifying validity

  scalar(data_type type, bool is_valid) : _type(type), _is_valid(is_valid) {}
};

template <typename T>
class numeric_scalar : public scalar {
  // TODO: prevent construction using anything other than arithmetic types
 public:
  using ValueType = T;

  numeric_scalar(T value, bool is_valid = true)
   : scalar(data_type(experimental::type_to_id<T>()), is_valid), _data(value)
  {}

  void set_value(T value) { _data.set_value(value); }
  T value() { return _data.value(); }

  // TODO: implement
  // numeric_scalar_device_view<T> device_view() {
  //   return numeric_scalar_device_view<T>(this->_type, this->_data.get(),
  //                                        this->_is_valid.get());
  // }

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing numeric value
};

class string_scalar : public scalar {
 public:
  using ValueType = cudf::string_view;

  // TODO: implement with copying string
  string_scalar(std::string& string) : scalar(data_type(STRING), true) {}

  std::string value() {} // TODO: implement

  // TODO: implement
  // string_scalar_device_view device_view() {
  //   return string_scalar_device_view(this->_type, this->_data.data().get(),
  //                                    this->_is_valid.get(), this->_data.size());
  // }

 protected:
  rmm::device_vector<char> _data{};  ///< device memory containing the string
};

template <typename T>
class timestamp_scalar : public scalar {
  // TODO: prevent construction using anything other than timestamp types
 public:

  using ValueType = T;

  timestamp_scalar(T value, bool is_valid = true)
   : scalar(experimental::type_to_id<T>(), is_valid), _data(value)
  {}

  void set_value() { _data.set_value(value); }
  T value() { return _data.value(); }

  // TODO: implement
  // timestamp_scalar_device_view<T> device_view() {
  //   return timestamp_scalar_device_view<T>(this->_type, this->_data.get(),
  //                                          this->_is_valid.get());
  // }

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing timestamp value
};

}  // namespace cudf
