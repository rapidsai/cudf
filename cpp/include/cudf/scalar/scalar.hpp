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
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device_memory_resource.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {

/**
 * @brief An owning class to represent a singular value
 * 
 * A scalar is a singular value of any of the supported datatypes in cudf. 
 * Classes derived from this class are used to represent a scalar. Objects of
 * derived classes should be upcasted to this class while passing to an
 * external libcudf API.
 */
class scalar {
 public:
  scalar() = default;
  ~scalar() = default;
  scalar(scalar&& other) = default;
  scalar(scalar const& other) = default;
  scalar& operator=(scalar const& other) = delete;
  scalar& operator=(scalar&& other) = delete;
  
  /**
   * @brief Construct a new scalar object
   * 
   * @note Do not use this constructor directly. Instead, use a factory method
   * like make_numeric_scalar or make_string_scalar
   * 
   * @param type Data type of the scalar
   * @param is_valid Whether the value held by the scalar is valid
   * @param stream The CUDA stream to do the allocation in
   * @param mr The memory resource to use for allocation
   */
  scalar(data_type type, bool is_valid = false, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
   : _type(type), _is_valid(is_valid, stream, mr)
  {}

  /**
   * @brief Returns the scalar's logical value type
   */
  data_type type() const noexcept { return _type; }

  /**
   * @brief Updates the validity of the value
   * 
   * @param is_valid true: set the value to valid. false: set it to null
   */
  void set_valid(bool is_valid) { _is_valid.set_value(is_valid); }

  /**
   * @brief Indicates whether the scalar contains a valid value
   *
   * @note Using the value when `is_valid() == false` is undefined behaviour
   * 
   * @return true Value is valid
   * @return false Value is invalid/null
   */
  bool is_valid() const { return _is_valid.value(); }

  /**
   * @brief Returns a raw pointer to the validity bool in device memory
   */
  bool* validity_data() { return _is_valid.get(); }

 protected:
  data_type _type{EMPTY};      ///< Logical type of value in the scalar
  rmm::device_scalar<bool> _is_valid{};  ///< Device bool signifying validity
};

/**
 * @brief An owning class to represent a numerical value in device memory
 * 
 * @tparam T the data type of the numerical value
 */
template <typename T>
class numeric_scalar : public scalar {
   static_assert(is_numeric<T>(), "Unexpected non-numeric type.");
 public:
  using value_type = T;

  numeric_scalar() = default;
  ~numeric_scalar() = default;
  numeric_scalar(numeric_scalar&& other) = default;
  numeric_scalar(numeric_scalar const& other) = default;

  /**
   * @brief Construct a new numeric scalar object
   * 
   * @param value The initial value of the scalar
   * @param is_valid Whether the value held by the scalar is valid
   * @param stream The CUDA stream to do the allocation in
   * @param mr The memory resource to use for allocation
   */
  numeric_scalar(T value, bool is_valid = true, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
   : scalar(data_type(experimental::type_to_id<T>()), is_valid, stream, mr)
   , _data(value, stream, mr)
  {}

  /**
   * @brief Set the value of the scalar
   */
  void set_value(T value) { 
    _data.set_value(value); 
    this->set_valid(true); 
  }

  /**
   * @brief Get the value of the scalar
   */
  T value() { return _data.value(); }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  T* data() { return _data.get(); }

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing numeric value
};

/**
 * @brief An owning class to represent a string in device memory
 */
class string_scalar : public scalar {
 public:
  using value_type = cudf::string_view;

  string_scalar() = default;
  ~string_scalar() = default;
  string_scalar(string_scalar&& other) = default;
  string_scalar(string_scalar const& other) = default;

  // TODO: stream and memory resource
  string_scalar(std::string const& string, bool is_valid = true);

  /**
   * @brief Get the value of the scalar in a host std::string
   */
  std::string value() const;
  
  /**
   * @brief Returns the size of the string in bytes
   */
  size_type size() const { return _data.size(); }

  /**
   * @brief Returns a raw pointer to the string in device memory
   */
  const char* data() const { return static_cast<const char*>(_data.data()); }

 protected:
  rmm::device_buffer _data{};  ///< device memory containing the string
};

/**
 * @brief An owning class to represent a timestamp value in device memory
 * 
 * @tparam T the data type of the timestamp value
 * @see cudf/wrappers/timestamps.hpp for a list of allowed types
 */
template <typename T>
class timestamp_scalar : public scalar {
  static_assert(is_timestamp<T>(), "Unexpected non-timestamp type");
 public:

  using value_type = T;

  timestamp_scalar() = default;
  ~timestamp_scalar() = default;
  timestamp_scalar(timestamp_scalar&& other) = default;
  timestamp_scalar(timestamp_scalar const& other) = default;

  /**
   * @brief Construct a new timestamp scalar object
   * 
   * @param value The initial value of the scalar
   * @param is_valid Whether the value held by the scalar is valid
   * @param stream The CUDA stream to do the allocation in
   * @param mr The memory resource to use for allocation
   */
  timestamp_scalar(T value, bool is_valid = true, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
   : scalar(data_type(experimental::type_to_id<T>()), is_valid, stream, mr)
   , _data(value, stream, mr)
  {}

  /**
   * @brief Set the value of the scalar
   */
  void set_value(T value) { 
    _data.set_value(value); 
    this->set_valid(true); 
  }

  /**
   * @brief Get the value of the scalar
   */
  T value() { return _data.value(); }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  T* data() { return _data.get(); }

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing timestamp value
};

}  // namespace cudf
