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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

namespace cudf {

namespace detail {

/**
 * @brief A non-owning view of scalar from device that is trivially copyable
 * and usable in CUDA device code.
 */
class scalar_device_view_base {
 public:
  ~scalar_device_view_base() = default;

  /**
   * @brief Returns the value type
   */
  __host__ __device__ data_type type() const noexcept { return _type; }

  /**
   * @brief Returns whether the scalar holds a valid value (i.e., not null).
   *
   * @return true The element is valid
   * @return false The element is null
   */
  __device__ bool is_valid() const noexcept {
    return *_is_valid;
  }

  /**
   * @brief Updates the validity of the value
   * 
   * @param is_valid true: set the value to valid. false: set it to null
   */
  __device__ void set_valid(bool is_valid) noexcept {
    *_is_valid = is_valid;
  }

 protected:
  data_type _type{EMPTY};   ///< Value data type
  bool * _is_valid{};       ///< Pointer to device memory containing
                            ///< boolean representing validity of the value.

  scalar_device_view_base(data_type type, bool* is_valid)
    : _type(type), _is_valid(is_valid) 
  {}

  scalar_device_view_base() = default;
};

/**
 * @brief A type of scalar_device_view where the value is a fixed width type
 */
template <typename T>
class fixed_width_scalar_device_view
    : public detail::scalar_device_view_base {
 public:
  using value_type = T;

  /**
   * @brief Returns reference to stored value.
   */
  __device__ T& value() noexcept {
    return *data();
  }

  /**
   * @brief Returns const reference to stored value.
   */
  __device__ T const& value() const noexcept {
    return *data();
  }

  __device__ void set_value(T value) {
    *_data = value;
  }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  __device__ T* data() noexcept {
    return static_cast<T*>(_data);
  }
  /**
   * @brief Returns a const raw pointer to the value in device memory
   */
  __device__ T const* data() const noexcept {
    return static_cast<T const*>(_data);
  }

 protected:
  T * _data{};      ///< Pointer to device memory containing the value

  /**
   * @brief Construct a new fixed width scalar device view object
   * 
   * This constructor should not be used directly. get_scalar_device_view
   * should be used to get the view of an existing scalar
   * 
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  fixed_width_scalar_device_view(data_type type, T* data, bool* is_valid)
   : detail::scalar_device_view_base(type, is_valid)
   , _data(data)
  {}
};

}  // namespace detail

/**
 * @brief A type of scalar_device_view that stores a pointer to a numerical value
 */
template <typename T>
class numeric_scalar_device_view
    : public detail::fixed_width_scalar_device_view<T>
{
 public:
  numeric_scalar_device_view(data_type type, T* data, bool* is_valid) 
    : detail::fixed_width_scalar_device_view<T>(type, data, is_valid)
  {}
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a string value
 */
class string_scalar_device_view
    : public detail::scalar_device_view_base {
 public:
  using ValueType = cudf::string_view;

  string_scalar_device_view(data_type type, const char* data, bool* is_valid,
                            size_type size) 
    : detail::scalar_device_view_base(type, is_valid), _data(data), _size(size)
  {}

  /**
   * @brief Returns string_view of the value of this scalar.
   */
  __device__ string_view value() const noexcept {
    return string_view(this->data(), _size);
  }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  __device__ char const* data() const noexcept {
    return static_cast<char const*>(_data);
  }

 private:
  const char* _data{};  ///< Pointer to device memory containing the value
  size_type _size;      ///< Size of the string in bytes
};

/**
 * @brief A type of scalar_device_view that stores a pointer to a timestamp value
 */
template <typename T>
class timestamp_scalar_device_view
    : public detail::fixed_width_scalar_device_view<T>
{
 public:
  timestamp_scalar_device_view(data_type type, T* data, bool* is_valid) 
    : detail::fixed_width_scalar_device_view<T>(type, data, is_valid)
  {}
};

/**
 * @brief Get the device view of a numeric_scalar
 */
template <typename T>
auto get_scalar_device_view(numeric_scalar<T>& s) {
  return numeric_scalar_device_view<T>(s.type(), s.data(), s.validity_data());
}

/**
 * @brief Get the device view of a string_scalar
 */
inline
auto get_scalar_device_view(string_scalar& s) {
  return string_scalar_device_view(s.type(), s.data(), s.validity_data(), s.size());
}

/**
 * @brief Get the device view of a timestamp_scalar
 */
template <typename T>
auto get_scalar_device_view(timestamp_scalar<T>& s) {
  return timestamp_scalar_device_view<T>(s.type(), s.data(), s.validity_data());
}

}  // namespace cudf
