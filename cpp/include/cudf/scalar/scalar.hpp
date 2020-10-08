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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf/fixed_point/fixed_point.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

#include <memory>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Class definitions for cudf::scalar
 */

namespace cudf {
/**
 * @addtogroup scalar_classes
 * @{
 */

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
  virtual ~scalar()           = default;
  scalar(scalar&& other)      = default;
  scalar(scalar const& other) = default;
  scalar& operator=(scalar const& other) = delete;
  scalar& operator=(scalar&& other) = delete;

  /**
   * @brief Returns the scalar's logical value type
   */
  data_type type() const noexcept { return _type; }

  /**
   * @brief Updates the validity of the value
   *
   * @param is_valid true: set the value to valid. false: set it to null
   * @param stream CUDA stream used for device memory operations.
   */
  void set_valid(bool is_valid, cudaStream_t stream = 0) { _is_valid.set_value(is_valid, stream); }

  /**
   * @brief Indicates whether the scalar contains a valid value
   *
   * @note Using the value when `is_valid() == false` is undefined behaviour
   *
   * @param stream CUDA stream used for device memory operations.
   * @return true Value is valid
   * @return false Value is invalid/null
   */
  bool is_valid(cudaStream_t stream = 0) const { return _is_valid.value(stream); }

  /**
   * @brief Returns a raw pointer to the validity bool in device memory
   */
  bool* validity_data() { return _is_valid.data(); }

  /**
   * @brief Returns a const raw pointer to the validity bool in device memory
   */
  bool const* validity_data() const { return _is_valid.data(); }

 protected:
  data_type _type{type_id::EMPTY};       ///< Logical type of value in the scalar
  rmm::device_scalar<bool> _is_valid{};  ///< Device bool signifying validity

  scalar() = default;

  /**
   * @brief Construct a new scalar object
   *
   * @note Do not use this constructor directly. Instead, use a factory method
   * like make_numeric_scalar or make_string_scalar
   *
   * @param[in] type Data type of the scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  scalar(data_type type,
         bool is_valid                       = false,
         cudaStream_t stream                 = 0,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : _type(type), _is_valid(is_valid, stream, mr)
  {
  }
};

namespace detail {
template <typename T>
class fixed_width_scalar : public scalar {
  static_assert(is_fixed_width<T>(), "Unexpected non-fixed-width type.");

 public:
  using value_type = T;

  ~fixed_width_scalar()                               = default;
  fixed_width_scalar(fixed_width_scalar&& other)      = default;
  fixed_width_scalar(fixed_width_scalar const& other) = default;
  fixed_width_scalar& operator=(fixed_width_scalar const& other) = delete;
  fixed_width_scalar& operator=(fixed_width_scalar&& other) = delete;

  /**
   * @brief Set the value of the scalar
   *
   * @param value New value of scalar
   * @param stream CUDA stream used for device memory operations.
   */
  void set_value(T value, cudaStream_t stream = 0)
  {
    _data.set_value(value, stream);
    this->set_valid(true, stream);
  }

  /**
   * @brief Implicit conversion operator to get the value of the scalar on the host
   */
  explicit operator value_type() const { return this->value(0); }

  /**
   * @brief Get the value of the scalar
   *
   * @param stream CUDA stream used for device memory operations.
   */
  T value(cudaStream_t stream = 0) const { return _data.value(stream); }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  T* data() { return _data.data(); }

  /**
   * @brief Returns a const raw pointer to the value in device memory
   */
  T const* data() const { return _data.data(); }

 protected:
  rmm::device_scalar<T> _data{};  ///< device memory containing the value

  fixed_width_scalar() : scalar(data_type(type_to_id<T>())) {}

  /**
   * @brief Construct a new fixed width scalar object
   *
   * @param[in] value The initial value of the scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_width_scalar(T value,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar(data_type(type_to_id<T>()), is_valid, stream, mr), _data(value, stream, mr)
  {
  }

  /**
   * @brief Construct a new fixed width scalar object from existing device memory.
   *
   * @param[in] data The scalar's data in device memory
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_width_scalar(rmm::device_scalar<T>&& data,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar(data_type(type_to_id<T>()), is_valid, stream, mr),
      _data{std::forward<rmm::device_scalar<T>>(data)}
  {
  }
};

}  // namespace detail

/**
 * @brief An owning class to represent a numerical value in device memory
 *
 * @tparam T the data type of the numerical value
 */
template <typename T>
class numeric_scalar : public detail::fixed_width_scalar<T> {
  static_assert(is_numeric<T>(), "Unexpected non-numeric type.");

 public:
  numeric_scalar()                            = default;
  ~numeric_scalar()                           = default;
  numeric_scalar(numeric_scalar&& other)      = default;
  numeric_scalar(numeric_scalar const& other) = default;
  numeric_scalar& operator=(numeric_scalar const& other) = delete;
  numeric_scalar& operator=(numeric_scalar&& other) = delete;

  /**
   * @brief Construct a new numeric scalar object
   *
   * @param[in] value The initial value of the scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  numeric_scalar(T value,
                 bool is_valid                       = true,
                 cudaStream_t stream                 = 0,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
  {
  }

  /**
   * @brief Construct a new numeric scalar object from existing device memory.
   *
   * @param[in] data The scalar's data in device memory
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  numeric_scalar(rmm::device_scalar<T>&& data,
                 bool is_valid                       = true,
                 cudaStream_t stream                 = 0,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
  {
  }
};

/**
 * @brief An owning class to represent a fixed_point number in device memory
 *
 * @tparam T the data type of the fixed_point number
 */
template <typename T>
class fixed_point_scalar : public scalar {
  static_assert(is_fixed_point<T>(), "Unexpected non-fixed_point type.");

 public:
  using rep_type = typename T::rep;

  fixed_point_scalar() : scalar(data_type(type_to_id<T>())){};
  ~fixed_point_scalar()                               = default;
  fixed_point_scalar(fixed_point_scalar&& other)      = default;
  fixed_point_scalar(fixed_point_scalar const& other) = default;
  fixed_point_scalar& operator=(fixed_point_scalar const& other) = delete;
  fixed_point_scalar& operator=(fixed_point_scalar&& other) = delete;

  /**
   * @brief Construct a new fixed_point scalar object from already shifted value and scale
   *
   * @param[in] value The initial shifted value of the fixed_point scalar
   * @param[in] scale The scale of the fixed_point scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_point_scalar(rep_type value,
                     numeric::scale_type scale,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar{data_type{type_to_id<T>(), static_cast<int32_t>(scale)}, is_valid, stream, mr},
      _data{value}
  {
  }

  /**
   * @brief Construct a new fixed_point scalar object from a value and default 0-scale
   *
   * @param[in] value The initial value of the fixed_point scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_point_scalar(rep_type value,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar{data_type{type_to_id<T>(), 0}, is_valid, stream, mr}, _data{value}
  {
  }

  /**
   * @brief Construct a new fixed_point scalar object from a fixed_point number
   *
   * @param[in] value The fixed_point number from which the fixed_point scalar will be initialized
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_point_scalar(T value,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar{data_type{type_to_id<T>(), 0}, is_valid, stream, mr},
      _data{numeric::scaled_integer<rep_type>{value}.value}
  {
    CUDF_EXPECTS(value == (T{_data.value(), numeric::scale_type{0}}),
                 "scale of fixed_point value should be zero");
  }

  /**
   * @brief Construct a new fixed_point scalar object from existing device memory.
   *
   * @param[in] data The scalar's data in device memory
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  fixed_point_scalar(rmm::device_scalar<rep_type>&& data,
                     bool is_valid                       = true,
                     cudaStream_t stream                 = 0,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar{data_type{type_to_id<T>()}, is_valid, stream, mr},  // note that scale is ignored here
      _data{std::forward<rmm::device_scalar<rep_type>>(data)}
  {
  }

  /**
   * @brief Get the value of the scalar
   *
   * @param stream CUDA stream used for device memory operations.
   */
  rep_type value(cudaStream_t stream = 0) const { return _data.value(stream); }

  /**
   * @brief Returns a raw pointer to the value in device memory
   */
  rep_type* data() { return _data.data(); }

  /**
   * @brief Returns a const raw pointer to the value in device memory
   */
  rep_type const* data() const { return _data.data(); }

 protected:
  rmm::device_scalar<rep_type> _data{};  ///< device memory containing the value
};

/**
 * @brief An owning class to represent a string in device memory
 */
class string_scalar : public scalar {
 public:
  using value_type = cudf::string_view;

  string_scalar() : scalar(data_type(type_id::STRING)) {}
  ~string_scalar()                          = default;
  string_scalar(string_scalar&& other)      = default;
  string_scalar(string_scalar const& other) = default;
  string_scalar& operator=(string_scalar const& other) = delete;
  string_scalar& operator=(string_scalar&& other) = delete;

  /**
   * @brief Construct a new string scalar object
   *
   * @param[in] value The value of the string
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  string_scalar(std::string const& string,
                bool is_valid                       = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar(data_type(type_id::STRING), is_valid), _data(string.data(), string.size(), stream, mr)
  {
  }

  /**
   * @brief Construct a new string scalar object from string_view
   * Note that this function copies the data pointed by string_view.
   *
   * @param[in] source string_view pointing string value to copy
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  string_scalar(value_type const& source,
                bool is_valid                       = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : scalar(data_type(type_id::STRING), is_valid),
      _data(source.data(), source.size_bytes(), stream, mr)
  {
  }

  /**
   * @brief Construct a new string scalar object from string_view in device memory
   * Note that this function copies the data pointed by string_view.
   *
   * @param[in] data device_scalar string_view pointing string value to copy
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  string_scalar(rmm::device_scalar<value_type>& data,
                bool is_valid                       = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : string_scalar(data.value(stream), is_valid, stream, mr)
  {
  }

  /**
   * @brief Implicit conversion operator to get the value of the scalar in a host std::string
   */
  explicit operator std::string() const { return this->to_string(0); }

  /**
   * @brief Get the value of the scalar in a host std::string
   *
   * @param stream CUDA stream used for device memory operations.
   */
  std::string to_string(cudaStream_t stream = 0) const;

  /**
   * @brief Get the value of the scalar as a string_view
   *
   * @param stream CUDA stream used for device memory operations.
   */
  value_type value(cudaStream_t stream = 0) const { return value_type{data(), size()}; }

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
 * @brief An owning class to represent a timestamp/duration value in device memory
 *
 * @tparam T the data type of the timestamp/duration value
 * @see cudf/wrappers/timestamps.hpp, cudf/wrappers/durations.hpp for a list of allowed types
 */
template <typename T>
class chrono_scalar : public detail::fixed_width_scalar<T> {
  static_assert(is_chrono<T>(), "Unexpected non-chrono type");

 public:
  chrono_scalar()                           = default;
  ~chrono_scalar()                          = default;
  chrono_scalar(chrono_scalar&& other)      = default;
  chrono_scalar(chrono_scalar const& other) = default;
  chrono_scalar& operator=(chrono_scalar const& other) = delete;
  chrono_scalar& operator=(chrono_scalar&& other) = delete;

  /**
   * @brief Construct a new chrono scalar object
   *
   * @param[in] value The initial value of the scalar
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  chrono_scalar(T value,
                bool is_valid                       = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
  {
  }

  /**
   * @brief Construct a new chrono scalar object from existing device memory.
   *
   * @param[in] data The scalar's data in device memory
   * @param[in] is_valid Whether the value held by the scalar is valid
   * @param[in] stream CUDA stream used for device memory operations.
   * @param[in] mr Device memory resource to use for device memory allocation
   */
  chrono_scalar(rmm::device_scalar<T>&& data,
                bool is_valid                       = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
  {
  }
};

template <typename T>
struct timestamp_scalar : chrono_scalar<T> {
  static_assert(is_timestamp<T>(), "Unexpected non-timestamp type");
  using chrono_scalar<T>::chrono_scalar;

  timestamp_scalar() = default;

  /**
   * @brief Construct a new timestamp scalar object from a duration that is
   * convertible to T::duration
   *
   * @param value Duration representing number of ticks since the UNIX epoch or
   * another duration that is convertible to timestamps duration
   * @param is_valid Whether the value held by the scalar is valid
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation
   */
  template <typename Duration2>
  timestamp_scalar(Duration2 const& value,
                   bool is_valid,
                   cudaStream_t stream                 = 0,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : chrono_scalar<T>(T{typename T::duration{value}}, is_valid, stream, mr)
  {
  }

  /**
   * @brief Return the duration in number of ticks since the UNIX epoch.
   */
  typename T::rep ticks_since_epoch() { return this->value().time_since_epoch().count(); }
};

template <typename T>
struct duration_scalar : chrono_scalar<T> {
  static_assert(is_duration<T>(), "Unexpected non-duration type");
  using chrono_scalar<T>::chrono_scalar;

  duration_scalar() = default;

  /**
   * @brief Construct a new duration scalar object from tick counts
   *
   * @param value Integer representing number of ticks since the UNIX epoch
   * @param is_valid Whether the value held by the scalar is valid
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation
   */
  duration_scalar(typename T::rep value,
                  bool is_valid,
                  cudaStream_t stream                 = 0,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : chrono_scalar<T>(T{value}, is_valid, stream, mr)
  {
  }

  /**
   * @brief Return the duration in number of ticks.
   */
  typename T::rep count() { return this->value().count(); }
};
/** @} */  // end of group
}  // namespace cudf
