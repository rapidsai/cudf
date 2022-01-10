/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

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
 * @brief An owning class to represent a singular value.
 *
 * A scalar is a singular value of any of the supported datatypes in cudf.
 * Classes derived from this class are used to represent a scalar. Objects of
 * derived classes should be upcasted to this class while passing to an
 * external libcudf API.
 */
class scalar {
 public:
  virtual ~scalar() = default;
  scalar& operator=(scalar const& other) = delete;
  scalar& operator=(scalar&& other) = delete;

  /**
   * @brief Returns the scalar's logical value type.
   */
  data_type type() const noexcept;

  /**
   * @brief Updates the validity of the value.
   *
   * @param is_valid true: set the value to valid. false: set it to null.
   * @param stream CUDA stream used for device memory operations.
   */
  void set_valid_async(bool is_valid, rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Indicates whether the scalar contains a valid value.
   *
   * @note Using the value when `is_valid() == false` is undefined behavior. In addition, this
   * function does a stream synchronization.
   *
   * @param stream CUDA stream used for device memory operations.
   * @return true Value is valid.
   * @return false Value is invalid/null.
   */
  bool is_valid(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Returns a raw pointer to the validity bool in device memory.
   */
  bool* validity_data();

  /**
   * @brief Returns a const raw pointer to the validity bool in device memory.
   */
  bool const* validity_data() const;

 protected:
  data_type _type{type_id::EMPTY};     ///< Logical type of value in the scalar
  rmm::device_scalar<bool> _is_valid;  ///< Device bool signifying validity

  scalar() = delete;

  scalar(scalar&& other) = default;

  /**
   * @brief Construct a new scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  scalar(scalar const& other,
         rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new scalar object.
   *
   * @note Do not use this constructor directly. Instead, use a factory method
   * like make_numeric_scalar or make_string_scalar
   *
   * @param type Data type of the scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  scalar(data_type type,
         bool is_valid                       = false,
         rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

namespace detail {
template <typename T>
class fixed_width_scalar : public scalar {
  static_assert(is_fixed_width<T>(), "Unexpected non-fixed-width type.");

 public:
  using value_type = T;

  ~fixed_width_scalar()                          = default;
  fixed_width_scalar(fixed_width_scalar&& other) = default;

  fixed_width_scalar& operator=(fixed_width_scalar const& other) = delete;
  fixed_width_scalar& operator=(fixed_width_scalar&& other) = delete;

  /**
   * @brief Construct a new fixed-width scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_width_scalar(fixed_width_scalar const& other,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Set the value of the scalar.
   *
   * @param value New value of scalar.
   * @param stream CUDA stream used for device memory operations.
   */
  void set_value(T value, rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Explicit conversion operator to get the value of the scalar on the host.
   */
  explicit operator value_type() const;

  /**
   * @brief Get the value of the scalar.
   *
   * @param stream CUDA stream used for device memory operations.
   */
  T value(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Returns a raw pointer to the value in device memory.
   */
  T* data();

  /**
   * @brief Returns a const raw pointer to the value in device memory.
   */
  T const* data() const;

 protected:
  rmm::device_scalar<T> _data;  ///< device memory containing the value

  fixed_width_scalar() = delete;

  /**
   * @brief Construct a new fixed width scalar object.
   *
   * @param value The initial value of the scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_width_scalar(T value,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new fixed width scalar object from existing device memory.
   *
   * @param data The scalar's data in device memory.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_width_scalar(rmm::device_scalar<T>&& data,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

}  // namespace detail

/**
 * @brief An owning class to represent a numerical value in device memory.
 *
 * @tparam T the data type of the numerical value.
 */
template <typename T>
class numeric_scalar : public detail::fixed_width_scalar<T> {
  static_assert(is_numeric<T>(), "Unexpected non-numeric type.");

 public:
  numeric_scalar()                       = delete;
  ~numeric_scalar()                      = default;
  numeric_scalar(numeric_scalar&& other) = default;

  numeric_scalar& operator=(numeric_scalar const& other) = delete;
  numeric_scalar& operator=(numeric_scalar&& other) = delete;

  /**
   * @brief Construct a new numeric scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  numeric_scalar(numeric_scalar const& other,
                 rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new numeric scalar object.
   *
   * @param value The initial value of the scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  numeric_scalar(T value,
                 bool is_valid                       = true,
                 rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new numeric scalar object from existing device memory.
   *
   * @param data The scalar's data in device memory.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  numeric_scalar(rmm::device_scalar<T>&& data,
                 bool is_valid                       = true,
                 rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

/**
 * @brief An owning class to represent a fixed_point number in device memory.
 *
 * @tparam T the data type of the fixed_point number.
 */
template <typename T>
class fixed_point_scalar : public scalar {
  static_assert(is_fixed_point<T>(), "Unexpected non-fixed_point type.");

 public:
  using rep_type   = typename T::rep;
  using value_type = T;

  fixed_point_scalar()                           = delete;
  ~fixed_point_scalar()                          = default;
  fixed_point_scalar(fixed_point_scalar&& other) = default;

  fixed_point_scalar& operator=(fixed_point_scalar const& other) = delete;
  fixed_point_scalar& operator=(fixed_point_scalar&& other) = delete;

  /**
   * @brief Construct a new fixed_point scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_point_scalar(fixed_point_scalar const& other,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new fixed_point scalar object from already shifted value and scale.
   *
   * @param value The initial shifted value of the fixed_point scalar.
   * @param scale The scale of the fixed_point scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_point_scalar(rep_type value,
                     numeric::scale_type scale,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new fixed_point scalar object from a value and default 0-scale.
   *
   * @param value The initial value of the fixed_point scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_point_scalar(rep_type value,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new fixed_point scalar object from a fixed_point number.
   *
   * @param value The fixed_point number from which the fixed_point scalar will be initialized.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_point_scalar(T value,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new fixed_point scalar object from existing device memory.
   *
   * @param data The scalar's data in device memory.
   * @param scale The scale of the fixed_point scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  fixed_point_scalar(rmm::device_scalar<rep_type>&& data,
                     numeric::scale_type scale,
                     bool is_valid                       = true,
                     rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Get the value of the scalar.
   *
   * @param stream CUDA stream used for device memory operations.
   */
  rep_type value(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Get the decimal32, decimal64 or decimal128.
   *
   * @param stream CUDA stream used for device memory operations.
   */
  T fixed_point_value(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Explicit conversion operator to get the value of the scalar on the host.
   */
  explicit operator value_type() const;

  /**
   * @brief Returns a raw pointer to the value in device memory.
   */
  rep_type* data();

  /**
   * @brief Returns a const raw pointer to the value in device memory.
   */
  rep_type const* data() const;

 protected:
  rmm::device_scalar<rep_type> _data;  ///< device memory containing the value
};

/**
 * @brief An owning class to represent a string in device memory.
 */
class string_scalar : public scalar {
 public:
  using value_type = cudf::string_view;

  string_scalar()                      = delete;
  ~string_scalar()                     = default;
  string_scalar(string_scalar&& other) = default;

  // string_scalar(string_scalar const& other) = delete;
  string_scalar& operator=(string_scalar const& other) = delete;
  string_scalar& operator=(string_scalar&& other) = delete;

  /**
   * @brief Construct a new string scalar object by deep copying another string_scalar.
   *
   * @param other The other string_scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  string_scalar(string_scalar const& other,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new string scalar object.
   *
   * @param string The value of the string.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  string_scalar(std::string const& string,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new string scalar object from string_view.
   *
   * Note that this function copies the data pointed by string_view.
   *
   * @param source The string_view pointing the string value to copy.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  string_scalar(value_type const& source,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new string scalar object from string_view in device memory.
   *
   * Note that this function copies the data pointed by string_view.
   *
   * @param data The device_scalar of string_view pointing to the string value to copy.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  string_scalar(rmm::device_scalar<value_type>& data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new string scalar object by moving an existing string data buffer.
   *
   * Note that this constructor moves the existing buffer into the internal data buffer;
   * no copy is performed.
   *
   * @param data The existing buffer to take over.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  string_scalar(rmm::device_buffer&& data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Explicit conversion operator to get the value of the scalar in a host std::string.
   */
  explicit operator std::string() const;

  /**
   * @brief Get the value of the scalar in a host std::string.
   *
   * @param stream CUDA stream used for device memory operations.
   */
  std::string to_string(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Get the value of the scalar as a string_view.
   *
   * @param stream CUDA stream used for device memory operations.
   */
  value_type value(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;

  /**
   * @brief Returns the size of the string in bytes.
   */
  size_type size() const;

  /**
   * @brief Returns a raw pointer to the string in device memory.
   */
  const char* data() const;

 protected:
  rmm::device_buffer _data{};  ///< device memory containing the string
};

/**
 * @brief An owning class to represent a timestamp/duration value in device memory.
 *
 * @tparam T the data type of the timestamp/duration value.
 * @see cudf/wrappers/timestamps.hpp, cudf/wrappers/durations.hpp for a list of allowed types.
 */
template <typename T>
class chrono_scalar : public detail::fixed_width_scalar<T> {
  static_assert(is_chrono<T>(), "Unexpected non-chrono type");

 public:
  chrono_scalar()                      = delete;
  ~chrono_scalar()                     = default;
  chrono_scalar(chrono_scalar&& other) = default;

  chrono_scalar& operator=(chrono_scalar const& other) = delete;
  chrono_scalar& operator=(chrono_scalar&& other) = delete;

  /**
   * @brief Construct a new chrono scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  chrono_scalar(chrono_scalar const& other,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new chrono scalar object.
   *
   * @param value The initial value of the scalar.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  chrono_scalar(T value,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new chrono scalar object from existing device memory.
   *
   * @param data The scalar's data in device memory.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  chrono_scalar(rmm::device_scalar<T>&& data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

template <typename T>
class timestamp_scalar : public chrono_scalar<T> {
 public:
  static_assert(is_timestamp<T>(), "Unexpected non-timestamp type");
  using chrono_scalar<T>::chrono_scalar;
  using rep_type = typename T::rep;

  timestamp_scalar()                         = delete;
  timestamp_scalar(timestamp_scalar&& other) = default;

  /**
   * @brief Construct a new timestamp scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  timestamp_scalar(timestamp_scalar const& other,
                   rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new timestamp scalar object from a duration that is
   * convertible to T::duration
   *
   * @param value Duration representing number of ticks since the UNIX epoch or another duration
   *        that is convertible to timestamps duration.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  template <typename Duration2>
  timestamp_scalar(Duration2 const& value,
                   bool is_valid,
                   rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Return the duration in number of ticks since the UNIX epoch.
   */
  rep_type ticks_since_epoch();
};

template <typename T>
class duration_scalar : public chrono_scalar<T> {
 public:
  static_assert(is_duration<T>(), "Unexpected non-duration type");
  using chrono_scalar<T>::chrono_scalar;
  using rep_type = typename T::rep;

  duration_scalar()                        = delete;
  duration_scalar(duration_scalar&& other) = default;

  /**
   * @brief Construct a new duration scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  duration_scalar(duration_scalar const& other,
                  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new duration scalar object from tick counts.
   *
   * @param value Integer representing number of ticks since the UNIX epoch.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  duration_scalar(rep_type value,
                  bool is_valid,
                  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Return the duration in number of ticks.
   */
  rep_type count();
};

/**
 * @brief An owning class to represent a list value in device memory.
 */
class list_scalar : public scalar {
 public:
  list_scalar()                    = delete;
  ~list_scalar()                   = default;
  list_scalar(list_scalar&& other) = default;

  list_scalar& operator=(list_scalar const& other) = delete;
  list_scalar& operator=(list_scalar&& other) = delete;

  /**
   * @brief Construct a new list scalar object by deep copying another.
   *
   * @param other The scalar to copy.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  list_scalar(list_scalar const& other,
              rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new list scalar object from column_view.
   *
   * The input column_view is copied.
   *
   * @param data The column data to copy.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  list_scalar(cudf::column_view const& data,
              bool is_valid                       = true,
              rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new list scalar object from existing column.
   *
   * @param data The column to take ownership of.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  list_scalar(cudf::column&& data,
              bool is_valid                       = true,
              rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Returns a non-owning, immutable view to underlying device data.
   */
  column_view view() const;

 private:
  cudf::column _data;
};

/**
 * @brief An owning class to represent a struct value in device memory.
 */
class struct_scalar : public scalar {
 public:
  struct_scalar()                      = delete;
  ~struct_scalar()                     = default;
  struct_scalar(struct_scalar&& other) = default;
  struct_scalar& operator=(struct_scalar const& other) = delete;
  struct_scalar& operator=(struct_scalar&& other) = delete;

  struct_scalar(struct_scalar const& other,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new struct scalar object from table_view.
   *
   * The input table_view is deep-copied.
   *
   * @param data The table data to copy.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  struct_scalar(table_view const& data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new struct scalar object from a host_span of column_views.
   *
   * The input column_views are deep-copied.
   *
   * @param data The column_views to copy.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  struct_scalar(host_span<column_view const> data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Construct a new struct scalar object from an existing table in device memory.
   *
   * Note that this constructor moves the existing table data into the internal table data;
   * no copies are performed.
   *
   * @param data The existing table data to take over.
   * @param is_valid Whether the value held by the scalar is valid.
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for device memory allocation.
   */
  struct_scalar(table&& data,
                bool is_valid                       = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Returns a non-owning, immutable view to underlying device data.
   */
  table_view view() const;

 private:
  table _data;

  void init(bool is_valid, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr);
  void superimpose_nulls(rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr);
};

/** @} */  // end of group
}  // namespace cudf
