/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_CALLABLE __host__ __device__ inline
#define CUDA_DEVICE_CALLABLE __device__ inline
#else
#define CUDA_HOST_DEVICE_CALLABLE inline
#define CUDA_DEVICE_CALLABLE inline
#endif

#include <thrust/optional.h>  // TODO no idea why this is needed ¯\_(ツ)_/¯

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>

/**
 * @file
 * @brief Type declarations for libcudf.
 *
 **/

/**
 * @brief Forward declaration of cudaStream_t
 **/
using cudaStream_t = struct CUstream_st*;

namespace bit_mask {
using bit_mask_t = uint32_t;
}

// Forward declarations
namespace rmm {
class device_buffer;
namespace mr {
class device_memory_resource;
device_memory_resource* get_current_device_resource();
}  // namespace mr

}  // namespace rmm

namespace cudf {
// Forward declaration
class column;
class column_view;
class mutable_column_view;
class string_view;
class list_view;
class struct_view;

class scalar;

// clang-format off
class list_scalar;
class string_scalar;
template <typename T> class numeric_scalar;
template <typename T> class fixed_point_scalar;
template <typename T> class timestamp_scalar;
template <typename T> class duration_scalar;

class string_scalar_device_view;
template <typename T> class numeric_scalar_device_view;
template <typename T> class fixed_point_scalar_device_view;
template <typename T> class timestamp_scalar_device_view;
template <typename T> class duration_scalar_device_view;
// clang-format on

class struct_scalar;

class table;
class table_view;
class mutable_table_view;

/**
 * @addtogroup utility_types
 * @{
 * @file
 */

using size_type    = int32_t;
using bitmask_type = uint32_t;
using valid_type   = uint8_t;

/**
 * @brief Similar to `std::distance` but returns `cudf::size_type` and performs `static_cast`
 *
 * @tparam T Iterator type
 * @param f "first" iterator
 * @param l "last" iterator
 * @return size_type The distance between first and last
 */
template <typename T>
size_type distance(T f, T l)
{
  return static_cast<size_type>(std::distance(f, l));
}

/**
 * @brief Indicates an unknown null count.
 *
 * Use this value when constructing any column-like object to indicate that
 * the null count should be computed on the first invocation of `null_count()`.
 **/
static constexpr size_type UNKNOWN_NULL_COUNT{-1};

/**
 * @brief Indicates the order in which elements should be sorted.
 **/
enum class order : bool {
  ASCENDING,  ///< Elements ordered from small to large
  DESCENDING  ///< Elements ordered from large to small
};

/**
 * @brief Enum to specify whether to include nulls or exclude nulls
 */
enum class null_policy : bool {
  EXCLUDE,  ///< exclude null elements
  INCLUDE   ///< include null elements
};

/**
 * @brief Enum to treat NaN floating point value as null or non-null element
 */
enum class nan_policy : bool {
  NAN_IS_NULL,  ///< treat nans as null elements
  NAN_IS_VALID  ///< treat nans as valid elements (non-null)
};

/**
 * @brief
 */
enum class null_equality : bool {
  EQUAL,   ///< nulls compare equal
  UNEQUAL  ///< nulls compare unequal
};

/**
 * @brief Indicates how null values compare against all other values.
 **/
enum class null_order : bool {
  AFTER,  ///< NULL values ordered *after* all other values
  BEFORE  ///< NULL values ordered *before* all other values
};

/**
 * @brief Indicates whether a collection of values is known to be sorted.
 */
enum class sorted : bool { NO, YES };

/**
 * @brief Indicates how a collection of values has been ordered.
 **/
struct order_info {
  sorted is_sorted;
  order ordering;
  null_order null_ordering;
};

/**
 * @brief Controls the allocation/initialization of a null mask.
 **/
enum class mask_state : int32_t {
  UNALLOCATED,    ///< Null mask not allocated, (all elements are valid)
  UNINITIALIZED,  ///< Null mask allocated, but not initialized
  ALL_VALID,      ///< Null mask allocated, initialized to all elements valid
  ALL_NULL        ///< Null mask allocated, initialized to all elements NULL
};

/**
 * @brief Interpolation method to use when the desired quantile lies between
 * two data points i and j
 */
enum class interpolation : int32_t {
  LINEAR,    ///< Linear interpolation between i and j
  LOWER,     ///< Lower data point (i)
  HIGHER,    ///< Higher data point (j)
  MIDPOINT,  ///< (i + j)/2
  NEAREST    ///< i or j, whichever is nearest
};

/**
 * @brief Identifies a column's logical element type
 **/
enum class type_id : int32_t {
  EMPTY,                   ///< Always null with no underlying data
  INT8,                    ///< 1 byte signed integer
  INT16,                   ///< 2 byte signed integer
  INT32,                   ///< 4 byte signed integer
  INT64,                   ///< 8 byte signed integer
  UINT8,                   ///< 1 byte unsigned integer
  UINT16,                  ///< 2 byte unsigned integer
  UINT32,                  ///< 4 byte unsigned integer
  UINT64,                  ///< 8 byte unsigned integer
  FLOAT32,                 ///< 4 byte floating point
  FLOAT64,                 ///< 8 byte floating point
  BOOL8,                   ///< Boolean using one byte per value, 0 == false, else true
  TIMESTAMP_DAYS,          ///< point in time in days since Unix Epoch in int32
  TIMESTAMP_SECONDS,       ///< point in time in seconds since Unix Epoch in int64
  TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in int64
  TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in int64
  TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in int64
  DURATION_DAYS,           ///< time interval of days in int32
  DURATION_SECONDS,        ///< time interval of seconds in int64
  DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
  DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
  DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
  DICTIONARY32,            ///< Dictionary type using int32 indices
  STRING,                  ///< String elements
  LIST,                    ///< List elements
  DECIMAL32,               ///< Fixed-point type with int32_t
  DECIMAL64,               ///< Fixed-point type with int64_t
  STRUCT,                  ///< Struct elements
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS  ///< Total number of type ids
};

/**
 * @brief Indicator for the logical data type of an element in a column.
 *
 * Simple types can be be entirely described by their `id()`, but some types
 * require additional metadata to fully describe elements of that type.
 **/
class data_type {
 public:
  data_type()                 = default;
  ~data_type()                = default;
  data_type(data_type const&) = default;
  data_type(data_type&&)      = default;
  data_type& operator=(data_type const&) = default;
  data_type& operator=(data_type&&) = default;

  /**
   * @brief Construct a new `data_type` object
   *
   * @param id The type's identifier
   **/
  explicit constexpr data_type(type_id id) : _id{id} {}

  /**
   * @brief Construct a new `data_type` object for `numeric::fixed_point`
   *
   * @param id The `fixed_point`'s identifier
   * @param scale The `fixed_point`'s scale (see `fixed_point::_scale`)
   **/
  explicit data_type(type_id id, int32_t scale) : _id{id}, _fixed_point_scale{scale}
  {
    assert(id == type_id::DECIMAL32 || id == type_id::DECIMAL64);
  }

  /**
   * @brief Returns the type identifier
   **/
  CUDA_HOST_DEVICE_CALLABLE type_id id() const noexcept { return _id; }

  /**
   * @brief Returns the scale (for fixed_point types)
   **/
  CUDA_HOST_DEVICE_CALLABLE int32_t scale() const noexcept { return _fixed_point_scale; }

 private:
  type_id _id{type_id::EMPTY};

  // Below is additional type specific metadata. Currently, only _fixed_point_scale is stored.

  int32_t _fixed_point_scale{};  // numeric::scale_type not available here, use int32_t
};

/**
 * @brief Compares two `data_type` objects for equality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is equal to `rhs`
 * @return false `lhs` is not equal to `rhs`
 */
inline bool operator==(data_type const& lhs, data_type const& rhs) { return lhs.id() == rhs.id(); }

/**
 * @brief Compares two `data_type` objects for inequality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is not equal to `rhs`
 * @return false `lhs` is equal to `rhs`
 */
inline bool operator!=(data_type const& lhs, data_type const& rhs) { return !(lhs == rhs); }

/**
 * @brief Returns the size in bytes of elements of the specified `data_type`
 *
 * @note Only fixed-width types are supported
 *
 * @throws cudf::logic_error if `is_fixed_width(element_type) == false`
 *
 * @return Size in bytes of an element of the specified `data_type`
 */
std::size_t size_of(data_type t);

/**
 *  @brief Identifies the hash function to be used
 */
enum class hash_id {
  HASH_IDENTITY = 0,  ///< Identity hash function that simply returns the key to be hashed
  HASH_MURMUR3,       ///< Murmur3 hash function
  HASH_MD5            ///< MD5 hash function
};

/** @} */
}  // namespace cudf
