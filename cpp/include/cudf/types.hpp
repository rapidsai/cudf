/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>

/**---------------------------------------------------------------------------*
 * @file types.hpp
 * @brief Type declarations for libcudf.
 *
 *---------------------------------------------------------------------------**/

/**---------------------------------------------------------------------------*
 * @brief Forward declaration of cudaStream_t
 *---------------------------------------------------------------------------**/
using cudaStream_t = struct CUstream_st*;

namespace bit_mask {
using bit_mask_t = uint32_t;
}

// Forward declarations
namespace rmm {
class device_buffer;
namespace mr {
class device_memory_resource;
device_memory_resource* get_default_resource();
}  // namespace mr

}  // namespace rmm

namespace cudf {

// Forward declaration
struct table;
class column;
class column_view;
class mutable_column_view;
class string_view;

class scalar;
template <typename T> class numeric_scalar;
class string_scalar;
template <typename T> class timestamp_scalar;

template <typename T> class numeric_scalar_device_view;
class string_scalar_device_view;
template <typename T> class timestamp_scalar_device_view;



namespace experimental {
class table;
}
class table_view;
class mutable_table_view;

using size_type = int32_t;
using bitmask_type = uint32_t;
using valid_type = uint8_t;

/**---------------------------------------------------------------------------*
 * @brief Indicates an unknown null count.
 *
 * Use this value when constructing any column-like object to indicate that
 * the null count should be computed on the first invocation of `null_count()`.
 *---------------------------------------------------------------------------**/
static constexpr size_type UNKNOWN_NULL_COUNT{-1};

/**---------------------------------------------------------------------------*
 * @brief Indicates the order in which elements should be sorted.
 *---------------------------------------------------------------------------**/
enum class order : bool {
  ASCENDING,  ///< Elements ordered from small to large
  DESCENDING  ///< Elements ordered from large to small
};

/**---------------------------------------------------------------------------*
 * @brief Indicates how null values compare against all other values.
 *---------------------------------------------------------------------------**/
enum class null_order : bool {
  AFTER,  ///< NULL values ordered *after* all other values
  BEFORE  ///< NULL values ordered *before* all other values
};

/**---------------------------------------------------------------------------*
 * @brief Indicates how a collection of values has been ordered.
 *---------------------------------------------------------------------------**/
struct order_info {
   bool is_ordered;
   order ordering;
   null_order null_ordering;
};

/**---------------------------------------------------------------------------*
 * @brief Controls the allocation/initialization of a null mask.
 *---------------------------------------------------------------------------**/
enum mask_state {
  UNALLOCATED,    ///< Null mask not allocated, (all elements are valid)
  UNINITIALIZED,  ///< Null mask allocated, but not initialized
  ALL_VALID,      ///< Null mask allocated, initialized to all elements valid
  ALL_NULL        ///< Null mask allocated, initialized to all elements NULL
};

namespace experimental{
/**
 * @brief Interpolation method to use when the desired quantile lies between
 * two data points i and j
 *
 */
enum class interpolation {
    LINEAR,  ///< Linear interpolation between i and j
    LOWER,       ///< Lower data point (i)
    HIGHER,      ///< Higher data point (j)
    MIDPOINT,    ///< (i + j)/2
    NEAREST      ///< i or j, whichever is nearest
};
}
/**---------------------------------------------------------------------------*
 * @brief Identifies a column's logical element type
 *---------------------------------------------------------------------------**/
enum type_id {
  EMPTY = 0,  ///< Always null with no underlying data
  INT8,       ///< 1 byte signed integer
  INT16,      ///< 2 byte signed integer
  INT32,      ///< 4 byte signed integer
  INT64,      ///< 8 byte signed integer
  FLOAT32,    ///< 4 byte floating point
  FLOAT64,    ///< 8 byte floating point
  BOOL8,      ///< Boolean using one byte per value, 0 == false, else true
  TIMESTAMP_DAYS,     ///< days since Unix Epoch in int32
  TIMESTAMP_SECONDS,  ///< duration of seconds since Unix Epoch in int64
  TIMESTAMP_MILLISECONDS,  ///< duration of milliseconds since Unix Epoch in int64
  TIMESTAMP_MICROSECONDS,  ///< duration of microseconds since Unix Epoch in int64
  TIMESTAMP_NANOSECONDS,  ///< duration of nanoseconds since Unix Epoch in int64
  CATEGORY,   ///< Categorial/Dictionary type
  STRING,     ///< String elements
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS  ///< Total number of type ids
};

/**---------------------------------------------------------------------------*
 * @brief Indicator for the logical data type of an element in a column.
 *
 * Simple types can be be entirely described by their `id()`, but some types
 * require additional metadata to fully describe elements of that type.
 *---------------------------------------------------------------------------**/
class data_type {
 public:
  data_type() = default;
  ~data_type() = default;
  data_type(data_type const&) = default;
  data_type(data_type&&) = default;
  data_type& operator=(data_type const&) = default;
  data_type& operator=(data_type&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new `data_type` object
   *
   * @param id The type's identifier
   *---------------------------------------------------------------------------**/
  explicit constexpr data_type(type_id id) : _id{id} {}

  /**---------------------------------------------------------------------------*
   * @brief Returns the type identifier
   *---------------------------------------------------------------------------**/
  CUDA_HOST_DEVICE_CALLABLE type_id id() const noexcept { return _id; }

 private:
  type_id _id{EMPTY};
  // Store additional type specific metadata, timezone, decimal precision and
  // scale, etc.
};

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
inline bool operator==(data_type const& lhs, data_type const& rhs) {
  return lhs.id() == rhs.id();
}

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

}  // namespace cudf
