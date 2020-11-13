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

namespace cudf {
/**
 * @addtogroup scalar_factories
 * @{
 * @file
 * @brief Scalar factory APIs
 */

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified numeric `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a numeric type
 *
 * @param type The desired numeric element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
std::unique_ptr<scalar> make_numeric_scalar(
  data_type type,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified timestamp `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a timestamp type
 *
 * @param type The desired timestamp element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
std::unique_ptr<scalar> make_timestamp_scalar(
  data_type type,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified duration `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a duration type
 *
 * @param type The desired duration element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
std::unique_ptr<scalar> make_duration_scalar(
  data_type type,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Construct scalar with uninitialized storage to hold a value of the
 * specified fixed-width `data_type`.
 *
 * @throws std::bad_alloc if device memory allocation fails
 * @throws cudf::logic_error if `type` is not a fixed-width type
 *
 * @param type The desired fixed-width element type
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
std::unique_ptr<scalar> make_fixed_width_scalar(
  data_type type,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Construct STRING type scalar given a `std::string`.
 * The size of the `std::string` must not exceed the maximum size of size_type.
 * The string characters are expected to be UTF-8 encoded sequence of char bytes.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param string The `std::string` to copy to device
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
std::unique_ptr<scalar> make_string_scalar(
  std::string const& string,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Constructs default constructed scalar of type `type`
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param type The desired element type
 */
std::unique_ptr<scalar> make_default_constructed_scalar(data_type type);

/**
 * @brief Construct scalar using the given value of fixed width type
 *
 * @tparam T Datatype of the value to be represented by the scalar
 * @param value The value to store in the scalar object
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
template <typename T>
std::unique_ptr<scalar> make_fixed_width_scalar(
  T value,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return std::make_unique<scalar_type_t<T>>(value, true, stream, mr);
}

/**
 * @brief Construct scalar using the given value of fixed_point type
 *
 * @tparam T Datatype of the value to be represented by the scalar
 * @param value The value to store in the scalar object
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource used to allocate the scalar's `data` and `is_valid` bool.
 */
template <typename T>
std::unique_ptr<scalar> make_fixed_point_scalar(
  typename T::rep value,
  numeric::scale_type scale,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return std::make_unique<scalar_type_t<T>>(value, scale, true, stream, mr);
}

/** @} */  // end of group
}  // namespace cudf
