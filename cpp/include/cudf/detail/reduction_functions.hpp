/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace reduction {
/**
 * @brief Computes sum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not arithmetic point type
 *
 * @param col input column to compute sum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Sum as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> sum(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);
/**
 * @brief Computes minimum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is convertible to `output_dtype`
 *
 * @param col input column to compute minimum.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Minimum element as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> min(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);
/**
 * @brief Computes maximum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is convertible to `output_dtype`
 *
 * @param col input column to compute maximum.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Maximum element as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> max(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);
/**
 * @brief Computes any of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute any_of.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return bool scalar if any of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> any(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);
/**
 * @brief Computes all of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute all_of.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return bool scalar if all of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> all(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);
/**
 * @brief Computes product of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not arithmetic point type
 *
 * @param col input column to compute product.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Product as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> product(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Computes sum of squares of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not arithmetic point type
 *
 * @param col input column to compute sum of squares.
 * @param output_dtype data type of return type and typecast elements of input column
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Sum of squares as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> sum_of_squares(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Computes mean of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute mean.
 * @param output_dtype data type of return type and typecast elements of input column.
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Mean as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> mean(
  column_view const& col,
  data_type const output_dtype,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Computes variance of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute variance.
 * @param output_dtype data type of return type and typecast elements of input column.
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Variance as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> variance(
  column_view const& col,
  data_type const output_dtype,
  cudf::size_type ddof,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Computes standard deviation of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute standard deviation.
 * @param output_dtype data type of return type and typecast elements of input column.
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Standard deviation as scalar of type `output_dtype`.
 */
std::unique_ptr<scalar> standard_deviation(
  column_view const& col,
  data_type const output_dtype,
  cudf::size_type ddof,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Returns nth element in input column
 *
 * A negative value `n` is interpreted as `n+count`, where `count` is the number of valid
 * elements in the input column if `null_handling` is `null_policy::EXCLUDE`, else `col.size()`.
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @warning This function is expensive (invokes a kernel launch). So, it is not
 * recommended to be used in performance sensitive code or inside a loop.
 * It takes O(`col.size()`) time and space complexity for nullable column with
 * `null_policy::EXCLUDE` as input.
 *
 * @throw cudf::logic_error if n falls outside the range `[-count, count)` where `count` is the
 * number of valid * elements in the input column if `null_handling` is `null_policy::EXCLUDE`,
 * else `col.size()`.
 *
 * @param col input column to get nth element from.
 * @param n index of element to get
 * @param null_handling Indicates if null values will be counted while indexing.
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return nth element as scalar
 */
std::unique_ptr<scalar> nth_element(
  column_view const& col,
  size_type n,
  null_policy null_handling,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

}  // namespace reduction
}  // namespace cudf
