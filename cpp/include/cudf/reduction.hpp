/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>

namespace cudf {
/**
 * @addtogroup aggregation_reduction
 * @{
 * @file
 */

/**
 *  @brief Enum to describe scan operation type
 */
enum class scan_type : bool { INCLUSIVE, EXCLUSIVE };

/**
 * @brief  Computes the reduction of the values in all rows of a column.
 *
 * This function does not detect overflows in reductions. When `output_dtype`
 * does not match the `col.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to
 * `output_dtype` before returning.
 *
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (e.g. timestamp or string).
 *
 * Any null values are skipped for the operation.
 *
 * If the column is empty or contains all null entries `col.size()==col.null_count()`,
 * the output scalar value will be `false` for reduction type `any` and `true`
 * for reduction type `all`. For all other reductions, the output scalar
 * returns with `is_valid()==false`.
 *
 * If the input column is an arithmetic type, the `output_dtype` can be any arithmetic
 * type. If the input column is a non-arithmetic type (e.g. timestamp or string)
 * the `output_dtype` must match the `col.type()`. If the reduction type is `any` or
 * `all`, the `output_dtype` must be type BOOL8.
 *
 * If the reduction fails, the output scalar returns with `is_valid()==false`.
 *
 * @throw cudf::logic_error if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw cudf::logic_error if input column data type is not convertible to
 * `output_dtype`.
 * @throw cudf::logic_error if `min` or `max` reduction is called and the
 * output type does not match the input column data type.
 * @throw cudf::logic_error if `any` or `all` reduction is called and the
 * output type is not BOOL8.
 * @throw cudf::logic_error if `mean`, `var`, or `std` reduction is called and
 * the `output_dtype` is not floating point.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_dtype The output scalar type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_dtype,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Computes the reduction of the values in all rows of a column with an initial value
 *
 * Only `sum`, `product`, `min`, `max`, `any`, and `all` reductions are supported.
 *
 * @throw cudf::logic_error if reduction is not `sum`, `product`, `min`, `max`, `any`, or `all`
 * and `init` is specified.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_dtype The output scalar type
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_dtype,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Compute reduction of each segment in the input column
 *
 * This function does not detect overflows in reductions. When `output_dtype`
 * does not match the `segmented_values.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to
 * `output_dtype` before returning.
 *
 * Null values are treated as identities during reduction.
 *
 * If the segment is empty, the row corresponding to the result of the
 * segment is null.
 *
 * If any index in `offsets` is out of bound of `segmented_values`, the behavior
 * is undefined.
 *
 * If the input column has arithmetic type, `output_dtype` can be any arithmetic
 * type. If the input column has non-arithmetic type, e.g. timestamp, the same
 * output type must be specified.
 *
 * If input is not empty, the result is always nullable.
 *
 * @throw cudf::logic_error if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw cudf::logic_error if input column data type is not convertible to
 * `output_dtype` type.
 * @throw cudf::logic_error if `min` or `max` reduction is called and the
 * `output_dtype` does not match the input column data type.
 * @throw cudf::logic_error if `any` or `all` reduction is called and the
 * `output_dtype` is not BOOL8.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_dtype  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_dtype,
  null_policy null_handling,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Compute reduction of each segment in the input column with an initial value. Only SUM,
 * PRODUCT, MIN, MAX, ANY, and ALL aggregations are supported.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_dtype  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction.
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_dtype,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Computes the scan of a column.
 *
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @throws cudf::logic_error if column datatype is not numeric type.
 *
 * @param[in] input The input column view for the scan
 * @param[in] agg unique_ptr to aggregation operator applied by the scan
 * @param[in] inclusive The flag for applying an inclusive scan if scan_type::INCLUSIVE, an
 * exclusive scan if scan_type::EXCLUSIVE.
 * @param[in] null_handling Exclude null values when computing the result if null_policy::EXCLUDE.
 * Include nulls if null_policy::INCLUDE. Any operation with a null results in a null.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Scanned output column
 */
std::unique_ptr<column> scan(
  column_view const& input,
  scan_aggregation const& agg,
  scan_type inclusive,
  null_policy null_handling         = null_policy::EXCLUDE,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Determines the minimum and maximum values of a column.
 *
 *
 * @param col column to compute minmax
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A std::pair of scalars with the first scalar being the minimum value and the second
 * scalar being the maximum value of the input column.
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  column_view const& col,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

}  // namespace cudf
