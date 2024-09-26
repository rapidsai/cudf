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

#include <cudf/ast/expressions.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::transform
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> transform(column_view const& input,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::compute_column
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::nans_to_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::bools_to_mask
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::one_hot_encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::mask_to_bools
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> mask_to_bools(bitmask_type const* null_mask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> row_bit_count(table_view const& t,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_row_bit_count(table_view const& t,
                                                size_type segment_length,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
