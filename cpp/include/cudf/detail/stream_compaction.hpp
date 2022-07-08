/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::drop_nulls(table_view const&, std::vector<size_type> const&,
 *                           cudf::size_type, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> drop_nulls(
  table_view const& input,
  std::vector<size_type> const& keys,
  cudf::size_type keep_threshold,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::drop_nans(table_view const&, std::vector<size_type> const&,
 *                          cudf::size_type, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> drop_nans(
  table_view const& input,
  std::vector<size_type> const& keys,
  cudf::size_type keep_threshold,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::apply_boolean_mask
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> apply_boolean_mask(
  table_view const& input,
  column_view const& boolean_mask,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::unique
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> unique(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep,
  null_equality nulls_equal           = null_equality::EQUAL,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::distinct
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> distinct(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep          = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a new table without duplicate rows.
 *
 * Given an `input` table_view, each row is copied to the output table to create a set of distinct
 * rows. The row order is guaranteed to be preserved as in the input.
 *
 * If there are duplicate rows, which row to be copied depends on the specified value of the `keep`
 * parameter.
 *
 * This API produces exactly the same set of output rows as `cudf::distinct`.
 *
 * @param input The input table
 * @param keys Vector of indices indicating key columns in the `input` table
 * @param keep Copy any, first, last, or none of the found duplicates
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return A table containing the resulting distinct rows
 */
std::unique_ptr<table> stable_distinct(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep          = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a column of indices of all distinct rows in the input table.
 *
 * Given an `input` table_view, an output vector of all row indices of the distinct rows is
 * generated. If there are duplicate rows, which index is kept depends on the `keep` parameter.
 *
 * @param input The input table
 * @param keep Get index of any, first, last, or none of the found duplicates
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device_uvector containing the result indices
 */
rmm::device_uvector<size_type> get_distinct_indices(
  table_view const& input,
  duplicate_keep_option keep          = duplicate_keep_option::KEEP_ANY,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::unique_count(column_view const&, null_policy, nan_policy)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream = cudf::default_stream_value);

/**
 * @copydoc cudf::unique_count(table_view const&, null_equality)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type unique_count(table_view const& input,
                             null_equality nulls_equal    = null_equality::EQUAL,
                             rmm::cuda_stream_view stream = cudf::default_stream_value);

/**
 * @copydoc cudf::distinct_count(column_view const&, null_policy, nan_policy)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream = cudf::default_stream_value);

/**
 * @copydoc cudf::distinct_count(table_view const&, null_equality)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal    = null_equality::EQUAL,
                               rmm::cuda_stream_view stream = cudf::default_stream_value);

}  // namespace detail
}  // namespace cudf
