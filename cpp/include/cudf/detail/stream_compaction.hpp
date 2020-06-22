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

#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>

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
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

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
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::apply_boolean_mask
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> apply_boolean_mask(
  table_view const& input,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::drop_duplicates
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> drop_duplicates(
  table_view const& input,
  std::vector<size_type> const& keys,
  duplicate_keep_option keep,
  null_equality nulls_equal           = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::distinct_count(column_view const&, null_policy, nan_policy)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               cudaStream_t stream = 0);

/**
 * @copydoc cudf::distinct_count(table_view const&, null_equality)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal = null_equality::EQUAL,
                               cudaStream_t stream       = 0);

}  // namespace detail
}  // namespace cudf
