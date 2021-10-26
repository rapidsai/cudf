/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

namespace tdigest {

/**
 * @brief Create a tdigest column from it's constituent components.
 *
 * @param num_rows The number of rows in the output column.
 * @param centroid_means The inner means column.  These values are partitioned into lists by the
 * `tdigest_offsets` column.
 * @param centroid_weights The inner weights column.  These values are partitioned into lists by the
 * `tdigest_offsets` column.
 * @param tdigest_offsets Offsets representing each individual tdigest in the output column. The
 * offsets partition the centroid means and weights.
 * @param min_values Column representing the minimum input value for each tdigest.
 * @param max_values Column representing the maximum input value for each tdigest.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns The constructed tdigest column.
 */
std::unique_ptr<column> make_tdigest_column(
  size_type num_rows,
  std::unique_ptr<column>&& centroid_means,
  std::unique_ptr<column>&& centroid_weights,
  std::unique_ptr<column>&& tdigest_offsets,
  std::unique_ptr<column>&& min_values,
  std::unique_ptr<column>&& max_values,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create an empty tdigest column.
 *
 * An empty tdigest column contains a single row of length 0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns An empty tdigest column.
 */
std::unique_ptr<column> make_empty_tdigest_column(
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace tdigest
}  // namespace detail
}  // namespace cudf