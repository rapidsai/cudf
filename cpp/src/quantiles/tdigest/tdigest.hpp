/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace cudf {
namespace detail {
namespace tdigest {

/**
 * @brief Generate a tdigest column from a grouped set of numeric input values.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 *
 * @param values Grouped (and sorted) values to merge.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param group_labels 0-based ID of group that the corresponding value belongs to
 * @param group_valid_counts Per-group counts of valid elements.
 * @param num_groups Number of groups.
 * @param max_centroids Parameter controlling the level of compression of the tdigest. Higher
 * values result in a larger, more precise tdigest.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns tdigest column, with 1 tdigest per row
 */
std::unique_ptr<column> group_tdigest(column_view const& values,
                                      cudf::device_span<size_type const> group_offsets,
                                      cudf::device_span<size_type const> group_labels,
                                      cudf::device_span<size_type const> group_valid_counts,
                                      size_type num_groups,
                                      int max_centroids,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Merges tdigests within the same group to generate a new tdigest.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 *
 * @param values Grouped tdigests to merge.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param group_labels 0-based ID of group that the corresponding value belongs to
 * @param num_groups Number of groups.
 * @param max_centroids Parameter controlling the level of compression of the tdigest. Higher
 * values result in a larger, more precise tdigest.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns tdigest column, with 1 tdigest per row
 */
std::unique_ptr<column> group_merge_tdigest(column_view const& values,
                                            cudf::device_span<size_type const> group_offsets,
                                            cudf::device_span<size_type const> group_labels,
                                            size_type num_groups,
                                            int max_centroids,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr);

}  // namespace tdigest
}  // namespace detail
}  // namespace cudf
