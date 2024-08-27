/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace tdigest::detail {

/**
 * @brief Generate a tdigest column from a grouped, sorted set of numeric input values.
 *
 * The input is expected to be sorted in ascending order within each group, with
 * nulls at the end.
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
                                      rmm::device_async_resource_ref mr);

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
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Create a tdigest column from its constituent components.
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
std::unique_ptr<column> make_tdigest_column(size_type num_rows,
                                            std::unique_ptr<column>&& centroid_means,
                                            std::unique_ptr<column>&& centroid_weights,
                                            std::unique_ptr<column>&& tdigest_offsets,
                                            std::unique_ptr<column>&& min_values,
                                            std::unique_ptr<column>&& max_values,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty tdigest column.
 *
 * An empty tdigest column contains rows of length 0.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns An empty tdigest column.
 */
CUDF_EXPORT
std::unique_ptr<column> make_tdigest_column_of_empty_clusters(size_type num_rows,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty tdigest scalar.
 *
 * An empty tdigest scalar is a struct_scalar that contains a single row of length 0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns An empty tdigest scalar.
 */
std::unique_ptr<scalar> make_empty_tdigest_scalar(rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

/**
 * @brief Generate a tdigest scalar from a set of numeric input values.
 *
 * The tdigest scalar produced is of the following structure:
 ** struct {
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
 *
 * @param values Values to merge.
 * @param max_centroids Parameter controlling the level of compression of the tdigest. Higher
 * values result in a larger, more precise tdigest.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 *
 * @returns tdigest scalar
 */
std::unique_ptr<scalar> reduce_tdigest(column_view const& values,
                                       int max_centroids,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Merges multiple tdigest columns to generate a new tdigest scalar.
 *
 * The tdigest scalar produced is of the following structure:
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
 * @param values tdigests to merge.
 * @param max_centroids Parameter controlling the level of compression of the tdigest. Higher
 * values result in a larger, more precise tdigest.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 *
 * @returns tdigest column, with 1 tdigest per row
 */
std::unique_ptr<scalar> reduce_merge_tdigest(column_view const& input,
                                             int max_centroids,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace tdigest::detail
}  // namespace CUDF_EXPORT cudf
