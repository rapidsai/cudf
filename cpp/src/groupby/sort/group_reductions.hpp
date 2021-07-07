/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

/** @internal @file Internal API in this file are mostly segmented reduction operations on column,
 * which are used in sort-based groupby aggregations.
 *
 */
namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Internal API to calculate groupwise sum
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_sum    = [7, -3, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get sum of
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_sum(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise product
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups    = 4
 *
 * group_product = [6, 2, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get product of
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_product(column_view const& values,
                                      size_type num_groups,
                                      cudf::device_span<size_type const> group_labels,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise minimum value
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_min    = [1, -2, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get minimum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_min(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise maximum value
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [4, -1, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get maximum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_max(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate group-wise indices of maximum values.
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [2, 0, 0, <NA>]
 * @endcode
 *
 * @param values Grouped values to get maximum value's index from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param key_sort_order Indices indicating sort order of groupby keys
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_argmax(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate group-wise indices of minimum values.
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [1, 1, 0, <NA>]
 * @endcode
 *
 * @param values Grouped values to get minimum value's index from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param key_sort_order Indices indicating sort order of groupby keys
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate number of non-null values in each group of
 *  @p values
 *
 * @code{.pseudo}
 * values            = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels      = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups        = 4
 *
 * group_count_valid = [3, 2, 1, 0]
 * @endcode
 *
 * @param values Grouped values to get valid count of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_count_valid(column_view const& values,
                                          cudf::device_span<size_type const> group_labels,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate number of values in each group of @p values
 *
 * @code{.pseudo}
 * group_offsets = [0, 3, 5, 7, 8]
 * num_groups    = 4
 *
 * group_count_all = [3, 2, 2, 1]
 * @endcode
 *
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_count_all(cudf::device_span<size_type const> group_offsets,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate sum of squares of differences from means.
 *
 * If there are only nulls in the group, the output value of that group will be null.
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_means   = [2.333333, -1.5, 4.0, <NA>]
 * group_m2(...) = [4.666666,  1.0, 0.0, <NA>]
 * @endcode
 *
 * @param values Grouped values to compute M2 values
 * @param group_means Pre-computed groupwise MEAN
 * @param group_labels ID of group corresponding value in @p values belongs to
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise variance
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_means  = [2.333333, -1.5, 4.0, <NA>]
 * group_sizes  = [3, 2, 2, 1]
 * ddof         = 1
 *
 * group_var    = [2.333333, 0.5, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped values to get variance of
 * @param group_means Pre-calculated groupwise MEAN
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group corresponding value in @p values belongs to
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `var` is `N - ddof`, where `N` is the group size.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_var(column_view const& values,
                                  column_view const& group_means,
                                  column_view const& group_sizes,
                                  cudf::device_span<size_type const> group_labels,
                                  size_type ddof,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate groupwise quantiles
 *
 * @code{.pseudo}
 * values       = [1, 2, 4, -2, -1, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_sizes  = [3, 2, 2, 1]
 * num_groups   = 4
 * quantiles    = [0.25, 0.5]
 *
 * group_quantiles = [1.5, 2, -1.75, -1.5,  4,  4, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_sizes Number of valid elements per group
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param quantiles List of quantiles q where q lies in [0,1]
 * @param interp Method to use when desired value lies between data points
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_quantiles(column_view const& values,
                                        column_view const& group_sizes,
                                        cudf::device_span<size_type const> group_offsets,
                                        size_type const num_groups,
                                        std::vector<double> const& quantiles,
                                        interpolation interp,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate number of unique values in each group of
 *  @p values
 *
 * @code{.pseudo}
 * values        = [2, 4, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_nunique(null_policy::EXCLUDE) = [2, 2, 1, 0]
 * group_nunique(null_policy::INCLUDE) = [2, 2, 2, 1]
 * @endcode
 *
 * @param values Grouped and sorted (within group) values to get unique count of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *  Include nulls if null_policy::INCLUDE.
 *  Nulls are treated equal.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_nunique(column_view const& values,
                                      cudf::device_span<size_type const> group_labels,
                                      size_type const num_groups,
                                      cudf::device_span<size_type const> group_offsets,
                                      null_policy null_handling,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to calculate nth values in each group of  @p values
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_sizes   = [3,        2,        2,       1]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_nth_element(n=0, null_policy::EXCLUDE) = [2, -1, 4, <NA>]
 * group_nth_element(n=0, null_policy::INCLUDE) = [2, -1, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped values to get nth value of
 * @param group_sizes Number of elements per group
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param n nth element to choose from each group of @p values
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *  Include nulls if null_policy::INCLUDE.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_nth_element(column_view const& values,
                                          column_view const& group_sizes,
                                          cudf::device_span<size_type const> group_labels,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          size_type n,
                                          null_policy null_handling,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr);
/**
 * @brief Internal API to collect grouped values into a lists column
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_collect(...) = [[2, 1, 4], [-1, -2], [<NA>, 4], [<NA>]]
 * @endcode
 *
 * @param values Grouped values to collect.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *        include nulls if null_policy::INCLUDE.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> group_collect(column_view const& values,
                                      cudf::device_span<size_type const> group_offsets,
                                      size_type num_groups,
                                      null_policy null_handling,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to merge grouped lists into one list.
 *
 * @code{.pseudo}
 * values        = [[2, 1], [], [4, -1, -2], [], [<NA>, 4, <NA>]]
 * group_offsets = [0,                        3,                  5]
 * num_groups    = 2
 *
 * group_merge_lists(...) = [[2, 1, 4, -1, -2], [<NA>, 4, <NA>]]
 * @endcode
 *
 * @param values Grouped values (lists column) to collect.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> group_merge_lists(column_view const& values,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr);

/**
 * @brief Internal API to merge grouped M2 values corresponding to the same key.
 *
 * The values of M2 are merged following the parallel algorithm described here:
 * `https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm`
 *
 * Merging M2 values require accessing to partial M2 values, means, and valid counts. Thus, the
 * input to this aggregation need to be a structs column containing tuples of 3 values
 * `(valid_count, mean, M2)`.
 *
 * This aggregation not only merges the partial results of `M2` but also merged all the partial
 * results of input aggregations (`COUNT_VALID`, `MEAN`, and `M2`). As such, the output will be a
 * structs column containing children columns of merged `COUNT_VALID`, `MEAN`, and `M2` values.
 *
 * @param values Grouped values (tuples of values `(valid_count, mean, M2)`) to merge.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       cudf::device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr);

/** @endinternal
 *
 */
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
