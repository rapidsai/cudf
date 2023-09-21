/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief Computes sum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute sum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the sum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Sum as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> sum(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes minimum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is convertible to `output_dtype`
 *
 * @param col input column to compute minimum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the minimum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Minimum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> min(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes maximum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is convertible to `output_dtype`
 *
 * @param col input column to compute maximum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the maximum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Maximum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> max(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes any of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute any
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the any
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return bool scalar if any of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> any(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes all of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute all
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the all
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return bool scalar if all of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> all(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

/**
 * @brief Compute frequency for each unique element in the input column.
 *
 * The result histogram is stored in structs column having two children. The first child contains
 * unique elements from the input, and the second child contains their corresponding frequencies.
 *
 * @param input The column to compute histogram
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return A list_scalar storing a structs column as the result histogram
 */
std::unique_ptr<scalar> histogram(column_view const& input,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @brief Merge multiple histograms together.
 *
 * @param input The input given as multiple histograms concatenated together
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return A list_scalar storing the result histogram
 */
std::unique_ptr<scalar> merge_histogram(column_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes product of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute product
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the product
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Product as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> product(column_view const& col,
                                data_type const output_dtype,
                                std::optional<std::reference_wrapper<scalar const>> init,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes sum of squares of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute sum of squares
 * @param output_dtype data type of return type and typecast elements of input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Sum of squares as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> sum_of_squares(column_view const& col,
                                       data_type const output_dtype,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes mean of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute mean
 * @param output_dtype data type of return type and typecast elements of input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Mean as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> mean(column_view const& col,
                             data_type const output_dtype,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes variance of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute variance
 * @param output_dtype data type of return type and typecast elements of input column
 * @param ddof Delta degrees of freedom. The divisor used is N - ddof, where N represents the number
 * of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Variance as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> variance(column_view const& col,
                                 data_type const output_dtype,
                                 size_type ddof,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr);

/**
 * @brief Computes standard deviation of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute standard deviation
 * @param output_dtype data type of return type and typecast elements of input column
 * @param ddof Delta degrees of freedom. The divisor used is N - ddof, where N represents the number
 * of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Standard deviation as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> standard_deviation(column_view const& col,
                                           data_type const output_dtype,
                                           size_type ddof,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr);

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
 * @param col input column to get nth element from
 * @param n index of element to get
 * @param null_handling Indicates if null values will be counted while indexing
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return nth element as scalar
 */
std::unique_ptr<scalar> nth_element(column_view const& col,
                                    size_type n,
                                    null_policy null_handling,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Collect input column into a (list) scalar
 *
 * @param col input column to collect from
 * @param null_handling Indicates if null values will be counted while collecting
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list as scalar
 */
std::unique_ptr<scalar> collect_list(column_view const& col,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr);

/**
 * @brief Merge a bunch of list scalars into single list scalar
 *
 * @param col input list column representing numbers of list scalars to be merged
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return merged list as scalar
 */
std::unique_ptr<scalar> merge_lists(lists_column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Collect input column into a (list) scalar without duplicated elements
 *
 * @param col input column to collect from
 * @param null_handling Indicates if null values will be counted while collecting
 * @param nulls_equal Indicates if null values will be considered as equal values
 * @param nans_equal Indicates if nan values will be considered as equal values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list with unique elements as scalar
 */
std::unique_ptr<scalar> collect_set(column_view const& col,
                                    null_policy null_handling,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @brief Merge a bunch of list scalars into single list scalar then drop duplicated elements
 *
 * @param col input list column representing numbers of list scalars to be merged
 * @param nulls_equal Indicates if null values will be considered as equal values
 * @param nans_equal Indicates if nan values will be considered as equal values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list with unique elements as scalar
 */
std::unique_ptr<scalar> merge_sets(lists_column_view const& col,
                                   null_equality nulls_equal,
                                   nan_equality nans_equal,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
