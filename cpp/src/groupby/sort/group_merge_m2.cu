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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
/**
 * @brief Struct to store partial results for merging.
 */
template <class result_type>
struct partial_result {
  size_type count;
  result_type mean;
  result_type M2;
};

/**
 * @brief Functor to accumulate (merge) all partial results corresponding to the same key into a
 * final result storing in a member variable. It performs merging for the partial results of
 * `COUNT_VALID`, `MEAN`, and `M2` at the same time.
 */
template <class result_type>
struct accumulate_fn {
  partial_result<result_type> merge_vals;

  void __device__ operator()(partial_result<result_type> const& partial_vals) noexcept
  {
    if (partial_vals.count == 0) { return; }

    auto const n_ab  = merge_vals.count + partial_vals.count;
    auto const delta = partial_vals.mean - merge_vals.mean;
    merge_vals.M2 += partial_vals.M2 + (delta * delta) *
                                         static_cast<result_type>(merge_vals.count) *
                                         static_cast<result_type>(partial_vals.count) / n_ab;
    merge_vals.mean =
      (merge_vals.mean * merge_vals.count + partial_vals.mean * partial_vals.count) / n_ab;
    merge_vals.count = n_ab;
  }
};

/**
 * @brief Functor to merge partial results of `COUNT_VALID`, `MEAN`, and `M2` aggregations
 * for a given group (key) index.
 */
template <class result_type>
struct merge_fn {
  size_type const* const d_offsets;
  size_type const* const d_counts;
  result_type const* const d_means;
  result_type const* const d_M2s;

  auto __device__ operator()(size_type const group_idx) noexcept
  {
    auto const start_idx = d_offsets[group_idx], end_idx = d_offsets[group_idx + 1];

    // This case should never happen, because all groups are non-empty as the results of
    // aggregation. Here we just to make sure we cover this case.
    if (start_idx == end_idx) {
      return thrust::make_tuple(size_type{0}, result_type{0}, result_type{0}, int8_t{0});
    }

    // If `(n = d_counts[idx]) > 0` then `d_means[idx] != null` and `d_M2s[idx] != null`.
    // Otherwise (`n == 0`), these value (mean and M2) will always be nulls.
    // In such cases, reading `mean` and `M2` from memory will return garbage values.
    // By setting these values to zero when `n == 0`, we can safely merge the all-zero tuple without
    // affecting the final result.
    auto get_partial_result = [&] __device__(size_type idx) {
      {
        auto const n = d_counts[idx];
        return n > 0 ? partial_result<result_type>{n, d_means[idx], d_M2s[idx]}
                     : partial_result<result_type>{size_type{0}, result_type{0}, result_type{0}};
      };
    };

    // Firstly, store tuple(count, mean, M2) of the first partial result in an accumulator.
    auto accumulator = accumulate_fn<result_type>{get_partial_result(start_idx)};

    // Then, accumulate (merge) the remaining partial results into that accumulator.
    for (auto idx = start_idx + 1; idx < end_idx; ++idx) {
      accumulator(get_partial_result(idx));
    }

    // Get the final result after merging.
    auto const& merge_vals = accumulator.merge_vals;

    // If there are all nulls in the partial results (i.e., sum of all valid counts is
    // zero), then the output is a null.
    auto const is_valid = int8_t{merge_vals.count > 0};

    return thrust::make_tuple(merge_vals.count, merge_vals.mean, merge_vals.M2, is_valid);
  }
};

}  // namespace

std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       cudf::device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_merge_m2` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 3,
               "Input to `group_merge_m2` must be a structs column having 3 children columns.");

  using result_type = id_to_type<type_id::FLOAT64>;
  static_assert(
    std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::M2>, result_type>);
  CUDF_EXPECTS(values.child(0).type().id() == type_id::INT32 &&
                 values.child(1).type().id() == type_to_id<result_type>() &&
                 values.child(2).type().id() == type_to_id<result_type>(),
               "Input to `group_merge_m2` must be a structs column having children columns "
               "containing tuples of (M2_value, mean, valid_count).");

  auto result_counts = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_means = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_M2s = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto validities = rmm::device_uvector<int8_t>(num_groups, stream);

  // Perform merging for all the aggregations. Their output (and their validity data) are written
  // out concurrently through an output zip iterator.
  using iterator_tuple  = thrust::tuple<size_type*, result_type*, result_type*, int8_t*>;
  using output_iterator = thrust::zip_iterator<iterator_tuple>;
  auto const out_iter =
    output_iterator{thrust::make_tuple(result_counts->mutable_view().template data<size_type>(),
                                       result_means->mutable_view().template data<result_type>(),
                                       result_M2s->mutable_view().template data<result_type>(),
                                       validities.begin())};

  auto const count_valid = values.child(0);
  auto const mean_values = values.child(1);
  auto const M2_values   = values.child(2);
  auto const iter        = thrust::make_counting_iterator<size_type>(0);

  auto const fn = merge_fn<result_type>{group_offsets.begin(),
                                        count_valid.template begin<size_type>(),
                                        mean_values.template begin<result_type>(),
                                        M2_values.template begin<result_type>()};
  thrust::transform(rmm::exec_policy(stream), iter, iter + num_groups, out_iter, fn);

  // Generate bitmask for the output.
  // Only mean and M2 values can be nullable. Count column must be non-nullable.
  auto [null_mask, null_count] =
    cudf::detail::valid_if(validities.begin(), validities.end(), thrust::identity{}, stream, mr);
  if (null_count > 0) {
    result_means->set_null_mask(null_mask, null_count, stream);   // copy null_mask
    result_M2s->set_null_mask(std::move(null_mask), null_count);  // take over null_mask
  }

  // Output is a structs column containing the merged values of `COUNT_VALID`, `MEAN`, and `M2`.
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.emplace_back(std::move(result_counts));
  out_columns.emplace_back(std::move(result_means));
  out_columns.emplace_back(std::move(result_M2s));
  auto result = cudf::make_structs_column(
    num_groups, std::move(out_columns), 0, rmm::device_buffer{0, stream, mr}, stream, mr);

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
