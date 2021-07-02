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

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {

namespace {
/**
 * @brief Functor to accumulate (merge) all partial results corresponding to the same key into a
 * final result storing in its member variables. It performs merging for the partial results of
 * `COUNT_VALID`, `MEAN`, and `M2` at the same time.
 */
template <class ResultType>
struct accumulate_fn {
  size_type n_a;
  ResultType mean_a;
  ResultType M2_a;

  void __device__ operator()(size_type const n_b,
                             ResultType const mean_b,
                             ResultType const M2_b) noexcept
  {
    if (n_b == 0) { return; }

    auto const n_ab  = n_a + n_b;
    auto const delta = mean_b - mean_a;
    M2_a +=
      M2_b + (delta * delta) * static_cast<ResultType>(n_a) * static_cast<ResultType>(n_b) / n_ab;
    mean_a = (mean_a * n_a + mean_b * n_b) / n_ab;
    n_a    = n_ab;
  }
};

/**
 * @brief Functor to merge partial results of `COUNT_VALID`, `MEAN`, and `M2` aggregations
 * for a given group (key) index.
 */
template <class ResultType>
struct merge_fn {
  size_type const* const d_offsets;
  size_type const* const d_counts;
  ResultType const* const d_means;
  ResultType const* const d_M2s;

  auto __device__ operator()(size_type const group_idx) noexcept
  {
    auto const start_idx = d_offsets[group_idx], end_idx = d_offsets[group_idx + 1];

    // This case should never happen, because all groups are non-empty due to the given input.
    // Here just to make sure we cover this case.
    if (start_idx == end_idx) {
      return thrust::make_tuple(size_type{0}, ResultType{0}, ResultType{0}, int8_t{0});
    }

    // Firstly, this stores (valid_count, mean, M2) of the first partial result.
    // Then, it accumulates (merges) the remaining partial results into it.
    // Note that, if `n_a == 0` then `mean_a` and `M2_a` will be null.
    // Thus, in such situations, we need to set zero for them before accumulating partial results.
    auto const n_a    = d_counts[start_idx];
    auto const mean_a = n_a > 0 ? d_means[start_idx] : ResultType{0};
    auto const M2_a   = n_a > 0 ? d_M2s[start_idx] : ResultType{0};
    auto accumulator  = accumulate_fn<ResultType>{n_a, mean_a, M2_a};

    for (auto idx = start_idx + 1; idx < end_idx; ++idx) {
      // if `n_b > 0` then we must have `d_means[idx] != null` and `d_M2s[idx] != null`.
      // if `n_b == 0` then `mean_b` and `M2_b` will be null.
      // In such situations, we need to set zero for them before merging (all zero partial results
      // will not change the final output).
      auto const n_b    = d_counts[idx];
      auto const mean_b = n_b > 0 ? d_means[idx] : ResultType{0};
      auto const M2_b   = n_b > 0 ? d_M2s[idx] : ResultType{0};
      accumulator(n_b, mean_b, M2_b);
    }

    // If there are all nulls in the partial results (i.e., sum of valid counts is
    // zero), then the output is null.
    auto const is_valid = int8_t{accumulator.n_a > 0};

    return accumulator.n_a > 0
             ? thrust::make_tuple(accumulator.n_a, accumulator.mean_a, accumulator.M2_a, is_valid)
             : thrust::make_tuple(size_type{0}, ResultType{0}, ResultType{0}, is_valid);
  }
};

}  // namespace

std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       cudf::device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_merge_m2` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 3,
               "Input to `group_merge_m2` must be a structs column having 3 children columns.");

  using ResultType = id_to_type<type_id::FLOAT64>;
  static_assert(
    std::is_same_v<cudf::detail::target_type_t<ResultType, aggregation::Kind::M2>, ResultType>);
  CUDF_EXPECTS(values.child(0).type().id() == type_id::INT32 &&
                 values.child(1).type().id() == type_to_id<ResultType>() &&
                 values.child(2).type().id() == type_to_id<ResultType>(),
               "Input to `group_merge_m2` must be a structs column having children columns "
               "containing tuples of groupwise (M2_value, mean, valid_count).");

  auto result_counts = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_means = make_numeric_column(
    data_type(type_to_id<ResultType>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_M2s = make_numeric_column(
    data_type(type_to_id<ResultType>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto validities = rmm::device_uvector<int8_t>(num_groups, stream);

  // Perform merging for all the aggregations. Their output (and their validity data) are written
  // out concurrently through an output zip iterator.
  using IteratorTuple = thrust::tuple<size_type*, ResultType*, ResultType*, int8_t*>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;
  auto const out_iter =
    ZipIterator{thrust::make_tuple(result_counts->mutable_view().template data<size_type>(),
                                   result_means->mutable_view().template data<ResultType>(),
                                   result_M2s->mutable_view().template data<ResultType>(),
                                   validities.begin())};

  auto const count_valid = values.child(0);
  auto const mean_values = values.child(1);
  auto const M2_values   = values.child(2);
  auto const iter        = thrust::make_counting_iterator<size_type>(0);

  auto const fn = merge_fn<ResultType>{group_offsets.begin(),
                                       count_valid.template begin<size_type>(),
                                       mean_values.template begin<ResultType>(),
                                       M2_values.template begin<ResultType>()};
  thrust::transform(rmm::exec_policy(stream), iter, iter + num_groups, out_iter, fn);

  // Generate bitmask for the output.
  // Only mean and M2 values can be nullable.
  auto [null_mask, null_count] = cudf::detail::valid_if(
    validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);
  if (null_count > 0) {
    result_means->set_null_mask(null_mask, null_count);           // copy null_mask
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
