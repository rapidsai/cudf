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
template <class ResultType>
struct accumulate_fn {
  ResultType M2_a;
  ResultType mean_a;
  size_type n_a;

  void __device__ operator()(ResultType const M2_b,
                             ResultType const mean_b,
                             size_type const n_b) noexcept
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
  CUDF_EXPECTS(values.child(0).type().id() == type_to_id<ResultType>() &&
                 values.child(1).type().id() == type_to_id<ResultType>() &&
                 values.child(2).type().id() == type_id::INT32,
               "Input to `group_merge_m2` must be a structs column having children columns "
               "containing tuples of groupwise (M2_value, mean, valid_count).");

  auto result = make_numeric_column(
    data_type(type_to_id<ResultType>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  auto const M2_values   = values.child(0);
  auto const mean_values = values.child(1);
  auto const count_valid = values.child(2);
  auto const iter        = thrust::make_counting_iterator<size_type>(0);
  auto validities        = rmm::device_uvector<int8_t>(num_groups, stream);

  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_groups,
                    result->mutable_view().data<ResultType>(),
                    [d_M2      = M2_values.template begin<ResultType>(),
                     d_mean    = mean_values.template begin<ResultType>(),
                     d_count   = count_valid.template begin<size_type>(),
                     d_offsets = group_offsets.begin(),
                     d_valid   = validities.begin()] __device__(auto const group_idx) {
                      auto const start_idx = d_offsets[group_idx],
                                 end_idx   = d_offsets[group_idx + 1];

                      // Firstly, this stores (M2, mean, valid_count) of the first partial result.
                      // Then, merge all the following partial results into it.
                      auto accumulator = accumulate_fn<ResultType>{
                        d_M2[start_idx], d_mean[start_idx], d_count[start_idx]};

                      for (auto idx = start_idx + 1; idx < end_idx; ++idx) {
                        auto const n_b    = d_count[idx];
                        auto const M2_b   = n_b > 0 ? d_M2[idx] : ResultType{0};
                        auto const mean_b = n_b > 0 ? d_mean[idx] : ResultType{0};
                        accumulator(M2_b, mean_b, n_b);
                      }

                      // If there are all nulls in the partial results (i.e., sum of valid counts is
                      // zero), then output a null.
                      d_valid[group_idx] = accumulator.n_a > 0;
                      return accumulator.n_a > 0 ? accumulator.M2_a : ResultType{0};
                    });

  auto [null_mask, null_count] = cudf::detail::valid_if(
    validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr);
  if (null_count > 0) { result->set_null_mask(null_mask, null_count); }

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
