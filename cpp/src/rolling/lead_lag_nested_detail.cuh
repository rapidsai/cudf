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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/scatter.cuh>
#include <cudf/utilities/traits.hpp>
#include <vector>

namespace cudf::detail {
namespace {
/**
 * @brief Functor to calculate the gather map used for calculating LEAD/LAG.
 *
 * @tparam op Aggregation Kind (LEAD vs LAG)
 * @tparam PrecedingIterator Iterator to retrieve preceding window bounds
 * @tparam FollowingIterator Iterator to retrieve following window bounds
 */
template <aggregation::Kind op, typename PrecedingIterator, typename FollowingIterator>
struct lead_lag_gather_map_builder {
  lead_lag_gather_map_builder(size_type input_size,
                              size_type row_offset,
                              PrecedingIterator preceding,
                              FollowingIterator following)
    : _input_size{input_size},
      NULL_INDEX{input_size + 1},
      _row_offset{row_offset},
      _preceding{preceding},
      _following{following}
  {
  }

  template <aggregation::Kind o = op, CUDF_ENABLE_IF(o == aggregation::LEAD)>
  size_type __device__ operator()(size_type i)
  {
    // Note: grouped_*rolling_window() trims preceding/following to
    // the beginning/end of the group. `rolling_window()` does not.
    // Must trim _following[i] so as not to go past the column end.
    auto following = min(_following[i], _input_size - i - 1);
    return (_row_offset > following) ? NULL_INDEX : (i + _row_offset);
  }

  template <aggregation::Kind o = op, CUDF_ENABLE_IF(o == aggregation::LAG)>
  size_type __device__ operator()(size_type i)
  {
    // Note: grouped_*rolling_window() trims preceding/following to
    // the beginning/end of the group. `rolling_window()` does not.
    // Must trim _preceding[i] so as not to go past the column start.
    auto preceding = min(_preceding[i], i + 1);
    return (_row_offset > (preceding - 1)) ? NULL_INDEX : (i - _row_offset);
  }

 private:
  size_type _input_size;
  size_type NULL_INDEX;
  size_type _row_offset;
  PrecedingIterator _preceding;
  FollowingIterator _following;
};

/**
 * @brief Predicate to find indices at which LEAD/LAG evaluated to null.
 */
template <typename GatherMapIter>
struct is_null_index_predicate_impl {
  is_null_index_predicate_impl(size_type input_size, GatherMapIter gather_)
    : NULL_INDEX{input_size + 1}, gather{gather_}
  {
  }

  bool __device__ operator()(size_type i) const { return gather[i] == NULL_INDEX; }

 private:
  const size_type NULL_INDEX;
  GatherMapIter gather;
};

/**
 * @brief Helper to construct is_null_index_predicate_impl
 */
template <typename GatherMapIter>
is_null_index_predicate_impl<GatherMapIter> is_null_index_predicate(size_type input_size,
                                                                    GatherMapIter gather)
{
  return is_null_index_predicate_impl<GatherMapIter>{input_size, gather};
}

}  // namespace

/**
 * @brief Helper function to calculate LEAD/LAG for nested-type input columns.
 *
 * @tparam op The sort of aggregation being done (LEAD vs LAG)
 * @tparam InputType The datatype of the input column being aggregated
 * @tparam PrecedingIterator Iterator-type that returns the preceding bounds
 * @tparam FollowingIterator Iterator-type that returns the following bounds
 * @param[in] input Nested-type input column for LEAD/LAG calculation
 * @param[in] default_outputs Default values to use as outputs, if LEAD/LAG
 *                            offset crosses column/group boundaries
 * @param[in] preceding Iterator to retrieve preceding window bounds
 * @param[in] following Iterator to retrieve following window bounds
 * @param[in] offset Lead/Lag offset, indicating which row after/before
 *                   the current row is to be returned
 * @param[in] stream CUDA stream for device memory operations/allocations
 * @param[in] mr device_memory_resource for device memory allocations
 */
template <aggregation::Kind op,
          typename InputType,
          typename PrecedingIter,
          typename FollowingIter,
          CUDF_ENABLE_IF(!cudf::is_fixed_width<InputType>())>
std::unique_ptr<column> compute_lead_lag_for_nested(column_view const& input,
                                                    column_view const& default_outputs,
                                                    PrecedingIter preceding,
                                                    FollowingIter following,
                                                    size_type offset,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(default_outputs.type().id() == input.type().id(),
               "Defaults column type must match input column.");  // Because LEAD/LAG.

  CUDF_EXPECTS(default_outputs.is_empty() || (input.size() == default_outputs.size()),
               "Number of defaults must match input column.");

  // For LEAD(0)/LAG(0), no computation need be performed.
  // Return copy of input.
  if (offset == 0) { return std::make_unique<column>(input, stream, mr); }

  // Algorithm:
  //
  // 1. Construct gather_map with the LEAD/LAG offset applied to the indices.
  //    E.g. A gather_map of:
  //        {0, 1, 2, 3, ..., N-3, N-2, N-1}
  //    would select the input column, unchanged.
  //
  //    For LEAD(2), the following gather_map is used:
  //        {3, 4, 5, 6, ..., N-1, NULL_INDEX, NULL_INDEX}
  //    where `NULL_INDEX` selects `NULL` for the gather.
  //
  //    Similarly, LAG(2) is implemented using the following gather_map:
  //        {NULL_INDEX, NULL_INDEX, 0, 1, 2...}
  //
  // 2. Gather input column based on the gather_map.
  // 3. If default outputs are available, scatter contents of default_outputs`
  //    to all positions where nulls where gathered in step 2.
  //
  // Note: Step 3 can be switched to use `copy_if_else()`, once it supports
  //       nested types.

  auto static constexpr size_data_type = data_type{type_to_id<size_type>()};

  auto gather_map_column =
    make_numeric_column(size_data_type, input.size(), mask_state::UNALLOCATED, stream);
  auto gather_map = gather_map_column->mutable_view();

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(size_type{0}),
                    thrust::make_counting_iterator(size_type{input.size()}),
                    gather_map.begin<size_type>(),
                    lead_lag_gather_map_builder<op, PrecedingIter, FollowingIter>{
                      input.size(), offset, preceding, following});

  auto output_with_nulls = cudf::gather(table_view{std::vector<column_view>{input}},
                                        gather_map_column->view(),
                                        out_of_bounds_policy::NULLIFY);

  if (default_outputs.is_empty()) { return std::move(output_with_nulls->release()[0]); }

  // Must scatter defaults.
  auto NULL_INDEX = size_type{input.size() + 1};

  auto scatter_map = rmm::device_uvector<size_type>(input.size(), stream);

  // Find all indices at which LEAD/LAG computed nulls previously.
  auto scatter_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(size_type{0}),
                    thrust::make_counting_iterator(size_type{input.size()}),
                    scatter_map.begin(),
                    is_null_index_predicate(input.size(), gather_map.begin<size_type>()));

  // Bail early, if all LEAD/LAG computations succeeded. No defaults need be substituted.
  if (scatter_map.is_empty()) { return std::move(output_with_nulls->release()[0]); }

  // Gather only those default values that are to be substituted.
  auto gathered_defaults =
    cudf::detail::gather(table_view{std::vector<column_view>{default_outputs}},
                         scatter_map.begin(),
                         scatter_map_end,
                         out_of_bounds_policy::DONT_CHECK,
                         stream);

  // Scatter defaults into locations where LEAD/LAG computed nulls.
  auto scattered_results = cudf::detail::scatter(
    table_view{std::vector<column_view>{gathered_defaults->release()[0]->view()}},
    scatter_map.begin(),
    scatter_map_end,
    table_view{std::vector<column_view>{output_with_nulls->release()[0]->view()}},
    false,
    stream);
  return std::move(scattered_results->release()[0]);
}

}  // namespace cudf::detail
