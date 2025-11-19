/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator_tests_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator)
{
  rmm::cuda_stream_view stream{cudf::get_default_stream()};

  auto const table_comparator = cudf::detail::row::equality::two_table_comparator{lhs, rhs, stream};

  auto const lhs_it = cudf::detail::row::lhs_iterator(0);
  auto const rhs_it = cudf::detail::row::rhs_iterator(0);

  auto output = cudf::make_numeric_column(
    cudf::data_type(cudf::type_id::BOOL8), lhs.num_rows(), cudf::mask_state::UNALLOCATED);

  if (cudf::has_nested_columns(lhs) or cudf::has_nested_columns(rhs)) {
    auto const equal_comparator =
      table_comparator.equal_to<true>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  } else {
    auto const equal_comparator =
      table_comparator.equal_to<false>(cudf::nullate::NO{}, cudf::null_equality::EQUAL, comparator);

    thrust::transform(rmm::exec_policy(stream),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      equal_comparator);
  }
  return output;
}

template std::unique_ptr<cudf::column> two_table_equality<physical_equality_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  physical_equality_t comparator);
template std::unique_ptr<cudf::column> two_table_equality<nan_equality_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  nan_equality_t comparator);
