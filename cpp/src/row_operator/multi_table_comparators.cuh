/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <span>

namespace cudf::detail::row::equality {
namespace comparator_detail {

template <typename Equality, typename Factory>
auto make_device_comparators(std::span<std::shared_ptr<preprocessed_table> const> preprocessed_rhs,
                             Factory factory,
                             rmm::cuda_stream_view stream)
{
  auto host_comparators =
    cudf::detail::make_empty_pinned_vector<Equality>(preprocessed_rhs.size(), stream);
  std::transform(preprocessed_rhs.begin(),
                 preprocessed_rhs.end(),
                 std::back_inserter(host_comparators),
                 factory);
  return cudf::detail::make_device_uvector_async(
    host_comparators, stream, cudf::get_current_device_resource_ref());
}

}  // namespace comparator_detail

template <bool has_nested>
auto make_device_row_comparators(
  std::shared_ptr<preprocessed_table> const& preprocessed_lhs,
  std::span<std::shared_ptr<preprocessed_table> const> preprocessed_rhs,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream)
{
  using equality_type =
    device_row_comparator<has_nested, nullate::DYNAMIC, nan_equal_physical_equality_comparator>;

  return comparator_detail::make_device_comparators<equality_type>(
    preprocessed_rhs,
    [&](auto const& rhs) {
      auto const comparator = two_table_comparator{preprocessed_lhs, rhs};
      return comparator.equal_to<has_nested>(has_nulls, compare_nulls).comparator;
    },
    stream);
}

inline auto make_device_primitive_row_comparators(
  std::shared_ptr<preprocessed_table> const& preprocessed_lhs,
  std::span<std::shared_ptr<preprocessed_table> const> preprocessed_rhs,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream)
{
  using equality_type = cudf::detail::row::primitive::row_equality_comparator;

  return comparator_detail::make_device_comparators<equality_type>(
    preprocessed_rhs,
    [&](auto const& rhs) { return equality_type{has_nulls, preprocessed_lhs, rhs, compare_nulls}; },
    stream);
}

}  // namespace cudf::detail::row::equality
