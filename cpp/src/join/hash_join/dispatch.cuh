/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/structs/utilities.hpp>

#include <utility>

namespace cudf::detail {

template <typename Fn>
decltype(auto) dispatch_join_comparator(
  table_view const& build_table,
  table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  bool has_nulls,
  null_equality compare_nulls,
  Fn&& fn)
{
  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(build_table)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{probe_nulls, preprocessed_probe};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      probe_nulls, preprocessed_probe, preprocessed_build, compare_nulls};
    return std::forward<Fn>(fn)(primitive_pair_equal{d_equal}, d_hasher);
  }

  auto const d_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(probe_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};

  if (cudf::detail::has_nested_columns(probe_table)) {
    auto const d_equal = row_comparator.equal_to<true>(probe_nulls, compare_nulls);
    return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
  }

  auto const d_equal = row_comparator.equal_to<false>(probe_nulls, compare_nulls);
  return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
}

}  // namespace cudf::detail
