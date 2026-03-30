/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hash_join_helpers.cuh"

#include <cudf/utilities/error.hpp>

namespace cudf::detail {

std::size_t compute_join_output_size(
  table_view const& build_table,
  table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  hash_table_t const& hash_table,
  join_kind join,
  bool has_nulls,
  cudf::null_equality nulls_equal,
  rmm::cuda_stream_view stream)
{
  size_type const build_table_num_rows{build_table.num_rows()};
  size_type const probe_table_num_rows{probe_table.num_rows()};

  if (0 == build_table_num_rows) {
    switch (join) {
      case join_kind::INNER_JOIN: return 0;
      case join_kind::LEFT_JOIN: return probe_table_num_rows;
      default: CUDF_FAIL("Unsupported join type");
    }
  }

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};

  auto compute_size = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (join == join_kind::LEFT_JOIN) {
      return hash_table.count_outer(
        iter, iter + probe_table_num_rows, equality, hash_table.hash_function(), stream.value());
    } else {
      return hash_table.count(
        iter, iter + probe_table_num_rows, equality, hash_table.hash_function(), stream.value());
    }
  };

  if (cudf::detail::is_primitive_row_op_compatible(build_table)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{probe_nulls, preprocessed_probe};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      probe_nulls, preprocessed_probe, preprocessed_build, nulls_equal};

    return compute_size(primitive_pair_equal{d_equal}, d_hasher);
  } else {
    auto const d_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(probe_nulls);
    auto const row_comparator =
      cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};

    if (cudf::detail::has_nested_columns(probe_table)) {
      auto const d_equal = row_comparator.equal_to<true>(has_nulls, nulls_equal);
      return compute_size(pair_equal{d_equal}, d_hasher);
    } else {
      auto const d_equal = row_comparator.equal_to<false>(has_nulls, nulls_equal);
      return compute_size(pair_equal{d_equal}, d_hasher);
    }
  }
}

}  // namespace cudf::detail
