/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "dispatch.cuh"

#include <cudf/detail/iterator.cuh>

namespace cudf::detail {

template <typename OutputIterator>
void compute_match_counts(
  table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  cudf::table_view const& probe,
  OutputIterator output_iter,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  auto const probe_table_num_rows = probe.num_rows();

  auto count_matches = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    hash_table.count_each(iter,
                          iter + probe_table_num_rows,
                          equality,
                          hash_table.hash_function(),
                          output_iter,
                          stream.value());
  };

  dispatch_join_comparator(
    build, probe, preprocessed_build, preprocessed_probe, has_nulls, compare_nulls, count_matches);
}

}  // namespace cudf::detail
