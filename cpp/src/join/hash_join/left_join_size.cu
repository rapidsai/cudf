/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "size_common.cuh"

#include <cudf/detail/nvtx/ranges.hpp>

namespace cudf::detail {

template <typename Hasher>
std::size_t hash_join<Hasher>::left_join_size(cudf::table_view const& probe,
                                              rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  if (_is_empty) { return probe.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::compute_join_output_size<join_kind::LEFT_JOIN>(_build,
                                                                      probe,
                                                                      _preprocessed_build,
                                                                      preprocessed_probe,
                                                                      _hash_table,
                                                                      _has_nulls,
                                                                      _nulls_equal,
                                                                      stream);
}

template std::size_t hash_join<hash_join_hasher>::left_join_size(
  cudf::table_view const& probe, rmm::cuda_stream_view stream) const;

}  // namespace cudf::detail
