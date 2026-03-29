/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "key_remapping_impl.cuh"

namespace cudf {
namespace detail {

std::unique_ptr<key_remap_table_interface> create_key_remap_table_non_nested(
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build,
  cudf::null_equality compare_nulls,
  bool compute_metrics,
  rmm::cuda_stream_view stream)
{
  auto const self_equal = cudf::detail::row::equality::self_comparator(preprocessed_build);
  auto const row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_build};
  auto const d_hasher   = row_hasher.device_hasher(cudf::nullate::DYNAMIC{HAS_NULLS});

  auto const d_equal = self_equal.equal_to<false>(cudf::nullate::DYNAMIC{HAS_NULLS}, compare_nulls);

  using comparator_type = build_comparator<decltype(d_equal)>;
  return std::make_unique<key_remap_table<comparator_type>>(build,
                                                            preprocessed_build,
                                                            comparator_type{d_equal},
                                                            d_hasher,
                                                            compare_nulls,
                                                            compute_metrics,
                                                            stream);
}

}  // namespace detail
}  // namespace cudf
