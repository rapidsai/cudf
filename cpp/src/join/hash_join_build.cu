/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hash_join_helpers.cuh"

#include <cudf/utilities/error.hpp>

namespace cudf::detail {

void build_hash_join(
  cudf::table_view const& build,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_build,
  hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Selected build dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != build.num_rows(), "Build side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& build, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(build)) {
      hash_table.insert(iter, iter + build.num_rows(), stream.value());
    } else {
      auto const stencil = thrust::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      // insert valid rows
      hash_table.insert_if(iter, iter + build.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nulls, preprocessed_build};
    insert_rows(build, d_hasher);
  } else {
    auto const row_hash = detail::row::hash::row_hasher{preprocessed_build};
    auto const d_hasher = row_hash.device_hasher(nulls);
    insert_rows(build, d_hasher);
  }
}

}  // namespace cudf::detail
