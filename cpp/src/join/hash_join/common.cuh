/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/hash_join.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/join.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>

namespace cudf::detail {

using hash_join_hasher = cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>;
using hash_table_t     = typename cudf::detail::hash_join<hash_join_hasher>::hash_table_t;

inline bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  if (left.is_empty() || right.is_empty()) { return true; }
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }
  return false;
}

void build_hash_join(
  cudf::table_view const& build,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream);

inline void validate_hash_join_probe(table_view const& build,
                                     table_view const& probe,
                                     bool has_nulls)
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty", std::invalid_argument);
  CUDF_EXPECTS(build.num_columns() == probe.num_columns(),
               "Mismatch in number of columns to be joined on",
               std::invalid_argument);
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::have_same_types(build, probe),
               "Mismatch in joining column data types",
               cudf::data_type_error);
}

std::size_t get_full_join_size(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::size_t compute_left_join_complement_size(
  std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
  size_type left_table_row_count,
  size_type right_table_row_count,
  rmm::cuda_stream_view stream);

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
