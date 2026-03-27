/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/join/join.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::detail {

/**
 * @brief Insert build table rows into the hash table using primitive probing scheme.
 */
void filtered_join_insert_primitive(
  bool has_nested_nulls,
  cudf::null_equality nulls_equal,
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  filtered_join::storage_type& bucket_storage,
  rmm::cuda_stream_view stream);

/**
 * @brief Insert build table rows into the hash table using nested probing scheme.
 */
void filtered_join_insert_nested(
  cudf::null_equality nulls_equal,
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  filtered_join::storage_type& bucket_storage,
  rmm::cuda_stream_view stream);

/**
 * @brief Insert build table rows into the hash table using simple probing scheme.
 */
void filtered_join_insert_simple(
  cudf::null_equality nulls_equal,
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  filtered_join::storage_type& bucket_storage,
  rmm::cuda_stream_view stream);

/**
 * @brief Query build table using primitive probing scheme.
 */
std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_query_primitive(
  cudf::table_view const& build,
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  cudf::null_equality nulls_equal,
  filtered_join::storage_type const& bucket_storage,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Query build table using nested probing scheme.
 */
std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_query_nested(
  cudf::table_view const& build,
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  cudf::null_equality nulls_equal,
  filtered_join::storage_type const& bucket_storage,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Query build table using simple probing scheme.
 */
std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_query_simple(
  cudf::table_view const& build,
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  cudf::null_equality nulls_equal,
  filtered_join::storage_type const& bucket_storage,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
