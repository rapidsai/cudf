/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

namespace cudf {
namespace detail {
namespace row {
namespace equality {
struct preprocessed_table;
}  // namespace equality
}  // namespace row

/**
 * @brief Abstract interface for key remap hash table implementations.
 */
class key_remap_table_interface {
 public:
  virtual ~key_remap_table_interface() = default;

  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe(
    cudf::table_view const& probe_keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;

  virtual bool has_metrics() const                        = 0;
  virtual cudf::size_type get_distinct_count() const      = 0;
  virtual cudf::size_type get_max_duplicate_count() const = 0;
};

std::unique_ptr<key_remap_table_interface> create_key_remap_table_primitive(
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build,
  cudf::null_equality compare_nulls,
  bool compute_metrics,
  rmm::cuda_stream_view stream);

std::unique_ptr<key_remap_table_interface> create_key_remap_table_nested(
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build,
  cudf::null_equality compare_nulls,
  bool compute_metrics,
  rmm::cuda_stream_view stream);

std::unique_ptr<key_remap_table_interface> create_key_remap_table_non_nested(
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build,
  cudf::null_equality compare_nulls,
  bool compute_metrics,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace cudf
