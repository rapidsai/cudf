/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "key_remapping_fwd.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/join.hpp>
#include <cudf/join/key_remapping.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/replace.h>

#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

std::unique_ptr<key_remap_table_interface> create_key_remap_table(cudf::table_view const& build,
                                                                  cudf::null_equality compare_nulls,
                                                                  bool compute_metrics,
                                                                  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (build.num_rows() == 0 || build.num_columns() == 0) { return nullptr; }

  auto preprocessed_build = cudf::detail::row::equality::preprocessed_table::create(build, stream);

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    return create_key_remap_table_primitive(
      build, std::move(preprocessed_build), compare_nulls, compute_metrics, stream);
  }

  if (cudf::has_nested_columns(build)) {
    return create_key_remap_table_nested(
      build, std::move(preprocessed_build), compare_nulls, compute_metrics, stream);
  }

  return create_key_remap_table_non_nested(
    build, std::move(preprocessed_build), compare_nulls, compute_metrics, stream);
}

}  // namespace

/**
 * @brief Implementation class for key_remapping
 */
class key_remapping_impl {
  friend class cudf::key_remapping;

 public:
  key_remapping_impl(cudf::table_view const& build,
                     cudf::null_equality compare_nulls,
                     bool compute_metrics,
                     rmm::cuda_stream_view stream)
    : _build{build},
      _compare_nulls{compare_nulls},
      _compute_metrics{compute_metrics},
      _table{create_key_remap_table(build, compare_nulls, compute_metrics, stream)}
  {
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe(
    cudf::table_view const& keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(keys.num_columns() == _build.num_columns(),
                 "Mismatch in number of columns to be joined on",
                 std::invalid_argument);

    if (keys.num_rows() == 0) {
      return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
    }

    CUDF_EXPECTS(cudf::have_same_types(_build, keys),
                 "Mismatch in joining column data types",
                 cudf::data_type_error);

    if (_table == nullptr) {
      auto result =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(keys.num_rows(), stream, mr);
      thrust::fill(
        rmm::exec_policy_nosync(stream), result->begin(), result->end(), cudf::JoinNoMatch);
      return result;
    }
    return _table->probe(keys, stream, mr);
  }

  bool has_metrics() const { return _table ? _table->has_metrics() : _compute_metrics; }

  cudf::size_type get_distinct_count() const
  {
    if (_table) { return _table->get_distinct_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::size_type get_max_duplicate_count() const
  {
    if (_table) { return _table->get_max_duplicate_count(); }
    CUDF_EXPECTS(_compute_metrics, "Metrics were not computed during construction");
    return 0;
  }

  cudf::null_equality get_compare_nulls() const { return _compare_nulls; }

 private:
  cudf::table_view const& get_build() const { return _build; }

  cudf::table_view _build;
  cudf::null_equality _compare_nulls;
  bool _compute_metrics;
  std::unique_ptr<key_remap_table_interface> _table;
};

}  // namespace detail

// Public API implementation

key_remapping::key_remapping(cudf::table_view const& build,
                             null_equality compare_nulls,
                             cudf::compute_metrics metrics,
                             rmm::cuda_stream_view stream)
  : _impl{std::make_unique<detail::key_remapping_impl>(
      build, compare_nulls, static_cast<bool>(metrics), stream)}
{
  CUDF_EXPECTS(build.num_columns() > 0, "Build table must have at least one column");
}

key_remapping::~key_remapping() = default;

namespace {
std::unique_ptr<cudf::column> remap_keys_internal(detail::key_remapping_impl const& impl,
                                                  cudf::table_view const& keys,
                                                  cudf::size_type not_found_sentinel,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto indices = impl.probe(keys, stream, mr);

  if (indices->size() == 0) { return cudf::make_empty_column(cudf::type_id::INT32); }

  thrust::replace(rmm::exec_policy_nosync(stream),
                  indices->begin(),
                  indices->end(),
                  cudf::JoinNoMatch,
                  not_found_sentinel);

  auto const row_count = static_cast<cudf::size_type>(indices->size());
  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, row_count, indices->release(), rmm::device_buffer{}, 0);
}
}  // namespace

std::unique_ptr<cudf::column> key_remapping::remap_build_keys(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  // Use the cached build table from the implementation
  return remap_keys_internal(*_impl, _impl->get_build(), KEY_REMAP_BUILD_NULL, stream, mr);
}

std::unique_ptr<cudf::column> key_remapping::remap_probe_keys(
  cudf::table_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return remap_keys_internal(*_impl, keys, KEY_REMAP_NOT_FOUND, stream, mr);
}

bool key_remapping::has_metrics() const { return _impl->has_metrics(); }

size_type key_remapping::get_distinct_count() const { return _impl->get_distinct_count(); }

size_type key_remapping::get_max_duplicate_count() const
{
  return _impl->get_max_duplicate_count();
}

}  // namespace cudf
