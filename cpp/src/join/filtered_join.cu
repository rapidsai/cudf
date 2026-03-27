/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_detail.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_filtered_join.cuh>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <thrust/sequence.h>

#include <memory>

namespace cudf {
namespace detail {

auto filtered_join::compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  auto const size_with_primitive_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<primitive_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                                 load_factor));
  auto const size_with_nested_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<nested_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  auto const size_with_simple_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<simple_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  return std::max({size_with_primitive_probe, size_with_nested_probe, size_with_simple_probe});
}

filtered_join::filtered_join(cudf::table_view const& build,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _build_props{build_properties{cudf::has_nested_columns(build)}},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<std::size_t>{compute_bucket_storage_size(build, load_factor)},
                    rmm::mr::polymorphic_allocator<char>{},
                    stream.value()}
{
  if (_build.num_rows() == 0) return;
  _bucket_storage.initialize(empty_sentinel_key, stream);
}

distinct_filtered_join::distinct_filtered_join(cudf::table_view const& build,
                                               cudf::null_equality compare_nulls,
                                               double load_factor,
                                               rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, load_factor, stream)
{
  cudf::scoped_range range{"distinct_filtered_join::distinct_filtered_join"};
  if (_build.num_rows() == 0) return;

  if (is_primitive_row_op_compatible(build)) {
    filtered_join_insert_primitive(cudf::has_nested_nulls(build),
                                   compare_nulls,
                                   build,
                                   _preprocessed_build,
                                   _bucket_storage,
                                   stream);
  } else if (_build_props.has_nested_columns) {
    filtered_join_insert_nested(compare_nulls, build, _preprocessed_build, _bucket_storage, stream);
  } else {
    filtered_join_insert_simple(compare_nulls, build, _preprocessed_build, _bucket_storage, stream);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_anti_join(
  cudf::table_view const& probe,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"distinct_filtered_join::semi_anti_join"};

  auto const preprocessed_probe = [&probe, stream] {
    cudf::scoped_range range{"distinct_filtered_join::semi_anti_join::preprocessed_probe"};
    return cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  }();

  if (is_primitive_row_op_compatible(_build)) {
    return filtered_join_query_primitive(_build,
                                         probe,
                                         _preprocessed_build,
                                         preprocessed_probe,
                                         kind,
                                         _nulls_equal,
                                         _bucket_storage,
                                         stream,
                                         mr);
  } else if (_build_props.has_nested_columns) {
    return filtered_join_query_nested(_build,
                                      probe,
                                      _preprocessed_build,
                                      preprocessed_probe,
                                      kind,
                                      _nulls_equal,
                                      _bucket_storage,
                                      stream,
                                      mr);
  } else {
    return filtered_join_query_simple(_build,
                                      probe,
                                      _preprocessed_build,
                                      preprocessed_probe,
                                      kind,
                                      _nulls_equal,
                                      _bucket_storage,
                                      stream,
                                      mr);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (_build.num_rows() == 0 || probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (probe.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (_build.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(probe.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result->begin(), result->end());
    return result;
  }
  return semi_anti_join(probe, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             double load_factor,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(
    reuse_tbl == set_as_build_table::RIGHT,
    "Left table reuse is yet to be implemented. Filtered join requires the right table to be the "
    "build table");
  _reuse_tbl = reuse_tbl;
  _impl      = std::make_unique<cudf::detail::distinct_filtered_join>(
    build, compare_nulls, load_factor, stream);
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, reuse_tbl, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::semi_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->semi_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::anti_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->anti_join(probe, stream, mr);
}

}  // namespace cudf
