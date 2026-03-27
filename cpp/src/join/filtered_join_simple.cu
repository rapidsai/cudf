/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_detail.cuh"
#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_filtered_join.cuh>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuda/iterator>
#include <thrust/iterator/counting_iterator.h>

namespace cudf::detail {

namespace {

std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }
  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

struct gather_mask {
  join_kind kind;
  device_span<bool const> flagged;
  __device__ bool operator()(size_type idx) const noexcept
  {
    return flagged[idx] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};

using fj = filtered_join;

}  // namespace

void filtered_join_insert_simple(
  cudf::null_equality nulls_equal,
  cudf::table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  fj::storage_type& bucket_storage,
  rmm::cuda_stream_view stream)
{
  auto const d_build_comparator =
    cudf::detail::row::equality::self_comparator{preprocessed_build}.equal_to<false>(
      nullate::YES{},
      nulls_equal,
      cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{fj::empty_sentinel_key,
                               fj::insertion_adapter{d_build_comparator},
                               fj::simple_probing_scheme{},
                               cuco::thread_scope_device,
                               bucket_storage.ref()};
  auto insert_ref = set_ref.rebind_operators(cuco::insert);

  constexpr auto CGSize = fj::simple_probing_scheme::cg_size;
  auto insert           = [&]<typename Iterator>(Iterator build_iter) {
    auto const grid_size = cuco::detail::grid_size(build.num_rows(), CGSize);
    if (cudf::has_nested_nulls(build) && nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(build, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          build.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          build.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
    }
  };

  auto const d_build_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_build}.device_hasher(nullate::YES{});
  auto const build_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, fj::key_pair_fn<lhs_index_type, fj::row_hasher>{d_build_hasher});
  insert(build_iter);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_query_simple(
  cudf::table_view const& build,
  cudf::table_view const& probe,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  cudf::null_equality nulls_equal,
  fj::storage_type const& bucket_storage,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const probe_has_nulls = has_nested_nulls(probe);

  auto const d_build_probe_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_build, preprocessed_probe};
  auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<false>(
    nullate::YES{},
    nulls_equal,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{fj::empty_sentinel_key,
                               fj::comparator_adapter{d_build_probe_nan_comparator},
                               fj::simple_probing_scheme{},
                               cuco::thread_scope_device,
                               bucket_storage.ref()};
  auto query_ref = set_ref.rebind_operators(cuco::op::contains);

  constexpr auto CGSize = fj::simple_probing_scheme::cg_size;
  auto contains_map     = rmm::device_uvector<bool>(probe.num_rows(), stream);

  auto query_set = [&]<typename InputProbeIterator>(InputProbeIterator probe_iter) {
    auto const grid_size = cuco::detail::grid_size(probe.num_rows(), CGSize);
    if (probe_has_nulls && nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contains_map.begin(),
          query_ref);
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          contains_map.begin(),
          query_ref);
    }
  };

  auto const d_probe_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(nullate::YES{});
  auto const probe_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, fj::key_pair_fn<rhs_index_type, fj::row_hasher>{d_probe_hasher});
  query_set(probe_iter);

  rmm::device_uvector<size_type> gather_map(probe.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy_nosync(stream),
                                        thrust::counting_iterator<size_type>(0),
                                        thrust::counting_iterator<size_type>(probe.num_rows()),
                                        gather_map.begin(),
                                        gather_mask{kind, contains_map});
  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

}  // namespace cudf::detail
