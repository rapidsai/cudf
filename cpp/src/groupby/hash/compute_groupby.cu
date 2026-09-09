/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "groupby/common/utils.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "hash_csr_kernels.cuh"
#include "helpers.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cudf::groupby::detail::hash {

namespace {

/// The keys grouped by the HashCSR build.
struct grouped_keys {
  size_type num_groups;
  size_type num_grouped_rows;
  rmm::device_uvector<size_type> key_rows;       ///< One representative input row per group
  rmm::device_uvector<size_type> group_offsets;  ///< `num_groups + 1` offsets into `grouped_rows`
  rmm::device_uvector<size_type> grouped_rows;   ///< Input rows reordered so groups are contiguous
};

cuda::std::uint32_t hash_csr_capacity(size_type num_rows)
{
  auto const requested =
    std::max(static_cast<double>(num_rows) + 1,
             std::ceil(static_cast<double>(num_rows) / cudf::detail::CUCO_DESIRED_LOAD_FACTOR));
  CUDF_EXPECTS(requested <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "HashCSR table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(requested);
}

/**
 * @brief Groups the input rows by key with a HashCSR build.
 *
 * Every valid row inserts its key into an open-addressed table and takes a rank within the slot
 * it lands in. The occupied slots become the groups, a scan of their row counts gives the group
 * offsets, and a scatter of the rows by slot offset plus rank yields the grouped row order.
 */
template <typename Equal, typename Hash>
grouped_keys group_keys(size_type num_rows,
                        bitmask_type const* row_bitmask,
                        Equal const& d_row_equal,
                        Hash const& d_row_hash,
                        bool need_grouped_rows,
                        cuda::stream_ref stream)
{
  auto const temp_mr  = cudf::get_current_device_resource_ref();
  auto const capacity = hash_csr_capacity(num_rows);
  auto const policy   = rmm::exec_policy_nosync(stream, temp_mr);

  rmm::device_uvector<hash_table_entry_type> entries(capacity, stream, temp_mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    entries.data(), 0xff, entries.size() * sizeof(hash_table_entry_type), stream.get()));
  auto const table       = hash_csr_table_ref{entries.data(), capacity};
  auto const is_occupied = [] __device__(size_type row) -> bool {
    return row != cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
  };
  auto const entry_rows = entries.begin();

  if (!need_grouped_rows) {
    // Only the distinct keys are needed, and the occupied slots hold one row for each of them.
    launch_hash_csr_build_kernel(
      num_rows, row_bitmask, nullptr, nullptr, table, d_row_equal, d_row_hash, stream);
    rmm::device_uvector<size_type> key_rows(num_rows, stream, temp_mr);
    auto const key_rows_end =
      thrust::copy_if(policy, entry_rows, entry_rows + capacity, key_rows.begin(), is_occupied);
    key_rows.resize(cuda::std::distance(key_rows.begin(), key_rows_end), stream);
    auto const num_groups = static_cast<size_type>(key_rows.size());
    return {num_groups,
            0,
            std::move(key_rows),
            rmm::device_uvector<size_type>{0, stream, temp_mr},
            rmm::device_uvector<size_type>{0, stream, temp_mr}};
  }

  rmm::device_uvector<size_type> slot_counts(capacity, stream, temp_mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(slot_counts.data(), 0, slot_counts.size() * sizeof(size_type), stream.get()));
  rmm::device_uvector<build_position_type> positions(num_rows, stream, temp_mr);
  launch_hash_csr_build_kernel(num_rows,
                               row_bitmask,
                               positions.data(),
                               slot_counts.data(),
                               table,
                               d_row_equal,
                               d_row_hash,
                               stream);

  // The occupied slots, in slot order, are the groups.
  rmm::device_uvector<cuda::std::uint32_t> group_slots(num_rows, stream, temp_mr);
  auto const group_slots_end =
    thrust::copy_if(policy,
                    cuda::counting_iterator<cuda::std::uint32_t>{0},
                    cuda::counting_iterator<cuda::std::uint32_t>{capacity},
                    slot_counts.begin(),
                    group_slots.begin(),
                    [] __device__(size_type count) -> bool { return count > 0; });
  group_slots.resize(cuda::std::distance(group_slots.begin(), group_slots_end), stream);
  auto const num_groups = static_cast<size_type>(group_slots.size());

  rmm::device_uvector<size_type> key_rows(num_groups, stream, temp_mr);
  thrust::gather(policy, group_slots.begin(), group_slots.end(), entry_rows, key_rows.begin());
  entries.resize(0, stream);
  entries.shrink_to_fit(stream);

  rmm::device_uvector<size_type> group_offsets(num_groups + 1, stream, temp_mr);
  group_offsets.set_element_to_zero_async(0, stream);
  auto const group_counts =
    cuda::make_permutation_iterator(slot_counts.begin(), group_slots.begin());
  thrust::inclusive_scan(
    policy, group_counts, group_counts + num_groups, group_offsets.begin() + 1);
  auto const num_grouped_rows =
    row_bitmask == nullptr ? num_rows : group_offsets.back_element(stream);

  // Reuse the slot counts to hold the start offset of the group of each occupied slot, then
  // scatter every row to its group.
  thrust::scatter(policy,
                  group_offsets.begin(),
                  group_offsets.begin() + num_groups,
                  group_slots.begin(),
                  slot_counts.begin());
  rmm::device_uvector<size_type> grouped_rows(num_grouped_rows, stream, temp_mr);
  launch_hash_csr_fill_kernel(
    num_rows, positions.data(), slot_counts.data(), grouped_rows.data(), stream);

  return {num_groups,
          num_grouped_rows,
          std::move(key_rows),
          std::move(group_offsets),
          std::move(grouped_rows)};
}

}  // namespace

template <typename Equal, typename Hash>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       std::span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       Hash const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const num_rows = keys.num_rows();

  [[maybe_unused]] auto [row_bitmask_data, row_bitmask] =
    skip_rows_with_nulls
      ? cudf::groupby::detail::compute_row_bitmask(keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{
          rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()}, nullptr};

  auto const groups =
    group_keys(num_rows, row_bitmask, d_row_equal, d_row_hash, !requests.empty(), stream);

  auto const gather_keys = [&] {
    return cudf::detail::gather(keys,
                                groups.key_rows,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  // In case of no requests, we still need to generate a set of unique keys.
  if (requests.empty()) { return gather_keys(); }

  // Compute all single pass aggs first.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_single_pass_aggs(requests, stream);

  auto const grouped = make_grouped_rows(groups.grouped_rows, groups.group_offsets, stream);
  auto results =
    compute_single_pass_aggs(values, agg_kinds, is_agg_intermediate, grouped, stream, mr);
  for (std::size_t i = 0; i < results.size(); ++i) {
    cache->add_result(values.column(i), *aggs[i], std::move(results[i]));
  }

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The finalizers only combine the single-pass results with linear transformations such as
      // addition/multiplication (e.g. for variance/stddev); they do not aggregate further.
      auto const finalizer = hash_compound_agg_finalizer(col, cache, row_bitmask, stream, mr);
      for (auto&& agg : agg_v) {
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  return gather_keys();
}

template std::unique_ptr<table> compute_groupby<row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
