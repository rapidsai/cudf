/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/compute_single_pass_aggs.hpp"
#include "groupby/hash/group_keys.hpp"
#include "groupby_helper_group_offsets.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <utility>

namespace cudf::groupby::detail {

size_type compact_group_offsets(bool* flags,
                                size_type size,
                                size_type* offsets,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  auto const begin = cuda::counting_iterator<size_type>{0};
  auto const end =
    cudf::detail::copy_if(begin, begin + size, flags, offsets, cuda::std::identity{}, stream, mr);
  return static_cast<size_type>(end - offsets);
}

namespace {

/**
 * @brief Restores input order within each group of the grouped rows.
 *
 * Uses `group_labels` as the group of each grouped row when they are cached. Otherwise, when
 * `keep_labels` is set, caches the labels sorted along with the rows in `group_labels`.
 */
void stabilize_rows(hash::grouped_keys& groups,
                    size_type size,
                    std::unique_ptr<cuda::device_buffer<size_type>>& group_labels,
                    size_type num_rows,
                    bool keep_labels,
                    cuda::stream_ref stream)
{
  auto const num_groups = static_cast<size_type>(groups.key_indices.size());
  // HashCSR determines group membership. Stably sorting those group IDs in input-row order
  // preserves the existing group offsets while restoring input order within each group.
  auto const mr           = cudf::get_current_device_resource_ref();
  auto const policy       = rmm::exec_policy_nosync(stream, mr);
  auto const filtered     = size != num_rows;
  auto const store_labels = keep_labels && !group_labels;
  auto input_labels       = cuda::device_buffer<size_type>(stream, mr, num_rows, cuda::no_init);
  auto ordered_labels     = cuda::device_buffer<size_type>(stream, mr, num_rows, cuda::no_init);
  auto ordered_rows       = cuda::device_buffer<size_type>(stream, mr, num_rows, cuda::no_init);
  if (filtered) {
    auto const initialize_rows = cuda::tabulate_output_iterator{
      [labels = input_labels.data(), rows = ordered_rows.data(), num_groups] __device__(
        cuda::std::ptrdiff_t index, size_type row) {
        labels[index] = num_groups;
        rows[index]   = row;
      }};
    thrust::sequence(policy, initialize_rows, initialize_rows + num_rows, size_type{0});
  }

  // Write into separate storage while the original CSR permutation is still being read.
  auto const write_label = [labels = input_labels.data(),
                            rows   = ordered_rows.data(),
                            filtered] __device__(cuda::std::ptrdiff_t row, size_type label) {
    labels[row] = label;
    if (!filtered) { rows[row] = static_cast<size_type>(row); }
  };
  auto const label_output = cuda::tabulate_output_iterator{
    [grouped_rows = groups.grouped_rows->data(), write_label] __device__(
      cuda::std::ptrdiff_t position, size_type label) {
      write_label(grouped_rows[position], label);
    }};
  if (group_labels) {
    thrust::copy_n(policy, group_labels->begin(), size, label_output);
  } else {
    // A CSR position's group ID is the number of segment ends at or before that position.
    // Compute it directly instead of materializing labels that only this ordering needs.
    auto const offsets = device_span<size_type const>{groups.group_offsets->data(),
                                                      static_cast<std::size_t>(num_groups) + 1};
    thrust::upper_bound(policy,
                        offsets.begin() + 1,
                        offsets.end(),
                        cuda::counting_iterator<size_type>{0},
                        cuda::counting_iterator<size_type>{size},
                        label_output);
  }

  // The unfiltered CSR buffer is now free to serve as the alternate sorting buffer. With
  // exclusions, retain its exact-sized allocation for the sorted prefix instead.
  auto alternate_rows =
    cuda::device_buffer<size_type>(stream, mr, filtered ? num_rows : 0, cuda::no_init);
  // Only excluded rows use the sentinel group ID, which sorts after every included group.
  auto const end_bit =
    cuda::std::bit_width(static_cast<cuda::std::uint32_t>(filtered ? num_groups : num_groups - 1));
  auto const sorted = hash::stable_sort_group_rows(
    {input_labels.data(), ordered_rows.data()},
    {ordered_labels.data(), filtered ? alternate_rows.data() : groups.grouped_rows->data()},
    num_rows,
    end_bit,
    stream,
    mr);
  if (filtered) {
    thrust::copy_n(policy, sorted.rows, size, groups.grouped_rows->begin());
  } else if (sorted.rows == ordered_rows.data()) {
    groups.grouped_rows = std::move(ordered_rows);
  }
  // The sorted labels are the group of each grouped row; keep them for group_labels().
  if (store_labels) {
    auto labels =
      sorted.labels == input_labels.data() ? std::move(input_labels) : std::move(ordered_labels);
    if (labels.size() != static_cast<std::size_t>(size)) {
      labels = cuda::device_buffer<size_type>{stream, mr, labels.begin(), labels.begin() + size};
    }
    group_labels = std::make_unique<cuda::device_buffer<size_type>>(std::move(labels));
  }
}

}  // namespace

groupby_helper::groupby_helper(table_view const& keys,
                               null_policy include_null_keys,
                               sorted keys_pre_sorted)
  : _keys{keys}, _keys_pre_sorted{keys_pre_sorted}, _include_null_keys{include_null_keys}
{
  // Cannot depend on caller's sorting if the column contains nulls,
  // and null values are to be excluded.
  // Hash the keys instead, to filter out nulls more easily.
  if (keys_pre_sorted == sorted::YES and include_null_keys == null_policy::EXCLUDE and
      has_nulls(keys)) {
    _keys_pre_sorted = sorted::NO;
  }
}

groupby_helper::~groupby_helper()                                    = default;
groupby_helper::groupby_helper(groupby_helper&&) noexcept            = default;
groupby_helper& groupby_helper::operator=(groupby_helper&&) noexcept = default;

void groupby_helper::build_groups(bool need_grouped_rows,
                                  bool stable_rows,
                                  bool keep_labels,
                                  cuda::stream_ref stream)
{
  if (_groups) {
    if (!need_grouped_rows || _groups->grouped_rows) { return; }
    // A previous counts-only aggregation needed no row permutation. Build it only when a later
    // operation needs rows; no result cache is shared across aggregation calls.
    _groups.reset();
    _stable = false;
  }
  auto const mr = cudf::get_current_device_resource_ref();
  if (_keys_pre_sorted == sorted::NO) {
    _groups = std::make_unique<hash::grouped_keys>(hash::group_keys(
      _keys, _include_null_keys, true, need_grouped_rows, stable_rows, keep_labels, stream, mr));
    if (_groups->grouped_rows) {
      _num_keys = static_cast<size_type>(_groups->grouped_rows->size());
    }
    if (_groups->group_labels) {
      // The stable build sorted the labels along with the rows.
      _group_labels =
        std::make_unique<cuda::device_buffer<size_type>>(std::move(*_groups->group_labels));
    }
    _stable = stable_rows;
    return;
  }

  // Sorted keys already form contiguous groups in input order, so adjacent equality identifies
  // boundaries without sorting or hashing.
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  auto const size   = _keys.num_rows();
  auto rows         = cuda::device_buffer<size_type>(stream, mr, size, cuda::no_init);
  thrust::sequence(policy, rows.begin(), rows.end(), size_type{0});
  auto offsets =
    cuda::device_buffer<size_type>(stream, mr, static_cast<std::size_t>(size) + 1, cuda::no_init);
  auto const num_groups =
    size == 0 ? size_type{0}
    : cudf::detail::has_nested_columns(_keys)
      ? compute_nested_group_offsets(_keys, rows.data(), size, offsets, stream)
      : compute_group_offsets<false>(_keys, rows.data(), size, offsets, stream);
  thrust::fill_n(policy, offsets.data() + num_groups, 1, size);
  // The first row of each group is its representative input row, which the group offset already
  // identifies.
  _groups = std::make_unique<hash::grouped_keys>(
    cuda::device_buffer<size_type>(stream, mr), std::move(offsets), std::move(rows), num_groups);
  _groups->key_indices = {_groups->group_offsets->data(), static_cast<std::size_t>(num_groups)};
  _num_keys            = size;
  _stable              = true;
}

void groupby_helper::make_stable(bool keep_labels, cuda::stream_ref stream)
{
  build_groups(true, true, keep_labels, stream);
  if (_stable) { return; }
  // Rows cached by an unordered operation keep their groups and are reordered within them.
  // Scheduling contains spans into the current row buffer and hints derived from its order.
  _reduction_groups.reset();
  if (static_cast<size_type>(_groups->key_indices.size()) < _num_keys) {
    stabilize_rows(*_groups, _num_keys, _group_labels, _keys.num_rows(), keep_labels, stream);
  }
  _stable = true;
}

size_type groupby_helper::num_keys(cuda::stream_ref stream)
{
  build_groups(true, false, false, stream);
  return _num_keys;
}

column_view groupby_helper::grouped_order(bool keep_labels, cuda::stream_ref stream)
{
  make_stable(keep_labels, stream);
  return column_view(device_span<size_type const>{_groups->grouped_rows->data(),
                                                  static_cast<std::size_t>(_num_keys)});
}

device_span<size_type const> groupby_helper::unordered_grouped_order(cuda::stream_ref stream)
{
  build_groups(true, false, false, stream);
  return {_groups->grouped_rows->data(), static_cast<std::size_t>(_num_keys)};
}

hash::grouped_rows const& groupby_helper::reduction_groups(cuda::stream_ref stream)
{
  if (!_reduction_groups) {
    // Build the grouped rows before reading offsets, which a counts-only grouping rebuild changes.
    auto const rows   = unordered_grouped_order(stream);
    _reduction_groups = std::make_unique<hash::grouped_rows>(hash::make_grouped_rows(
      rows, group_offsets(stream), stream, cudf::get_current_device_resource_ref()));
  }
  return *_reduction_groups;
}

device_span<size_type const> groupby_helper::group_offsets(cuda::stream_ref stream)
{
  build_groups(false, false, false, stream);
  return {_groups->group_offsets->data(), _groups->key_indices.size() + 1};
}

device_span<size_type const> groupby_helper::group_labels(cuda::stream_ref stream)
{
  // Labels must use the final grouping order before any row consumer can observe them.
  build_groups(true, false, false, stream);
  if (!_group_labels) {
    auto labels = std::make_unique<cuda::device_buffer<size_type>>(
      stream, cudf::get_current_device_resource_ref(), _num_keys, cuda::no_init);
    if (!labels->empty()) {
      auto const offsets = group_offsets(stream);
      cudf::detail::label_segments(
        offsets.begin(), offsets.end(), labels->begin(), labels->end(), stream);
    }
    _group_labels = std::move(labels);
  }
  return {_group_labels->data(), static_cast<std::size_t>(_num_keys)};
}

device_span<size_type const> groupby_helper::input_labels(cuda::stream_ref stream)
{
  if (_input_labels) { return *_input_labels; }
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const policy  = rmm::exec_policy_nosync(stream, mr);
  auto const& labels = group_labels(stream);
  auto input =
    std::make_unique<cuda::device_buffer<size_type>>(stream, mr, _keys.num_rows(), cuda::no_init);
  // Excluded rows take the label past the last group so that they sort after every group.
  if (labels.size() != input->size()) {
    thrust::fill(policy, input->begin(), input->end(), num_groups(stream));
  }
  // Labels are constant within each group, so the unordered CSR permutation is sufficient.
  thrust::scatter(
    policy, labels.begin(), labels.end(), _groups->grouped_rows->begin(), input->begin());
  _input_labels = std::move(input);
  return *_input_labels;
}

std::unique_ptr<column> groupby_helper::sorted_values(column_view const& values,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  // Group labels avoid sorting the original key columns. Starting with input rows also preserves
  // input order among equal values without materializing the stable CSR row permutation first.
  auto const labels = column_view(device_span<size_type const>{input_labels(stream)});
  auto order        = cudf::detail::stable_sorted_order(
    table_view{{labels, values}}, {}, std::vector<null_order>(2, null_order::AFTER), stream, mr);
  auto rows   = cudf::detail::slice(order->view(), 0, num_keys(stream), stream);
  auto result = cudf::detail::gather(table_view{{values}},
                                     rows,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> groupby_helper::grouped_values(column_view const& values,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto result = cudf::detail::gather(table_view{{values}},
                                     grouped_order(false, stream),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> groupby_helper::unordered_grouped_values(column_view const& values,
                                                                 cuda::stream_ref stream,
                                                                 rmm::device_async_resource_ref mr)
{
  auto result = cudf::detail::gather(table_view{{values}},
                                     unordered_grouped_order(stream),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<table> groupby_helper::unique_keys(cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  std::unique_ptr<hash::grouped_keys> key_groups;
  if (!_groups && _keys_pre_sorted == sorted::NO) {
    // Empty aggregation requests need only representatives. Keep this metadata local so a
    // later operation can populate the helper with complete grouped rows.
    key_groups = std::make_unique<hash::grouped_keys>(
      hash::group_keys(_keys,
                       _include_null_keys,
                       false,
                       false,
                       false,
                       false,
                       stream,
                       cudf::get_current_device_resource_ref()));
  } else {
    build_groups(false, false, false, stream);
  }
  auto const& groups = key_groups ? *key_groups : *_groups;
  return cudf::detail::gather(_keys,
                              groups.key_indices,
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

std::unique_ptr<table> groupby_helper::grouped_keys(cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return cudf::detail::gather(_keys,
                              grouped_order(false, stream),
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace cudf::groupby::detail
