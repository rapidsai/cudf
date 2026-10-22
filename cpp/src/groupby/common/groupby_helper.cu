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
#include <cuda/std/functional>
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
  if (_groups && (!need_grouped_rows || (_groups->grouped_rows && (_stable || !stable_rows)))) {
    return;
  }
  // A rebuild may change the group order, so drop everything derived from the old grouping.
  _groups.reset();
  _group_labels.reset();
  _input_labels.reset();
  _reduction_groups.reset();
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
    // Singleton groups retain input order in any build.
    _stable = stable_rows || static_cast<size_type>(_groups->key_indices.size()) == _num_keys;
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
  // The first row of each group represents it, and in input order its index is the group offset.
  auto key_indices =
    cuda::device_buffer<size_type>(stream, mr, offsets.data(), offsets.data() + num_groups);
  _groups = std::make_unique<hash::grouped_keys>(
    std::move(key_indices), std::move(offsets), std::move(rows), num_groups);
  _num_keys = size;
  _stable   = true;
}

size_type groupby_helper::num_keys(cuda::stream_ref stream)
{
  build_groups(true, false, false, stream);
  return _num_keys;
}

column_view groupby_helper::grouped_order(bool keep_labels, cuda::stream_ref stream)
{
  build_groups(true, true, keep_labels, stream);
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
    // Request the grouped rows before the offsets, which a rebuild for the rows invalidates.
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
