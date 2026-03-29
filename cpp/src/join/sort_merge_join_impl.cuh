/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort_merge_join.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_copy.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <memory>
#include <utility>

namespace cudf::detail::sort_merge_join_detail {

/**
 * @brief Functor to map indices through a provided mapping container.
 */
template <typename T>
struct index_mapping {
  T mapping;

  __device__ size_type operator()(size_type idx) const noexcept
  {
    return idx >= 0 ? mapping[idx] : idx;
  }
};

template <typename InputIt, typename ScalarType>
cudf::detail::device_scalar<ScalarType> reduce(InputIt input,
                                               cudf::detail::device_scalar<ScalarType>&& output,
                                               size_type num_items,
                                               rmm::cuda_stream_view stream)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, input, output.data(), num_items, stream.value());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream);
  cub::DeviceReduce::Sum(
    temp_storage.data(), temp_storage_bytes, input, output.data(), num_items, stream.value());
  return output;
}

template <typename InputIts, typename OutputIts, typename SizeIt>
void batched_copy(InputIts input_iterators,
                  OutputIts output_iterators,
                  SizeIt sizes,
                  size_type num_ranges,
                  rmm::cuda_stream_view stream)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(nullptr,
                           temp_storage_bytes,
                           input_iterators,
                           output_iterators,
                           sizes,
                           num_ranges,
                           stream.value());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream);
  cub::DeviceCopy::Batched(temp_storage.data(),
                           temp_storage_bytes,
                           input_iterators,
                           output_iterators,
                           sizes,
                           num_ranges,
                           stream.value());
}

template <typename LargerIterator, typename SmallerIterator>
class merge {
 private:
  table_view smaller;
  table_view larger;
  SmallerIterator sorted_smaller_order_begin;
  SmallerIterator sorted_smaller_order_end;
  LargerIterator sorted_larger_order_begin;
  LargerIterator sorted_larger_order_end;
  std::unique_ptr<detail::row::lexicographic::two_table_comparator> tt_comparator;

 public:
  merge(table_view const& smaller,
        SmallerIterator sorted_smaller_order_begin,
        SmallerIterator sorted_smaller_order_end,
        table_view const& larger,
        LargerIterator sorted_larger_order_begin,
        LargerIterator sorted_larger_order_end,
        rmm::cuda_stream_view stream)
    : smaller{smaller},
      sorted_smaller_order_begin{sorted_smaller_order_begin},
      sorted_smaller_order_end{sorted_smaller_order_end},
      larger{larger},
      sorted_larger_order_begin{sorted_larger_order_begin},
      sorted_larger_order_end{sorted_larger_order_end}
  {
    std::vector<cudf::order> column_order(smaller.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_precedence(smaller.num_columns(), cudf::null_order::BEFORE);
    tt_comparator = std::make_unique<detail::row::lexicographic::two_table_comparator>(
      smaller, larger, column_order, null_precedence, stream);
  }

  std::unique_ptr<rmm::device_uvector<size_type>> matches_per_row(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};

template <typename LargerIterator, typename SmallerIterator>
std::unique_ptr<rmm::device_uvector<size_type>>
merge<LargerIterator, SmallerIterator>::matches_per_row(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();
  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(larger_numrows + 1, stream, mr);

  auto const comparator = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
  auto match_counts_it  = match_counts.begin();
  auto smaller_it =
    cuda::transform_iterator(sorted_smaller_order_begin,
                             cuda::proclaim_return_type<detail::row::lhs_index_type>(
                               [] __device__(size_type idx) -> detail::row::lhs_index_type {
                                 return static_cast<detail::row::lhs_index_type>(idx);
                               }));
  thrust::upper_bound(rmm::exec_policy_nosync(stream),
                      smaller_it,
                      smaller_it + smaller_numrows,
                      cudf::detail::row::rhs_iterator(0),
                      cudf::detail::row::rhs_iterator(0) + larger_numrows,
                      match_counts_it,
                      comparator);

  auto match_counts_update_it =
    thrust::tabulate_output_iterator([match_counts = match_counts.begin()] __device__(
                                       size_type idx, size_type val) { match_counts[idx] -= val; });
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      smaller_it,
                      smaller_it + smaller_numrows,
                      cudf::detail::row::rhs_iterator(0),
                      cudf::detail::row::rhs_iterator(0) + larger_numrows,
                      match_counts_update_it,
                      comparator);

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(match_counts));
}

template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::inner(rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto temp_mr               = cudf::get_current_device_resource_ref();
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();

  auto match_counts = matches_per_row(stream, temp_mr);

  auto count_matches_it = cuda::transform_iterator(
    match_counts->begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) -> size_type { return c != 0; }));
  auto const count_matches = thrust::reduce(
    rmm::exec_policy_nosync(stream), count_matches_it, count_matches_it + larger_numrows);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, temp_mr);
  cudf::detail::copy_if_async(thrust::counting_iterator<size_type>(0),
                              thrust::counting_iterator<size_type>(larger_numrows),
                              match_counts->begin(),
                              nonzero_matches.begin(),
                              cuda::std::identity{},
                              stream);

  auto match_offsets =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(match_counts->size(), stream, temp_mr);
  auto last_element = cudf::detail::device_scalar<int64_t>(0, stream);
  auto output_itr   = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), last_element.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         match_counts->begin(),
                         match_counts->end(),
                         output_itr,
                         int64_t{0});
  auto const total_matches = static_cast<std::size_t>(last_element.value(stream));

  auto larger_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);

  {
    auto const input_iterators = cuda::transform_iterator{
      nonzero_matches.begin(),
      cuda::proclaim_return_type<cuda::constant_iterator<size_type>>(
        [] __device__(auto val) { return cuda::constant_iterator<size_type>(val); })};
    auto const output_iterators = cuda::transform_iterator{
      cuda::permutation_iterator{match_offsets.begin(), nonzero_matches.begin()},
      cuda::proclaim_return_type<rmm::device_uvector<size_type>::iterator>(
        [larger_indices = larger_indices.begin()] __device__(auto val) {
          return larger_indices + val;
        })};
    auto const sizes = cuda::permutation_iterator{match_counts->begin(), nonzero_matches.begin()};

    batched_copy(input_iterators, output_iterators, sizes, count_matches, stream);
  }

  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);

  cub::DeviceTransform::Fill(smaller_indices.begin(), smaller_indices.size(), 1, stream.value());

  {
    auto const comparator    = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
    auto smaller_tabulate_it = thrust::tabulate_output_iterator(
      [nonzero_matches = nonzero_matches.begin(),
       match_offsets   = match_offsets.begin(),
       smaller_indices = smaller_indices.begin()] __device__(auto idx, auto lb) {
        auto const lhs_idx   = nonzero_matches[idx];
        auto const pos       = match_offsets[lhs_idx];
        smaller_indices[pos] = lb;
      });
    auto smaller_it = cuda::transform_iterator(
      sorted_smaller_order_begin,
      cuda::proclaim_return_type<detail::row::lhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::lhs_index_type>(idx); }));
    auto larger_it = cuda::transform_iterator(
      nonzero_matches.begin(),
      cuda::proclaim_return_type<detail::row::rhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::rhs_index_type>(idx); }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream),
                        smaller_it,
                        smaller_it + smaller_numrows,
                        larger_it,
                        larger_it + nonzero_matches.size(),
                        smaller_tabulate_it,
                        comparator);
  }

  {
    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       larger_indices.begin(),
                                       smaller_indices.begin(),
                                       smaller_indices.begin(),
                                       total_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
    rmm::device_buffer tmp_storage(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSumByKey(tmp_storage.data(),
                                       temp_storage_bytes,
                                       larger_indices.begin(),
                                       smaller_indices.begin(),
                                       smaller_indices.begin(),
                                       total_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
  }

  cub::DeviceTransform::Transform(smaller_indices.begin(),
                                  smaller_indices.begin(),
                                  smaller_indices.size(),
                                  index_mapping<SmallerIterator>{sorted_smaller_order_begin},
                                  stream.value());

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::left(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto temp_mr               = cudf::get_current_device_resource_ref();
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();

  auto match_counts = matches_per_row(stream, temp_mr);

  cudf::detail::device_scalar<size_type> count_matches(stream, temp_mr);
  auto count_matches_it = cuda::transform_iterator(
    match_counts->begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) -> size_type { return c != 0; }));
  count_matches = reduce(count_matches_it, std::move(count_matches), larger_numrows, stream);
  auto const h_count_matches = count_matches.value(stream);
  rmm::device_uvector<size_type> nonzero_matches(h_count_matches, stream, temp_mr);
  cudf::detail::copy_if_async(cuda::counting_iterator<size_type>(0),
                              cuda::counting_iterator<size_type>(larger_numrows),
                              match_counts->begin(),
                              nonzero_matches.begin(),
                              cuda::std::identity{},
                              stream);

  cudf::detail::device_scalar<int64_t> total_matches(stream, temp_mr);
  auto match_offsets =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(match_counts->size(), stream, temp_mr);
  auto output_itr = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), total_matches.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         match_counts->begin(),
                         match_counts->end(),
                         output_itr,
                         int64_t{0});
  auto const inner_join_matches     = total_matches.value(stream);
  auto const left_join_only_matches = static_cast<int64_t>(larger_numrows - h_count_matches);
  auto larger_indices               = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    inner_join_matches + left_join_only_matches, stream, mr);
  rmm::device_uvector<size_type> smaller_indices(
    inner_join_matches + left_join_only_matches, stream, mr);

  cudf::detail::copy_if_async(
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(larger_numrows),
    match_counts->begin(),
    larger_indices.begin(),
    [] __device__(auto c) -> bool { return c == 0; },
    stream);
  cub::DeviceTransform::Fill(
    smaller_indices.begin(), left_join_only_matches, JoinNoMatch, stream.value());

  {
    auto const input_iterators = cuda::transform_iterator{
      nonzero_matches.begin(),
      cuda::proclaim_return_type<cuda::constant_iterator<size_type>>(
        [] __device__(auto val) { return cuda::constant_iterator<size_type>(val); })};
    auto const output_iterators = cuda::transform_iterator{
      cuda::permutation_iterator{match_offsets.begin(), nonzero_matches.begin()},
      cuda::proclaim_return_type<rmm::device_uvector<size_type>::iterator>(
        [larger_indices = larger_indices.begin() + left_join_only_matches] __device__(auto val) {
          return larger_indices + val;
        })};
    auto const sizes = cuda::permutation_iterator{match_counts->begin(), nonzero_matches.begin()};

    batched_copy(input_iterators, output_iterators, sizes, h_count_matches, stream);
  }

  cub::DeviceTransform::Fill(
    smaller_indices.begin() + left_join_only_matches, inner_join_matches, 1, stream.value());

  {
    auto const comparator    = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
    auto smaller_tabulate_it = thrust::tabulate_output_iterator(
      [nonzero_matches = nonzero_matches.begin(),
       match_offsets   = match_offsets.begin(),
       smaller_indices = smaller_indices.begin() + left_join_only_matches] __device__(auto idx,
                                                                                      auto lb) {
        auto const lhs_idx   = nonzero_matches[idx];
        auto const pos       = match_offsets[lhs_idx];
        smaller_indices[pos] = lb;
      });
    auto smaller_it = cuda::transform_iterator(
      sorted_smaller_order_begin,
      cuda::proclaim_return_type<detail::row::lhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::lhs_index_type>(idx); }));
    auto larger_it = cuda::transform_iterator(
      nonzero_matches.begin(),
      cuda::proclaim_return_type<detail::row::rhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::rhs_index_type>(idx); }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream),
                        smaller_it,
                        smaller_it + smaller_numrows,
                        larger_it,
                        larger_it + nonzero_matches.size(),
                        smaller_tabulate_it,
                        comparator);
  }

  {
    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       larger_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       inner_join_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
    rmm::device_buffer tmp_storage(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSumByKey(tmp_storage.data(),
                                       temp_storage_bytes,
                                       larger_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       inner_join_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
  }

  cub::DeviceTransform::Transform(smaller_indices.begin() + left_join_only_matches,
                                  smaller_indices.begin() + left_join_only_matches,
                                  inner_join_matches,
                                  index_mapping<SmallerIterator>{sorted_smaller_order_begin},
                                  stream.value());

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

}  // namespace cudf::detail::sort_merge_join_detail

namespace cudf::detail {

/**
 * @brief invoke_merge implementation — dispatches to the correct merge template based on
 * whether left/right tables have pre-computed sort orders.
 */
template <typename MergeOperation>
auto sort_merge_join::invoke_merge(preprocessed_table const& preprocessed_left,
                                   table_view right_view,
                                   table_view left_view,
                                   MergeOperation&& op,
                                   rmm::cuda_stream_view stream) const
{
  using sort_merge_join_detail::merge;
  auto has_right_sorting_order = preprocessed_right._null_processed_table_sorted_order.has_value();
  auto has_left_sorting_order  = preprocessed_left._null_processed_table_sorted_order.has_value();
  if (has_right_sorting_order && has_left_sorting_order) {
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>(),
              stream);
    return op(obj);
  } else if (has_right_sorting_order && !has_left_sorting_order) {
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              thrust::counting_iterator(0),
              thrust::counting_iterator(left_view.num_rows()),
              stream);
    return op(obj);
  } else if (!has_right_sorting_order && has_left_sorting_order) {
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              thrust::counting_iterator(0),
              thrust::counting_iterator(preprocessed_right._null_processed_table_view.num_rows()),
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>(),
              stream);
    return op(obj);
  }
  merge obj(right_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(preprocessed_right._null_processed_table_view.num_rows()),
            left_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(left_view.num_rows()),
            stream);
  return op(obj);
}

}  // namespace cudf::detail
