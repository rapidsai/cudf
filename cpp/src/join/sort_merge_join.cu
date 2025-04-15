/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/sort_merge_join.cuh>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/gather.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <utility>

namespace cudf {

#ifndef SORT_MERGE_JOIN_DEBUG
#define SORT_MERGE_JOIN_DEBUG 0
#endif

namespace {

#if SORT_MERGE_JOIN_DEBUG
template <typename T>
void debug_print(std::string str, host_span<const T> span)
{
  std::cout << str << " : ";
  for (const auto& element : span) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}
#endif

enum class bound_type { UPPER, LOWER };

struct row_comparator {
  row_comparator(table_device_view const lhs,
                 table_device_view const rhs,
                 device_span<detail::dremel_device_view const> lhs_dremel,
                 device_span<detail::dremel_device_view const> rhs_dremel,
                 bound_type* d_ptr)
    : _lhs{lhs},
      _rhs{rhs},
      _d_ptr{d_ptr},
      _lhs_dremel{lhs_dremel},
      _rhs_dremel{rhs_dremel},
      ub_comparator{true, _lhs, _rhs, _lhs_dremel, _rhs_dremel},
      lb_comparator{true, _rhs, _lhs, _rhs_dremel, _lhs_dremel}
  {
  }

  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    if (*_d_ptr == bound_type::UPPER) {
      return ub_comparator(lhs_index, rhs_index) == weak_ordering::LESS;
    }
    return lb_comparator(lhs_index, rhs_index) == weak_ordering::LESS;
  }

  bound_type* _d_ptr;

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  device_span<detail::dremel_device_view const> _lhs_dremel;
  device_span<detail::dremel_device_view const> _rhs_dremel;
  cudf::experimental::row::lexicographic::device_row_comparator<true, bool> ub_comparator;
  cudf::experimental::row::lexicographic::device_row_comparator<true, bool> lb_comparator;
};

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> sort(table_view const& left,
                                                                 table_view const& right,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<cudf::order> column_order(left.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(left.num_columns(), cudf::null_order::BEFORE);
  auto sorted_left_order_col = cudf::sorted_order(left, column_order, null_precedence, stream, mr);

  column_order.resize(right.num_columns(), cudf::order::ASCENDING);
  null_precedence.resize(right.num_columns(), cudf::null_order::BEFORE);
  auto sorted_right_order_col =
    cudf::sorted_order(right, column_order, null_precedence, stream, mr);

  return {std::move(sorted_left_order_col), std::move(sorted_right_order_col)};
}

template <typename SmallerIt, typename LargerIt>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
local_merge(table_view const& smaller,
      SmallerIt sorted_smaller_order_begin,
      SmallerIt sorted_smaller_order_end,
      table_view const& larger,
      LargerIt sorted_larger_order_begin,
      LargerIt sorted_larger_order_end,
      null_equality compare_nulls,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr)
{
  auto smaller_dv_ptr      = cudf::table_device_view::create(smaller, stream);
  auto larger_dv_ptr       = cudf::table_device_view::create(larger, stream);
  auto list_lex_preprocess = [stream](table_view const& table) {
    std::vector<detail::dremel_data> dremel_data;
    auto const num_list_columns = std::count_if(
      table.begin(), table.end(), [](auto const& col) { return col.type().id() == type_id::LIST; });
    auto dremel_device_views =
      detail::make_empty_host_vector<detail::dremel_device_view>(num_list_columns, stream);
    for (auto const& col : table) {
      if (col.type().id() == type_id::LIST) {
        dremel_data.push_back(detail::get_comparator_data(col, {}, false, stream));
        dremel_device_views.push_back(dremel_data.back());
      }
    }
    auto d_dremel_device_views = detail::make_device_uvector(
      dremel_device_views, stream, cudf::get_current_device_resource_ref());
    return std::make_tuple(std::move(dremel_data), std::move(d_dremel_device_views));
  };
  auto [smaller_dremel, smaller_dremel_dv] = list_lex_preprocess(smaller);
  auto [larger_dremel, larger_dremel_dv]   = list_lex_preprocess(larger);

  // naive: iterate through larger table and binary search on smaller table
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();
  rmm::device_scalar<bound_type> d_lb_type(bound_type::LOWER, stream, mr);
  rmm::device_scalar<bound_type> d_ub_type(bound_type::UPPER, stream, mr);

  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(larger_numrows + 1, stream, mr);

#if SORT_MERGE_JOIN_DEBUG
  rmm::device_uvector<size_type> lb1(larger_numrows, stream, mr);
  row_comparator comp_(
    *larger_dv_ptr, *smaller_dv_ptr, larger_dremel_dv, smaller_dremel_dv, d_lb_type.data());
  thrust::lower_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      lb1.begin(),
                      comp_);
  debug_print<size_type>("h_lb1", cudf::detail::make_host_vector(lb1, stream));

  rmm::device_uvector<size_type> ub1(larger_numrows, stream, mr);
  comp_._d_ptr = d_ub_type.data();
  thrust::upper_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      ub1.begin(),
                      comp_);
  debug_print<size_type>("h_ub1", cudf::detail::make_host_vector(ub1, stream));
#endif

  row_comparator comp(
    *larger_dv_ptr, *smaller_dv_ptr, larger_dremel_dv, smaller_dremel_dv, d_ub_type.data());
  auto match_counts_it = match_counts.begin();
  nvtxRangePushA("upper bound");
  thrust::upper_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      match_counts_it,
                      comp);
  nvtxRangePop();

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts",
                         cudf::detail::make_host_vector(match_counts, stream));
#endif

  comp._d_ptr                 = d_lb_type.data();
  auto match_counts_update_it = thrust::make_tabulate_output_iterator(
    [match_counts = match_counts.begin()] __device__(size_type idx, size_type val) {
      match_counts[idx] -= val;
    });
  nvtxRangePushA("lower bound");
  thrust::lower_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      match_counts_update_it,
                      comp);
  nvtxRangePop();

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts",
                         cudf::detail::make_host_vector(match_counts, stream));
#endif

  auto count_matches_it = thrust::make_transform_iterator(
    match_counts.begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) { return c ? 1 : 0; }));
  auto const count_matches =
    thrust::reduce(rmm::exec_policy(stream), count_matches_it, count_matches_it + larger_numrows);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, mr);
  thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + larger_numrows,
    nonzero_matches.begin(),
    [match_counts = match_counts.begin()] __device__(auto idx) { return match_counts[idx]; });

#if SORT_MERGE_JOIN_DEBUG
  std::printf("count_matches = %d\n", count_matches);
  debug_print<size_type>("h_nonzero_matches",
                         cudf::detail::make_host_vector(nonzero_matches, stream));
#endif

  thrust::exclusive_scan(
    rmm::exec_policy(stream), match_counts.begin(), match_counts.end(), match_counts.begin());
  auto const total_matches = match_counts.back_element(stream);

  // populate larger indices
  auto larger_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);
  nvtxRangePushA("larger indices");
  thrust::scatter(rmm::exec_policy(stream),
                  nonzero_matches.begin(),
                  nonzero_matches.end(),
                  thrust::make_permutation_iterator(match_counts.begin(), nonzero_matches.begin()),
                  larger_indices.begin());
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         larger_indices.begin(),
                         larger_indices.end(),
                         larger_indices.begin(),
                         thrust::maximum<size_type>{});
  nvtxRangePop();

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts",
                         cudf::detail::make_host_vector(match_counts, stream));
  rmm::device_uvector<size_type> dlb(nonzero_matches.size(), stream, mr);
  thrust::lower_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      dlb.begin(),
                      comp);
  debug_print<size_type>("h_dlb", cudf::detail::make_host_vector(dlb, stream));
#endif

  // populate smaller indices
  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);
  nvtxRangePushA("smaller indices");
  thrust::fill(rmm::exec_policy(stream), smaller_indices.begin(), smaller_indices.end(), 1);
  auto smaller_tabulate_it = thrust::make_tabulate_output_iterator(
    [nonzero_matches = nonzero_matches.begin(),
     match_counts    = match_counts.begin(),
     smaller_indices = smaller_indices.begin()] __device__(auto idx, auto lb) {
      auto lhsidx          = nonzero_matches[idx];
      auto pos             = match_counts[lhsidx];
      smaller_indices[pos] = lb;
    });
  thrust::lower_bound(rmm::exec_policy(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      smaller_tabulate_it,
                      comp);
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                larger_indices.begin(),
                                larger_indices.end(),
                                smaller_indices.begin(),
                                smaller_indices.begin());
  thrust::transform(rmm::exec_policy(stream),
                    smaller_indices.begin(),
                    smaller_indices.end(),
                    smaller_indices.begin(),
                    [sorted_smaller_order = sorted_smaller_order_begin] __device__(auto idx) {
                      return sorted_smaller_order[idx];
                    });
  nvtxRangePop();

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_larger_indices",
                         cudf::detail::make_host_vector(larger_indices, stream));
  debug_print<size_type>("h_smaller_indices",
                         cudf::detail::make_host_vector(smaller_indices, stream));
#endif

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

std::tuple<table_view,
           table_view,
           std::optional<std::unique_ptr<table>>,
           std::optional<std::unique_ptr<table>>>
preprocess_tables(table_view const left,
                  table_view const right,
                  null_equality compare_nulls,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  if (compare_nulls == null_equality::EQUAL) { return {left, right, std::nullopt, std::nullopt}; }

  auto preprocess_table = [stream, mr](table_view const& tbl) {
    // remove rows that have nulls at any nesting level
    // step 1: identify nulls at root level
    auto [validity_mask, num_nulls] = cudf::bitmask_and(tbl, stream, mr);
    // step 2: identify nulls at non-root levels
    for (size_type col_idx = 0; col_idx < tbl.num_columns(); col_idx++) {
      auto col = tbl.column(col_idx);
      if (col.type().id() == type_id::LIST) {
        auto lcv     = lists_column_view(col);
        auto offsets = lcv.offsets();
        auto child   = lcv.child();

        rmm::device_uvector<int32_t> offsets_subset(offsets.size(), stream, mr);
        rmm::device_uvector<int32_t> child_positions(offsets.size(), stream, mr);
        auto unique_end = thrust::unique_by_key_copy(
          rmm::exec_policy(stream),
          thrust::make_reverse_iterator(lcv.offsets_end()),
          thrust::make_reverse_iterator(lcv.offsets_end()) + offsets.size(),
          thrust::make_reverse_iterator(thrust::make_counting_iterator(offsets.size())),
          thrust::make_reverse_iterator(offsets_subset.end()),
          thrust::make_reverse_iterator(child_positions.end()));
        auto subset_size   = thrust::distance(thrust::make_reverse_iterator(offsets_subset.end()),
                                            thrust::get<0>(unique_end));
        auto subset_offset = offsets.size() - subset_size;

        auto [reduced_validity_mask, num_nulls] =
          detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                                offsets_subset.data() + subset_offset,
                                                offsets_subset.data() + offsets_subset.size() - 1,
                                                offsets_subset.data() + subset_offset + 1,
                                                null_policy::INCLUDE,
                                                std::nullopt,
                                                stream,
                                                mr);

        thrust::for_each(
          rmm::exec_policy(stream),
          thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(0) + subset_size,
          [validity_mask         = static_cast<bitmask_type*>(validity_mask.data()),
           reduced_validity_mask = static_cast<bitmask_type*>(reduced_validity_mask.data()),
           child_positions       = child_positions.begin() + subset_offset] __device__(auto idx) {
            if (!bit_is_set(reduced_validity_mask, idx))
              clear_bit(validity_mask, child_positions[idx]);
          });
      }
    }
    // step 3: construct bool column to apply mask
    cudf::scalar_type_t<bool> true_scalar(true);
    auto bool_mask = cudf::make_column_from_scalar(true_scalar, tbl.num_rows(), stream, mr);
    num_nulls =
      null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, tbl.num_rows(), stream);
    bool_mask->set_null_mask(std::move(validity_mask), num_nulls);

    auto non_null_tbl = apply_boolean_mask(tbl, *bool_mask, stream, mr);
    return non_null_tbl;
  };

  auto non_null_left  = preprocess_table(left);
  auto non_null_right = preprocess_table(right);
  auto nnlv           = non_null_left->view();
  auto nnrv           = non_null_right->view();

  return {nnlv, nnrv, std::move(non_null_left), std::move(non_null_right)};
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
postprocess_indices(table_view const& left,
                    table_view const& right,
                    std::unique_ptr<rmm::device_uvector<size_type>> smaller_indices,
                    std::unique_ptr<rmm::device_uvector<size_type>> larger_indices,
                    bool is_left_smaller,
                    null_equality compare_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  if (compare_nulls == null_equality::EQUAL) {
    if (is_left_smaller) { return {std::move(smaller_indices), std::move(larger_indices)}; }
    return {std::move(larger_indices), std::move(smaller_indices)};
  }
  auto get_mapping = [stream, mr](table_view const& tbl) {
    // auto [tbl_result_mask, tbl_num_nulls] = detail::bitmask_and(tbl, stream, mr);
    auto [tbl_result_mask, tbl_num_nulls] = [=]() {
      // remove rows that have nulls at any nesting level
      // step 1: identify nulls at root level
      auto [validity_mask, num_nulls] = cudf::bitmask_and(tbl, stream, mr);
      // step 2: identify nulls at non-root levels
      for (size_type col_idx = 0; col_idx < tbl.num_columns(); col_idx++) {
        auto col = tbl.column(col_idx);
        if (col.type().id() == type_id::LIST) {
          auto lcv     = lists_column_view(col);
          auto offsets = lcv.offsets();
          auto child   = lcv.child();
          rmm::device_uvector<int32_t> offsets_subset(offsets.size(), stream, mr);
          rmm::device_uvector<int32_t> child_positions(offsets.size(), stream, mr);
          auto unique_end = thrust::unique_by_key_copy(
            rmm::exec_policy(stream),
            thrust::make_reverse_iterator(lcv.offsets_end()),
            thrust::make_reverse_iterator(lcv.offsets_end()) + offsets.size(),
            thrust::make_reverse_iterator(thrust::make_counting_iterator(offsets.size())),
            thrust::make_reverse_iterator(offsets_subset.end()),
            thrust::make_reverse_iterator(child_positions.end()));
          auto subset_size   = thrust::distance(thrust::make_reverse_iterator(offsets_subset.end()),
                                              thrust::get<0>(unique_end));
          auto subset_offset = offsets.size() - subset_size;

          auto [reduced_validity_mask, num_nulls] =
            detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                                  offsets_subset.data() + subset_offset,
                                                  offsets_subset.data() + offsets_subset.size() - 1,
                                                  offsets_subset.data() + subset_offset + 1,
                                                  null_policy::INCLUDE,
                                                  std::nullopt,
                                                  stream,
                                                  mr);

          thrust::for_each(
            rmm::exec_policy(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + subset_size,
            [validity_mask         = static_cast<bitmask_type*>(validity_mask.data()),
             reduced_validity_mask = static_cast<bitmask_type*>(reduced_validity_mask.data()),
             child_positions       = child_positions.begin() + subset_offset] __device__(auto idx) {
              if (!bit_is_set(reduced_validity_mask, idx))
                clear_bit(validity_mask, child_positions[idx]);
            });
        }
      }
      num_nulls =
        null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, tbl.num_rows(), stream);
      return std::pair{std::move(validity_mask), num_nulls};
    }();
    rmm::device_uvector<size_type> tbl_mapping(tbl.num_rows() - tbl_num_nulls, stream, mr);
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(tbl.num_rows()),
                    tbl_mapping.begin(),
                    [mask = static_cast<bitmask_type*>(tbl_result_mask.data())] __device__(
                      size_type idx) { return cudf::bit_is_set(mask, idx); });
    return tbl_mapping;
  };

  auto left_mapping  = get_mapping(left);
  auto right_mapping = get_mapping(right);

  if (is_left_smaller) {
    thrust::transform(
      rmm::exec_policy(stream),
      smaller_indices->begin(),
      smaller_indices->end(),
      smaller_indices->begin(),
      [left_mapping = left_mapping.begin()] __device__(auto idx) { return left_mapping[idx]; });
    thrust::transform(
      rmm::exec_policy(stream),
      larger_indices->begin(),
      larger_indices->end(),
      larger_indices->begin(),
      [right_mapping = right_mapping.begin()] __device__(auto idx) { return right_mapping[idx]; });
    return {std::move(smaller_indices), std::move(larger_indices)};
  }
  thrust::transform(
    rmm::exec_policy(stream),
    smaller_indices->begin(),
    smaller_indices->end(),
    smaller_indices->begin(),
    [right_mapping = right_mapping.begin()] __device__(auto idx) { return right_mapping[idx]; });
  thrust::transform(
    rmm::exec_policy(stream),
    larger_indices->begin(),
    larger_indices->end(),
    larger_indices->begin(),
    [left_mapping = left_mapping.begin()] __device__(auto idx) { return left_mapping[idx]; });
  return {std::move(larger_indices), std::move(smaller_indices)};
}

}  // anonymous namespace

void sort_merge_join::preprocessed_table::populate_nonnull_filter(
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  auto tbl = this->raw_tbl_view;
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(tbl, stream, mr);
  // step 2: identify nulls at non-root levels
  for (size_type col_idx = 0; col_idx < tbl.num_columns(); col_idx++) {
    auto col = tbl.column(col_idx);
    if (col.type().id() == type_id::LIST) {
      auto lcv     = lists_column_view(col);
      auto offsets = lcv.offsets();
      auto child   = lcv.child();

      rmm::device_uvector<int32_t> offsets_subset(offsets.size(), stream, mr);
      rmm::device_uvector<int32_t> child_positions(offsets.size(), stream, mr);
      auto unique_end = thrust::unique_by_key_copy(
        rmm::exec_policy(stream),
        thrust::make_reverse_iterator(lcv.offsets_end()),
        thrust::make_reverse_iterator(lcv.offsets_end()) + offsets.size(),
        thrust::make_reverse_iterator(thrust::make_counting_iterator(offsets.size())),
        thrust::make_reverse_iterator(offsets_subset.end()),
        thrust::make_reverse_iterator(child_positions.end()));
      auto subset_size   = thrust::distance(thrust::make_reverse_iterator(offsets_subset.end()),
                                          thrust::get<0>(unique_end));
      auto subset_offset = offsets.size() - subset_size;

      auto [reduced_validity_mask, num_nulls] =
        detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                              offsets_subset.data() + subset_offset,
                                              offsets_subset.data() + offsets_subset.size() - 1,
                                              offsets_subset.data() + subset_offset + 1,
                                              null_policy::INCLUDE,
                                              std::nullopt,
                                              stream,
                                              mr);

      thrust::for_each(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + subset_size,
        [validity_mask         = static_cast<bitmask_type*>(validity_mask.data()),
         reduced_validity_mask = static_cast<bitmask_type*>(reduced_validity_mask.data()),
         child_positions       = child_positions.begin() + subset_offset] __device__(auto idx) {
          if (!bit_is_set(reduced_validity_mask, idx))
            clear_bit(validity_mask, child_positions[idx]);
        });
    }
  }
  this->raw_num_nulls =
    null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, tbl.num_rows(), stream);
  this->raw_validity_mask = std::move(validity_mask);
}

void sort_merge_join::preprocessed_table::apply_nonnull_filter(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  // construct bool column to apply mask
  cudf::scalar_type_t<bool> true_scalar(true);
  auto bool_mask = cudf::make_column_from_scalar(true_scalar, raw_tbl_view.num_rows(), stream, mr);
  CUDF_EXPECTS(raw_validity_mask.has_value() && raw_num_nulls.has_value(), "Something went wrong while dropping nulls in the raw tables");
  bool_mask->set_null_mask(raw_validity_mask.value(), raw_num_nulls.value(), stream);

  tbl = apply_boolean_mask(raw_tbl_view, *bool_mask, stream, mr);
  tbl_view = tbl.value()->view();
}

void sort_merge_join::preprocessed_table::preprocess_raw_table(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  populate_nonnull_filter(stream, mr);
  apply_nonnull_filter(stream, mr);
}

void sort_merge_join::preprocess_tables(table_view const left,
                  table_view const right,
                  null_equality compare_nulls,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  if (compare_nulls == null_equality::EQUAL) {
    ptleft.tbl_view = left;
    ptright.tbl_view = right;
  }

  ptleft.preprocess_raw_table(stream, mr);
  ptright.preprocess_raw_table(stream, mr);
}

void sort_merge_join::preprocessed_table::get_sorted_order(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<cudf::order> column_order(tbl_view.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(tbl_view.num_columns(), cudf::null_order::BEFORE);
  this->tbl_sorted_order = cudf::sorted_order(tbl_view, column_order, null_precedence, stream, mr);
}


sort_merge_join::sort_merge_join(table_view const &left, bool is_left_sorted, table_view const &right, bool is_right_sorted,
                  null_equality compare_nulls,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr) {

  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(),
               "Number of columns must match for a join");

  ptleft.raw_tbl_view = left;
  ptright.raw_tbl_view = right;
  this->compare_nulls = compare_nulls;
  preprocess_tables(left, right, compare_nulls, stream, mr);

  if(!is_left_sorted) {
    ptleft.get_sorted_order(stream, mr); 
  }
  if(!is_right_sorted) {
    ptright.get_sorted_order(stream, mr); 
  }
}

rmm::device_uvector<size_type> sort_merge_join::preprocessed_table::map_tbl_to_raw(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(raw_validity_mask.has_value() && raw_num_nulls.has_value(), "Mapping is not possible");
  rmm::device_uvector<size_type> tbl_mapping(raw_tbl_view.num_rows() - raw_num_nulls.value(), stream, mr);
  thrust::copy_if(rmm::exec_policy(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(raw_tbl_view.num_rows()),
                  tbl_mapping.begin(),
                  [mask = static_cast<bitmask_type*>(raw_validity_mask.value().data())] __device__(
                    size_type idx) { return cudf::bit_is_set(mask, idx); });
  return tbl_mapping;
}

struct mapping_functor {
  device_span<size_type> mapping;
  __device__ size_type operator()(size_type idx)
  {
    return mapping[idx];
  }
};

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::postprocess_indices(
                    std::unique_ptr<rmm::device_uvector<size_type>> smaller_indices,
                    std::unique_ptr<rmm::device_uvector<size_type>> larger_indices,
                    null_equality compare_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  bool is_left_smaller           = ptleft.tbl_view.num_rows() < ptright.tbl_view.num_rows();
  if (compare_nulls == null_equality::EQUAL) {
    if (is_left_smaller) { return {std::move(smaller_indices), std::move(larger_indices)}; }
    return {std::move(larger_indices), std::move(smaller_indices)};
  }

  auto left_mapping  = ptleft.map_tbl_to_raw(stream, mr);
  auto right_mapping = ptright.map_tbl_to_raw(stream, mr);

  if (is_left_smaller) {
    thrust::transform(
      rmm::exec_policy(stream),
      smaller_indices->begin(),
      smaller_indices->end(),
      smaller_indices->begin(),
      mapping_functor{left_mapping});
    thrust::transform(
      rmm::exec_policy(stream),
      larger_indices->begin(),
      larger_indices->end(),
      larger_indices->begin(),
      mapping_functor{right_mapping});
    return {std::move(smaller_indices), std::move(larger_indices)};
  }
  thrust::transform(
    rmm::exec_policy(stream),
    smaller_indices->begin(),
    smaller_indices->end(),
    smaller_indices->begin(),
    mapping_functor{right_mapping});
  thrust::transform(
    rmm::exec_policy(stream),
    larger_indices->begin(),
    larger_indices->end(),
    larger_indices->begin(),
    mapping_functor{left_mapping});
  return {std::move(larger_indices), std::move(smaller_indices)};
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {

  // we passed pre-sorted tables
  if(!ptleft.tbl_sorted_order.has_value()) {
    cudf::merge obj(ptleft.tbl_view, thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + ptleft.tbl_view.num_rows(), ptright.tbl_view, thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + ptright.tbl_view.num_rows());
    auto [smaller_indices, larger_indices] = obj(stream, mr);
    return postprocess_indices(std::move(smaller_indices), std::move(larger_indices), compare_nulls, stream, mr);
  }
  cudf::merge obj(ptleft.tbl_view, ptleft.tbl_sorted_order.value()->view().begin<size_type>(), ptleft.tbl_sorted_order.value()->view().end<size_type>(), ptright.tbl_view, ptright.tbl_sorted_order.value()->view().begin<size_type>(), ptright.tbl_sorted_order.value()->view().end<size_type>());
  auto [smaller_indices, larger_indices] = obj(stream, mr);
  return postprocess_indices(std::move(smaller_indices), std::move(larger_indices), compare_nulls, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(table_view const& left,
                      table_view const& right,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(),
               "Number of columns must match for a join");

  auto [true_left_view, true_right_view, left_table, right_table] =
    preprocess_tables(left, right, compare_nulls, stream, mr);

  auto [sorted_left_order_col, sorted_right_order_col] =
    sort(true_left_view, true_right_view, stream, mr);

  bool is_left_smaller           = true_left_view.num_rows() < true_right_view.num_rows();
  auto& smaller                  = is_left_smaller ? true_left_view : true_right_view;
  auto& sorted_smaller_order_col = is_left_smaller ? sorted_left_order_col : sorted_right_order_col;
  auto& larger                   = !is_left_smaller ? true_left_view : true_right_view;
  auto& sorted_larger_order_col = !is_left_smaller ? sorted_left_order_col : sorted_right_order_col;

  auto [smaller_indices, larger_indices] =
    local_merge(smaller,
          sorted_smaller_order_col->view().begin<size_type>(),
          sorted_smaller_order_col->view().end<size_type>(),
          larger,
          sorted_larger_order_col->view().begin<size_type>(),
          sorted_larger_order_col->view().end<size_type>(),
          compare_nulls,
          stream,
          mr);

  return postprocess_indices(left,
                             right,
                             std::move(smaller_indices),
                             std::move(larger_indices),
                             is_left_smaller,
                             compare_nulls,
                             stream,
                             mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge_inner_join(table_view const& sorted_left,
                 table_view const& sorted_right,
                 null_equality compare_nulls,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Sanity checks
  CUDF_EXPECTS(sorted_left.num_columns() == sorted_right.num_columns(),
               "Number of columns must match for a join");

  auto [true_sorted_left_view, true_sorted_right_view, left_table, right_table] =
    preprocess_tables(sorted_left, sorted_right, compare_nulls, stream, mr);

  bool is_sorted_left_smaller =
    true_sorted_left_view.num_rows() < true_sorted_right_view.num_rows();
  auto& smaller = is_sorted_left_smaller ? sorted_left : sorted_right;
  auto& larger  = !is_sorted_left_smaller ? sorted_left : sorted_right;

  auto [smaller_indices, larger_indices] =
    local_merge(smaller,
          thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(0) + smaller.num_rows(),
          larger,
          thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(0) + larger.num_rows(),
          compare_nulls,
          stream,
          mr);

  return postprocess_indices(sorted_left,
                             sorted_right,
                             std::move(smaller_indices),
                             std::move(larger_indices),
                             is_sorted_left_smaller,
                             compare_nulls,
                             stream,
                             mr);
}

}  // namespace cudf
