/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
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

#include <optional>

namespace CUDF_EXPORT cudf {

#define SORT_MERGE_JOIN_DEBUG 1

/**
 * @addtogroup column_join
 * @{
 * @file
 */

template <typename Iterator>
class merge {
 private:
  enum class bound_type { UPPER, LOWER };

  struct row_comparator {
    row_comparator(table_device_view const lhs,
                   table_device_view const rhs,
                   device_span<detail::dremel_device_view const> lhs_dremel,
                   device_span<detail::dremel_device_view const> rhs_dremel,
                   bound_type* d_ptr)
      : _d_ptr{d_ptr},
        _lhs{lhs},
        _rhs{rhs},
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

  template <typename T>
  void debug_print(std::string str, host_span<const T> span)
  {
    std::cout << str << " : ";
    for (const auto& element : span) {
      std::cout << element << " ";
    }
    std::cout << std::endl;
  }

  table_view smaller;
  table_view larger;
  Iterator sorted_smaller_order_begin;
  Iterator sorted_smaller_order_end;
  Iterator sorted_larger_order_begin;
  Iterator sorted_larger_order_end;

 public:
  merge(table_view const &left, Iterator sorted_left_order_col_begin, Iterator sorted_left_order_col_end, table_view const &right, Iterator sorted_right_order_col_begin, Iterator sorted_right_order_col_end) {
    bool is_left_smaller           = left.num_rows() < right.num_rows();
    if(is_left_smaller) {
      smaller = left;
      larger = right;
      sorted_smaller_order_begin = sorted_left_order_col_begin;
      sorted_smaller_order_end = sorted_left_order_col_end;
      sorted_larger_order_begin = sorted_right_order_col_begin;
      sorted_larger_order_end = sorted_right_order_col_end;
    }
    else {
      larger = left;
      smaller = right;
      sorted_larger_order_begin = sorted_left_order_col_begin;
      sorted_larger_order_end = sorted_left_order_col_end;
      sorted_smaller_order_begin = sorted_right_order_col_begin;
      sorted_smaller_order_end = sorted_right_order_col_end;
    }
  }

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  operator()(
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr);
};

class sort_merge_join {
 public:
  sort_merge_join(table_view const &left, bool is_left_sorted, table_view const &right, bool is_right_sorted,
                    null_equality compare_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
    inner_join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  struct preprocessed_table {
    table_view raw_tbl_view;
    table_view tbl_view;
    // filters for null_equality::UNEQUAL
    std::optional<rmm::device_buffer> raw_validity_mask = std::nullopt;
    std::optional<size_type> raw_num_nulls = std::nullopt;
    std::optional<std::unique_ptr<table>> tbl = std::nullopt;
    // optional reordering if we are given pre-sorted tables
    std::optional<std::unique_ptr<column>> tbl_sorted_order = std::nullopt;

    void populate_nonnull_filter(rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
    void apply_nonnull_filter(rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
    void preprocess_raw_table(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    void get_sorted_order(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    rmm::device_uvector<size_type> map_tbl_to_raw(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  };
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  postprocess_indices(
                      std::unique_ptr<rmm::device_uvector<size_type>> smaller_indices,
                      std::unique_ptr<rmm::device_uvector<size_type>> larger_indices,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
 private:
  preprocessed_table ptleft;
  preprocessed_table ptright;
  null_equality compare_nulls;

  void preprocess_tables(table_view const left,
                  table_view const right,
                  null_equality compare_nulls,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr);
};

template <typename Iterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<Iterator>::operator()(
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
    auto d_dremel_device_views = detail::make_device_uvector_sync(
      dremel_device_views, stream, cudf::get_current_device_resource_ref());
    return std::make_tuple(std::move(dremel_data), std::move(d_dremel_device_views));
  };
  auto [smaller_dremel, smaller_dremel_dv] = list_lex_preprocess(smaller);
  auto [larger_dremel, larger_dremel_dv]   = list_lex_preprocess(larger);

  // naive: iterate through larger table and binary search on smaller table
  auto const larger_numrows  = larger.num_rows();
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

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
