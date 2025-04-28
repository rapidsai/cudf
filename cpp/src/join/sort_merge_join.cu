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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <utility>

namespace cudf {

namespace {

/*
struct mapping_functor {
  device_span<size_type> mapping;
  __device__ size_type operator()(size_type idx) const noexcept { return mapping[idx]; }
};
*/

template <typename T>
struct mapping_functor {
  T mapping;
  __device__ size_type operator()(size_type idx) const noexcept { return mapping[idx]; }
};

struct list_nonnull_filter {
  bitmask_type* const validity_mask;
  bitmask_type const* const reduced_validity_mask;
  device_span<size_type const> child_positions;
  size_type const subset_offset;
  __device__ void operator()(size_type idx) const noexcept
  {
    if (!bit_is_set(reduced_validity_mask, idx))
      clear_bit(validity_mask, child_positions[idx + subset_offset]);
  };
};

struct raw_tbl_mapper {
  bitmask_type const* const raw_validity_mask;
  __device__ auto operator()(size_type idx) const noexcept
  {
    return cudf::bit_is_set(raw_validity_mask, idx);
  }
};

template <typename LargerIterator, typename SmallerIterator>
class merge {
 private:
  table_view smaller;
  table_view larger;
  SmallerIterator sorted_smaller_order_begin;
  SmallerIterator sorted_smaller_order_end;
  LargerIterator sorted_larger_order_begin;
  LargerIterator sorted_larger_order_end;

 public:
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

  merge(table_view const& smaller,
        SmallerIterator sorted_smaller_order_begin,
        SmallerIterator sorted_smaller_order_end,
        table_view const& larger,
        LargerIterator sorted_larger_order_begin,
        LargerIterator sorted_larger_order_end)
    : smaller{smaller},
      sorted_smaller_order_begin{sorted_smaller_order_begin},
      sorted_smaller_order_end{sorted_smaller_order_end},
      larger{larger},
      sorted_larger_order_begin{sorted_larger_order_begin},
      sorted_larger_order_end{sorted_larger_order_end}
  {
  }

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  operator()(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};

template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::operator()(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto temp_mr             = cudf::get_current_device_resource_ref();
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
    return std::pair(std::move(dremel_data), std::move(d_dremel_device_views));
  };
  auto [smaller_dremel, smaller_dremel_dv] = list_lex_preprocess(smaller);
  auto [larger_dremel, larger_dremel_dv]   = list_lex_preprocess(larger);

  // naive: iterate through larger table and binary search on smaller table
  auto const larger_numrows = larger.num_rows();
  rmm::device_scalar<bound_type> d_lb_type(bound_type::LOWER, stream, temp_mr);
  rmm::device_scalar<bound_type> d_ub_type(bound_type::UPPER, stream, temp_mr);

  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(larger_numrows + 1, stream, temp_mr);

  row_comparator comp(
    *larger_dv_ptr, *smaller_dv_ptr, larger_dremel_dv, smaller_dremel_dv, d_ub_type.data());
  auto match_counts_it = match_counts.begin();
  thrust::upper_bound(rmm::exec_policy_nosync(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::counting_iterator(0),
                      thrust::counting_iterator(0) + larger_numrows,
                      match_counts_it,
                      comp);

  comp._d_ptr = d_lb_type.data();
  auto match_counts_update_it =
    thrust::tabulate_output_iterator([match_counts = match_counts.begin()] __device__(
                                       size_type idx, size_type val) { match_counts[idx] -= val; });
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      thrust::counting_iterator(0),
                      thrust::counting_iterator(0) + larger_numrows,
                      match_counts_update_it,
                      comp);

  auto count_matches_it = thrust::transform_iterator(
    match_counts.begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) { return c ? 1 : 0; }));
  auto const count_matches =
    thrust::reduce(rmm::exec_policy(stream), count_matches_it, count_matches_it + larger_numrows);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, temp_mr);
  thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator(0),
    thrust::counting_iterator(0) + larger_numrows,
    nonzero_matches.begin(),
    [match_counts = match_counts.begin()] __device__(auto idx) { return match_counts[idx]; });

  thrust::exclusive_scan(
    rmm::exec_policy(stream), match_counts.begin(), match_counts.end(), match_counts.begin());
  auto const total_matches = match_counts.back_element(stream);

  // populate larger indices
  auto larger_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);
  thrust::scatter(rmm::exec_policy_nosync(stream),
                  nonzero_matches.begin(),
                  nonzero_matches.end(),
                  thrust::permutation_iterator(match_counts.begin(), nonzero_matches.begin()),
                  larger_indices.begin());
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                         larger_indices.begin(),
                         larger_indices.end(),
                         larger_indices.begin(),
                         thrust::maximum<size_type>{});

  // populate smaller indices
  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream), smaller_indices.begin(), smaller_indices.end(), 1);
  auto smaller_tabulate_it = thrust::tabulate_output_iterator(
    [nonzero_matches = nonzero_matches.begin(),
     match_counts    = match_counts.begin(),
     smaller_indices = smaller_indices.begin()] __device__(auto idx, auto lb) {
      auto lhsidx          = nonzero_matches[idx];
      auto pos             = match_counts[lhsidx];
      smaller_indices[pos] = lb;
    });
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      smaller_tabulate_it,
                      comp);
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                larger_indices.begin(),
                                larger_indices.end(),
                                smaller_indices.begin(),
                                smaller_indices.begin());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    smaller_indices.begin(),
                    smaller_indices.end(),
                    smaller_indices.begin(),
                    [sorted_smaller_order = sorted_smaller_order_begin] __device__(auto idx) {
                      return sorted_smaller_order[idx];
                    });
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    smaller_indices.begin(),
    smaller_indices.end(),
    smaller_indices.begin(),
    mapping_functor<thrust::counting_iterator<size_type>>{sorted_smaller_order_begin});

  stream.synchronize();
  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

}  // anonymous namespace

void sort_merge_join::preprocessed_table::populate_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto tbl     = this->raw_tbl_view;
  auto temp_mr = cudf::get_current_device_resource_ref();
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(tbl, stream, temp_mr);
  // step 2: identify nulls at non-root levels
  for (size_type col_idx = 0; col_idx < tbl.num_columns(); col_idx++) {
    auto col = tbl.column(col_idx);
    if (col.type().id() == type_id::LIST) {
      auto lcv     = lists_column_view(col);
      auto offsets = lcv.offsets();
      auto child   = lcv.child();

      rmm::device_uvector<int32_t> offsets_subset(offsets.size(), stream, temp_mr);
      rmm::device_uvector<int32_t> child_positions(offsets.size(), stream, temp_mr);
      auto unique_end = thrust::unique_by_key_copy(
        rmm::exec_policy(stream),
        thrust::reverse_iterator(lcv.offsets_end()),
        thrust::reverse_iterator(lcv.offsets_end()) + offsets.size(),
        thrust::reverse_iterator(thrust::counting_iterator(offsets.size())),
        thrust::reverse_iterator(offsets_subset.end()),
        thrust::reverse_iterator(child_positions.end()));
      auto subset_size   = cuda::std::distance(thrust::reverse_iterator(offsets_subset.end()),
                                             cuda::std::get<0>(unique_end));
      auto subset_offset = offsets.size() - subset_size;

      auto [reduced_validity_mask, num_nulls] =
        detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                              offsets_subset.data() + subset_offset,
                                              offsets_subset.data() + offsets_subset.size() - 1,
                                              offsets_subset.data() + subset_offset + 1,
                                              null_policy::INCLUDE,
                                              std::nullopt,
                                              stream,
                                              temp_mr);

      thrust::for_each(
        rmm::exec_policy_nosync(stream),
        thrust::counting_iterator(0),
        thrust::counting_iterator(0) + subset_size,
        list_nonnull_filter{static_cast<bitmask_type*>(validity_mask.data()),
                            static_cast<bitmask_type const*>(reduced_validity_mask.data()),
                            child_positions,
                            static_cast<size_type>(subset_offset)});
    }
  }
  this->raw_num_nulls =
    null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, tbl.num_rows(), stream);
  this->raw_validity_mask = std::move(validity_mask);
}

void sort_merge_join::preprocessed_table::apply_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  // construct bool column to apply mask
  cudf::scalar_type_t<bool> true_scalar(true, true, stream, temp_mr);
  auto bool_mask =
    cudf::make_column_from_scalar(true_scalar, raw_tbl_view.num_rows(), stream, temp_mr);
  CUDF_EXPECTS(raw_validity_mask.has_value() && raw_num_nulls.has_value(),
               "Something went wrong while dropping nulls in the raw tables");
  bool_mask->set_null_mask(raw_validity_mask.value(), raw_num_nulls.value(), stream);

  tbl      = apply_boolean_mask(raw_tbl_view, *bool_mask, stream, temp_mr);
  tbl_view = tbl.value()->view();
}

void sort_merge_join::preprocessed_table::preprocess_raw_table(rmm::cuda_stream_view stream)
{
  populate_nonnull_filter(stream);
  apply_nonnull_filter(stream);
}

void sort_merge_join::preprocess_tables(table_view const left,
                                        table_view const right,
                                        rmm::cuda_stream_view stream)
{
  // if a table has no nullable column, then there's no preprocessing to be done
  auto is_nullable_table = [](table_view const& t) {
    for (auto&& col : t) {
      if (col.nullable()) { return true; }
    }
    return false;
  };

  if (compare_nulls == null_equality::EQUAL) {
    preprocessed_left.tbl_view  = left;
    preprocessed_right.tbl_view = right;
  } else {
    auto is_left_nullable  = is_nullable_table(left);
    auto is_right_nullable = is_nullable_table(right);
    if (is_left_nullable) {
      preprocessed_left.preprocess_raw_table(stream);
    } else {
      preprocessed_left.tbl_view = left;
    }
    if (is_right_nullable) {
      preprocessed_right.preprocess_raw_table(stream);
    } else {
      preprocessed_right.tbl_view = right;
    }
  }
}

void sort_merge_join::preprocessed_table::get_sorted_order(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::order> column_order(tbl_view.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(tbl_view.num_columns(), cudf::null_order::BEFORE);
  this->tbl_sorted_order =
    cudf::sorted_order(tbl_view, column_order, null_precedence, stream, temp_mr);
}

sort_merge_join::sort_merge_join(table_view const& left,
                                 sorted is_left_sorted,
                                 table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
{
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(),
               "Number of columns must match for a join");

  preprocessed_left.raw_tbl_view  = left;
  preprocessed_right.raw_tbl_view = right;
  this->compare_nulls             = compare_nulls;
  preprocess_tables(left, right, stream);

  if (is_left_sorted == cudf::sorted::NO) { preprocessed_left.get_sorted_order(stream); }
  if (is_right_sorted == cudf::sorted::NO) { preprocessed_right.get_sorted_order(stream); }
}

rmm::device_uvector<size_type> sort_merge_join::preprocessed_table::map_tbl_to_raw(
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(raw_validity_mask.has_value() && raw_num_nulls.has_value(),
               "Mapping is not possible");
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<size_type> tbl_mapping(
    raw_tbl_view.num_rows() - raw_num_nulls.value(), stream, temp_mr);
  thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(raw_tbl_view.num_rows()),
    tbl_mapping.begin(),
    raw_tbl_mapper{static_cast<bitmask_type const*>(raw_validity_mask.value().data())});
  return tbl_mapping;
}

void sort_merge_join::postprocess_indices(device_span<size_type> smaller_indices,
                                          device_span<size_type> larger_indices,
                                          rmm::cuda_stream_view stream)
{
  bool is_left_smaller =
    preprocessed_left.tbl_view.num_rows() < preprocessed_right.tbl_view.num_rows();
  // if a table has no nullable column, then there's no postprocessing to be done
  auto is_nullable_table = [](table_view const& t) {
    for (auto&& col : t) {
      if (col.nullable()) { return true; }
    }
    return false;
  };

  if (compare_nulls == null_equality::UNEQUAL) {
    auto is_left_nullable  = is_nullable_table(preprocessed_left.tbl_view);
    auto is_right_nullable = is_nullable_table(preprocessed_right.tbl_view);
    if (is_left_nullable) {
      auto left_mapping = preprocessed_left.map_tbl_to_raw(stream);
      if (is_left_smaller) {
        thrust::transform(rmm::exec_policy_nosync(stream),
                          smaller_indices.begin(),
                          smaller_indices.end(),
                          smaller_indices.begin(),
                          mapping_functor<device_span<size_type>>{left_mapping});
      } else {
        thrust::transform(rmm::exec_policy_nosync(stream),
                          larger_indices.begin(),
                          larger_indices.end(),
                          larger_indices.begin(),
                          mapping_functor<device_span<size_type>>{left_mapping});
      }
    }
    if (is_right_nullable) {
      auto right_mapping = preprocessed_right.map_tbl_to_raw(stream);
      if (is_left_smaller) {
        thrust::transform(rmm::exec_policy_nosync(stream),
                          larger_indices.begin(),
                          larger_indices.end(),
                          larger_indices.begin(),
                          mapping_functor<device_span<size_type>>{right_mapping});
      } else {
        thrust::transform(rmm::exec_policy_nosync(stream),
                          smaller_indices.begin(),
                          smaller_indices.end(),
                          smaller_indices.begin(),
                          mapping_functor<device_span<size_type>>{right_mapping});
      }
    }
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // TODO: what if one is sorted but not the other?
  bool is_left_smaller =
    preprocessed_left.tbl_view.num_rows() < preprocessed_right.tbl_view.num_rows();
  auto& smaller = is_left_smaller ? preprocessed_left : preprocessed_right;
  auto& larger  = is_left_smaller ? preprocessed_right : preprocessed_left;
  if (smaller.tbl_sorted_order.has_value() && larger.tbl_sorted_order.has_value()) {
    merge obj(smaller.tbl_view,
              smaller.tbl_sorted_order.value()->view().begin<size_type>(),
              smaller.tbl_sorted_order.value()->view().end<size_type>(),
              larger.tbl_view,
              larger.tbl_sorted_order.value()->view().begin<size_type>(),
              larger.tbl_sorted_order.value()->view().end<size_type>());
    auto [smaller_indices, larger_indices] = obj(stream, mr);
    postprocess_indices(*smaller_indices, *larger_indices, stream);
    stream.synchronize();
    if (is_left_smaller) { return {std::move(smaller_indices), std::move(larger_indices)}; }
    return {std::move(larger_indices), std::move(smaller_indices)};
  }
  // we passed pre-sorted tables
  merge obj(smaller.tbl_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(smaller.tbl_view.num_rows()),
            larger.tbl_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(larger.tbl_view.num_rows()));
  auto [smaller_indices, larger_indices] = obj(stream, mr);
  postprocess_indices(*smaller_indices, *larger_indices, stream);
  stream.synchronize();
  if (is_left_smaller) { return {std::move(smaller_indices), std::move(larger_indices)}; }
  return {std::move(larger_indices), std::move(smaller_indices)};
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  cudf::sort_merge_join obj(left_keys, sorted::NO, right_keys, sorted::NO, compare_nulls, stream);
  return obj.inner_join(stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge_inner_join(cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 null_equality compare_nulls,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  cudf::sort_merge_join obj(left_keys, sorted::YES, right_keys, sorted::YES, compare_nulls, stream);
  return obj.inner_join(stream, mr);
}

}  // namespace cudf
