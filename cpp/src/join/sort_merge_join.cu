/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/unique.h>

#include <utility>

namespace cudf {

namespace {

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
    if (!bit_is_set(reduced_validity_mask, idx)) {
      clear_bit(validity_mask, child_positions[idx + subset_offset]);
    }
  };
};

struct unprocessed_table_mapper {
  bitmask_type const* const _validity_mask;
  __device__ auto operator()(size_type idx) const noexcept
  {
    return cudf::bit_is_set(_validity_mask, idx);
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
    row_comparator(table_device_view const& lhs,
                   table_device_view const& rhs,
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
        return ub_comparator(lhs_index, rhs_index) == cudf::detail::weak_ordering::LESS;
      }
      return lb_comparator(lhs_index, rhs_index) == cudf::detail::weak_ordering::LESS;
    }

    bound_type* _d_ptr;

   private:
    table_device_view _lhs;
    table_device_view _rhs;
    device_span<detail::dremel_device_view const> _lhs_dremel;
    device_span<detail::dremel_device_view const> _rhs_dremel;
    cudf::detail::row::lexicographic::device_row_comparator<true, bool> ub_comparator;
    cudf::detail::row::lexicographic::device_row_comparator<true, bool> lb_comparator;
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

  std::unique_ptr<rmm::device_uvector<size_type>> matches_per_row(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  operator()(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};

template <typename LargerIterator, typename SmallerIterator>
std::unique_ptr<rmm::device_uvector<size_type>>
merge<LargerIterator, SmallerIterator>::matches_per_row(rmm::cuda_stream_view stream,
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
  cudf::detail::device_scalar<bound_type> d_lb_type(bound_type::LOWER, stream, temp_mr);
  cudf::detail::device_scalar<bound_type> d_ub_type(bound_type::UPPER, stream, temp_mr);

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

  stream.synchronize();

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(match_counts));
}

template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::operator()(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto temp_mr              = cudf::get_current_device_resource_ref();
  auto const larger_numrows = larger.num_rows();
  auto smaller_dv_ptr       = cudf::table_device_view::create(smaller, stream);
  auto larger_dv_ptr        = cudf::table_device_view::create(larger, stream);
  auto list_lex_preprocess  = [stream](table_view const& table) {
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

  rmm::device_scalar<bound_type> d_lb_type(bound_type::LOWER, stream, temp_mr);
  row_comparator lb_comp(
    *larger_dv_ptr, *smaller_dv_ptr, larger_dremel_dv, smaller_dremel_dv, d_lb_type.data());

  // naive: iterate through larger table and binary search on smaller table
  auto match_counts = matches_per_row(stream, temp_mr);

  auto count_matches_it = thrust::transform_iterator(
    match_counts->begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) -> size_type { return c != 0; }));
  auto const count_matches =
    thrust::reduce(rmm::exec_policy(stream), count_matches_it, count_matches_it + larger_numrows);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, temp_mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator(0),
                  thrust::counting_iterator(0) + larger_numrows,
                  match_counts->begin(),
                  nonzero_matches.begin(),
                  cuda::std::identity{});

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         match_counts->begin(),
                         match_counts->end(),
                         match_counts->begin());
  auto const total_matches = match_counts->back_element(stream);

  // populate larger indices
  auto larger_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);
  thrust::scatter(rmm::exec_policy_nosync(stream),
                  nonzero_matches.begin(),
                  nonzero_matches.end(),
                  thrust::permutation_iterator(match_counts->begin(), nonzero_matches.begin()),
                  larger_indices.begin());
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                         larger_indices.begin(),
                         larger_indices.end(),
                         larger_indices.begin(),
                         thrust::maximum<size_type>{});

  // populate smaller indices
  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), smaller_indices.begin(), smaller_indices.end(), 1);
  auto smaller_tabulate_it = thrust::tabulate_output_iterator(
    [nonzero_matches = nonzero_matches.begin(),
     match_counts    = match_counts->begin(),
     smaller_indices = smaller_indices.begin()] __device__(auto idx, auto lb) {
      auto const lhs_idx   = nonzero_matches[idx];
      auto const pos       = match_counts[lhs_idx];
      smaller_indices[pos] = lb;
    });
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      sorted_smaller_order_begin,
                      sorted_smaller_order_end,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      smaller_tabulate_it,
                      lb_comp);
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                larger_indices.begin(),
                                larger_indices.end(),
                                smaller_indices.begin(),
                                smaller_indices.begin());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    smaller_indices.begin(),
                    smaller_indices.end(),
                    smaller_indices.begin(),
                    mapping_functor<SmallerIterator>{sorted_smaller_order_begin});

  stream.synchronize();

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

}  // anonymous namespace

void sort_merge_join::preprocessed_table::populate_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto table   = this->_table_view;
  auto temp_mr = cudf::get_current_device_resource_ref();
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(table, stream, temp_mr);
  // step 2: identify nulls at non-root levels
  for (size_type col_idx = 0; col_idx < table.num_columns(); col_idx++) {
    auto col = table.column(col_idx);
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
  this->_num_nulls =
    null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, table.num_rows(), stream);
  this->_validity_mask = std::move(validity_mask);
}

void sort_merge_join::preprocessed_table::apply_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  // construct bool column to apply mask
  cudf::scalar_type_t<bool> true_scalar(true, true, stream, temp_mr);
  auto bool_mask =
    cudf::make_column_from_scalar(true_scalar, _table_view.num_rows(), stream, temp_mr);
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(),
               "Something went wrong while dropping nulls in the unprocessed tables");
  bool_mask->set_null_mask(_validity_mask.value(), _num_nulls.value(), stream);

  _null_processed_table      = apply_boolean_mask(_table_view, *bool_mask, stream, temp_mr);
  _null_processed_table_view = _null_processed_table.value()->view();
}

void sort_merge_join::preprocessed_table::preprocess_unprocessed_table(rmm::cuda_stream_view stream)
{
  populate_nonnull_filter(stream);
  apply_nonnull_filter(stream);
}

void sort_merge_join::preprocessed_table::get_sorted_order(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::order> column_order(_null_processed_table_view.num_columns(),
                                        cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(_null_processed_table_view.num_columns(),
                                                cudf::null_order::BEFORE);
  this->_null_processed_table_sorted_order =
    cudf::sorted_order(_null_processed_table_view, column_order, null_precedence, stream, temp_mr);
}

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
{
  cudf::scoped_range range{"sort_merge_join::sort_merge_join"};
  // Sanity checks
  CUDF_EXPECTS(right.num_columns() != 0,
               "Number of columns the keys table must be non-zero for a join",
               std::invalid_argument);

  this->compare_nulls = compare_nulls;

  // Preprocessing the right table
  preprocessed_right._table_view = right;
  if (compare_nulls == null_equality::EQUAL) {
    preprocessed_right._null_processed_table_view = right;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    auto is_right_nullable = has_nested_nulls(right);
    if (is_right_nullable) {
      preprocessed_right.preprocess_unprocessed_table(stream);
    } else {
      preprocessed_right._null_processed_table_view = right;
    }
  }
  if (is_right_sorted == cudf::sorted::NO) { preprocessed_right.get_sorted_order(stream); }
}

rmm::device_uvector<size_type> sort_merge_join::preprocessed_table::map_table_to_unprocessed(
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(), "Mapping is not possible");
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<size_type> table_mapping(
    _table_view.num_rows() - _num_nulls.value(), stream, temp_mr);
  thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(_table_view.num_rows()),
    table_mapping.begin(),
    unprocessed_table_mapper{static_cast<bitmask_type const*>(_validity_mask.value().data())});
  return table_mapping;
}

void sort_merge_join::postprocess_indices(device_span<size_type> smaller_indices,
                                          device_span<size_type> larger_indices,
                                          rmm::cuda_stream_view stream)
{
  if (compare_nulls == null_equality::UNEQUAL) {
    // if a table has no nullable column, then there's no postprocessing to be done
    auto is_left_nullable  = has_nested_nulls(preprocessed_left._table_view);
    auto is_right_nullable = has_nested_nulls(preprocessed_right._table_view);
    if (is_left_nullable) {
      auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        larger_indices.begin(),
                        larger_indices.end(),
                        larger_indices.begin(),
                        mapping_functor<device_span<size_type>>{left_mapping});
    }
    if (is_right_nullable) {
      auto right_mapping = preprocessed_right.map_table_to_unprocessed(stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        smaller_indices.begin(),
                        smaller_indices.end(),
                        smaller_indices.begin(),
                        mapping_functor<device_span<size_type>>{right_mapping});
    }
  }
}

template <typename MergeOperation>
auto sort_merge_join::invoke_merge(table_view right_view, table_view left_view, MergeOperation&& op)
{
  auto has_right_sorting_order = preprocessed_right._null_processed_table_sorted_order.has_value();
  auto has_left_sorting_order  = preprocessed_left._null_processed_table_sorted_order.has_value();
  if (has_right_sorting_order && has_left_sorting_order) {
    // Both sorted
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>());
    return op(obj);
  } else if (has_right_sorting_order && !has_left_sorting_order) {
    // preprocessed_right sorted, preprocessed_left unsorted
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              thrust::counting_iterator(0),
              thrust::counting_iterator(left_view.num_rows()));
    return op(obj);
  } else if (!has_right_sorting_order && has_left_sorting_order) {
    // preprocessed_right sorted, preprocessed_left unsorted
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              thrust::counting_iterator(0),
              thrust::counting_iterator(preprocessed_right._null_processed_table_view.num_rows()),
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>());
    return op(obj);
  }
  // Both unsorted
  merge obj(right_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(preprocessed_right._null_processed_table_view.num_rows()),
            left_view,
            thrust::counting_iterator(0),
            thrust::counting_iterator(left_view.num_rows()));
  return op(obj);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            sorted is_left_sorted,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"sort_merge_join::inner_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Preprocessing the left table
  preprocessed_left._table_view = left;
  if (compare_nulls == null_equality::EQUAL) {
    preprocessed_left._null_processed_table_view = left;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    auto is_left_nullable = has_nested_nulls(left);
    if (is_left_nullable) {
      preprocessed_left.preprocess_unprocessed_table(stream);
    } else {
      preprocessed_left._null_processed_table_view = left;
    }
  }
  if (is_left_sorted == cudf::sorted::NO) { preprocessed_left.get_sorted_order(stream); }

  return invoke_merge(
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj(stream, mr);
      postprocess_indices(*preprocessed_right_indices, *preprocessed_left_indices, stream);
      stream.synchronize();
      return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
    });
}

cudf::join_match_context sort_merge_join::inner_join_match_context(
  table_view const& left,
  sorted is_left_sorted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"sort_merge_join::inner_join_match_context"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Preprocessing the left table
  preprocessed_left._table_view = left;
  if (compare_nulls == null_equality::EQUAL) {
    preprocessed_left._null_processed_table_view = left;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    auto is_left_nullable = has_nested_nulls(left);
    if (is_left_nullable) {
      preprocessed_left.preprocess_unprocessed_table(stream);
    } else {
      preprocessed_left._null_processed_table_view = left;
    }
  }
  if (is_left_sorted == cudf::sorted::NO) { preprocessed_left.get_sorted_order(stream); }

  return invoke_merge(
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, left, stream, mr](auto& obj) {
      auto matches_per_row = obj.matches_per_row(stream, cudf::get_current_device_resource_ref());
      matches_per_row->resize(matches_per_row->size() - 1, stream);
      if (compare_nulls == null_equality::UNEQUAL &&
          has_nested_nulls(preprocessed_left._table_view)) {
        // Now we need to post-process the matches i.e. insert zero counts for all the null
        // positions
        auto unprocessed_matches_per_row =
          cudf::detail::make_zeroed_device_uvector_async<size_type>(
            preprocessed_left._table_view.num_rows(), stream, mr);
        auto mapping = preprocessed_left.map_table_to_unprocessed(stream);
        thrust::scatter(rmm::exec_policy_nosync(stream),
                        matches_per_row->begin(),
                        matches_per_row->end(),
                        mapping.begin(),
                        unprocessed_matches_per_row.begin());
        stream.synchronize();
        return join_match_context{
          left,
          std::make_unique<rmm::device_uvector<size_type>>(std::move(unprocessed_matches_per_row))};
      }
      return join_match_context{left, std::move(matches_per_row)};
    });
}

// left_partition_end exclusive
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"sort_merge_join::partitioned_inner_join"};
  auto const left_partition_start_idx = context.left_start_idx;
  auto const left_partition_end_idx   = context.left_end_idx;
  auto null_processed_table_start_idx = left_partition_start_idx;
  auto null_processed_table_end_idx   = left_partition_end_idx;
  if (compare_nulls == null_equality::UNEQUAL && has_nested_nulls(preprocessed_left._table_view)) {
    auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
    null_processed_table_start_idx =
      cuda::std::distance(left_mapping.begin(),
                          thrust::lower_bound(rmm::exec_policy(stream),
                                              left_mapping.begin(),
                                              left_mapping.end(),
                                              left_partition_start_idx));
    null_processed_table_end_idx =
      cuda::std::distance(left_mapping.begin(),
                          thrust::upper_bound(rmm::exec_policy(stream),
                                              left_mapping.begin(),
                                              left_mapping.end(),
                                              left_partition_end_idx - 1));
  }
  auto null_processed_left_partition =
    cudf::slice(preprocessed_left._null_processed_table_view,
                {null_processed_table_start_idx, null_processed_table_end_idx},
                stream)[0];

  auto [preprocessed_right_indices, preprocessed_left_indices] = invoke_merge(
    preprocessed_right._null_processed_table_view,
    null_processed_left_partition,
    [this, left_partition_start_idx, stream, mr](auto& obj) { return obj(stream, mr); });
  // Map from slice to total null processed table
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->end(),
    preprocessed_left_indices->begin(),
    [left_partition_start_idx] __device__(auto idx) { return left_partition_start_idx + idx; });
  // Map from total null processed table to unprocessed table
  postprocess_indices(*preprocessed_right_indices, *preprocessed_left_indices, stream);
  stream.synchronize();
  return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"sort_merge_inner_join"};
  cudf::sort_merge_join obj(right_keys, sorted::NO, compare_nulls, stream);
  return obj.inner_join(left_keys, sorted::NO, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge_inner_join(cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 null_equality compare_nulls,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"merge_inner_join"};
  cudf::sort_merge_join obj(right_keys, sorted::YES, compare_nulls, stream);
  return obj.inner_join(left_keys, sorted::YES, stream, mr);
}

}  // namespace cudf
