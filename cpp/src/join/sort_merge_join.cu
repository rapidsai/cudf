/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_merge_join_impl.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <memory>
#include <utility>

namespace cudf {

namespace {

/**
 * @brief Functor to filter and update validity masks for list columns.
 *
 * Propagates null information from a reduced validity mask to specific child positions
 * in the output validity mask for nested list columns.
 */
struct list_nonnull_filter {
  bitmask_type* validity_mask;                   ///< Output validity mask to update
  bitmask_type const* reduced_validity_mask;     ///< Input reduced validity mask
  device_span<size_type const> child_positions;  ///< Positions in the child column
  size_type subset_offset;                       ///< Offset into child_positions

  __device__ void operator()(size_type idx) const noexcept
  {
    if (!bit_is_set(reduced_validity_mask, idx)) {
      clear_bit(validity_mask, child_positions[idx + subset_offset]);
    }
  };
};

/**
 * @brief Functor to check if a row is valid in an unprocessed table.
 *
 * Maps table row indices to boolean values based on the validity mask.
 */
struct is_row_valid {
  bitmask_type const* _validity_mask;  ///< Validity mask for the table

  __device__ auto operator()(size_type idx) const noexcept
  {
    return bit_is_set(_validity_mask, idx);
  }
};

}  // anonymous namespace

namespace detail {

void sort_merge_join::preprocessed_table::populate_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto table   = this->_table_view;
  auto temp_mr = cudf::get_current_device_resource_ref();
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(table, stream, temp_mr);

  // If the table has no nullable top-level columns, then we need to create
  // an all-valid bitmask that is passed to subsequent operations. This bitmask
  // is updated if any of the nested struct/list children columns have nulls.
  if (validity_mask.is_empty())
    validity_mask =
      cudf::create_null_mask(table.num_rows(), mask_state::ALL_VALID, stream, temp_mr);

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
        rmm::exec_policy_nosync(stream),
        cuda::std::reverse_iterator(lcv.offsets_end()),
        cuda::std::reverse_iterator(lcv.offsets_end()) + offsets.size(),
        cuda::std::reverse_iterator(thrust::counting_iterator(offsets.size())),
        cuda::std::reverse_iterator(offsets_subset.end()),
        cuda::std::reverse_iterator(child_positions.end()));
      auto subset_size   = cuda::std::distance(cuda::std::reverse_iterator(offsets_subset.end()),
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
    } else if (col.type().id() == type_id::STRUCT) {
      // Recursive lambda to traverse struct hierarchy and accumulate null information.
      // This lambda ANDs the column's null mask with the accumulated mask in-place and recursively
      // processes all child columns to capture nested nulls.
      auto and_bitmasks = [&](auto&& self, bitmask_type* mask, column_view const& colview) -> void {
        auto const num_rows = colview.size();
        if (colview.type().id() == cudf::type_id::EMPTY) { return; }

        if (colview.nullable()) {
          // AND this column's null mask with the accumulated mask
          auto colmask = colview.null_mask();
          std::vector masks{reinterpret_cast<bitmask_type const*>(colmask),
                            reinterpret_cast<bitmask_type const*>(mask)};
          std::vector<size_type> begin_bits{0, 0};
          cudf::detail::inplace_bitmask_and(
            device_span<bitmask_type>(mask, num_bitmask_words(num_rows)),
            masks,
            begin_bits,
            num_rows,
            stream);
        }

        if (colview.type().id() == cudf::type_id::STRUCT ||
            colview.type().id() == cudf::type_id::LIST) {
          CUDF_EXPECTS(
            std::all_of(colview.child_begin(),
                        colview.child_end(),
                        [&](auto const& child_col) { return num_rows == child_col.size(); }),
            "Child columns must have the same number of rows as the Struct column.");

          // Recursively process child columns to capture nulls at deeper nesting levels.
          for (auto it = colview.child_begin(); it != colview.child_end(); it++) {
            auto& child = *it;
            self(self, mask, child);
          }
        }
      };
      // Process all children of the struct column
      for (auto it = col.child_begin(); it != col.child_end(); it++) {
        auto& child = *it;
        and_bitmasks(and_bitmasks, static_cast<bitmask_type*>(validity_mask.data()), child);
      }
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

void sort_merge_join::preprocessed_table::compute_sorted_order(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::order> column_order(_null_processed_table_view.num_columns(),
                                        cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(_null_processed_table_view.num_columns(),
                                                cudf::null_order::BEFORE);
  this->_null_processed_table_sorted_order =
    cudf::sorted_order(_null_processed_table_view, column_order, null_precedence, stream, temp_mr);
}

sort_merge_join::preprocessed_table sort_merge_join::preprocessed_table::create(
  table_view const& table,
  null_equality compare_nulls,
  sorted is_sorted,
  rmm::cuda_stream_view stream)
{
  preprocessed_table result;
  result._table_view = table;

  if (compare_nulls == null_equality::EQUAL) {
    result._null_processed_table_view = table;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    if (has_nested_nulls(table)) {
      result.preprocess_unprocessed_table(stream);
    } else {
      result._null_processed_table_view = table;
    }
  }

  if (is_sorted == cudf::sorted::NO) { result.compute_sorted_order(stream); }

  return result;
}

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
  : preprocessed_right{preprocessed_table::create(right, compare_nulls, is_right_sorted, stream)},
    compare_nulls{compare_nulls}
{
  cudf::scoped_range range{"sort_merge_join::sort_merge_join"};
  // Sanity checks
  CUDF_EXPECTS(right.num_columns() != 0,
               "Number of columns the keys table must be non-zero for a join",
               std::invalid_argument);
}

rmm::device_uvector<size_type> sort_merge_join::preprocessed_table::map_table_to_unprocessed(
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(), "Mapping is not possible");
  auto temp_mr                  = cudf::get_current_device_resource_ref();
  auto const table_mapping_size = _table_view.num_rows() - _num_nulls.value();
  rmm::device_uvector<size_type> table_mapping(table_mapping_size, stream, temp_mr);
  cudf::detail::copy_if_async(
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(_table_view.num_rows()),
    cuda::counting_iterator<size_type>(0),
    table_mapping.begin(),
    is_row_valid{static_cast<bitmask_type const*>(_validity_mask.value().data())},
    stream);
  return table_mapping;
}

void sort_merge_join::postprocess_indices(preprocessed_table const& preprocessed_left,
                                          device_span<size_type> smaller_indices,
                                          device_span<size_type> larger_indices,
                                          rmm::cuda_stream_view stream) const
{
  using sort_merge_join_detail::index_mapping;
  if (compare_nulls == null_equality::UNEQUAL) {
    // if a table has no nullable column, then there's no postprocessing to be done
    if (has_nested_nulls(preprocessed_left._table_view)) {
      auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      cub::DeviceTransform::Transform(larger_indices.begin(),
                                      larger_indices.begin(),
                                      larger_indices.size(),
                                      index_mapping<device_span<size_type>>{left_mapping},
                                      stream.value());
    }
    if (has_nested_nulls(preprocessed_right._table_view)) {
      auto right_mapping = preprocessed_right.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      cub::DeviceTransform::Transform(smaller_indices.begin(),
                                      smaller_indices.begin(),
                                      smaller_indices.size(),
                                      index_mapping<device_span<size_type>>{right_mapping},
                                      stream.value());
    }
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            sorted is_left_sorted,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.inner(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
      return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
    },
    stream);
}

// left_partition_end exclusive
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::partitioned_inner_join"};

  // Extract preprocessed_left from the context
  auto const& preprocessed_left =
    static_cast<sort_merge_join_match_context const*>(context.left_table_context.get())
      ->preprocessed_left;

  auto const left_partition_start_idx = context.left_start_idx;
  auto const left_partition_end_idx   = context.left_end_idx;
  auto null_processed_table_start_idx = left_partition_start_idx;
  auto null_processed_table_end_idx   = left_partition_end_idx;
  if (compare_nulls == null_equality::UNEQUAL && has_nested_nulls(preprocessed_left._table_view)) {
    auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
    null_processed_table_start_idx =
      cuda::std::distance(left_mapping.begin(),
                          thrust::lower_bound(rmm::exec_policy_nosync(stream),
                                              left_mapping.begin(),
                                              left_mapping.end(),
                                              left_partition_start_idx));
    null_processed_table_end_idx =
      cuda::std::distance(left_mapping.begin(),
                          thrust::upper_bound(rmm::exec_policy_nosync(stream),
                                              left_mapping.begin(),
                                              left_mapping.end(),
                                              left_partition_end_idx - 1));
  }
  auto null_processed_left_partition =
    cudf::slice(preprocessed_left._null_processed_table_view,
                {null_processed_table_start_idx, null_processed_table_end_idx},
                stream)[0];

  auto [preprocessed_right_indices, preprocessed_left_indices] = invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    null_processed_left_partition,
    [this, left_partition_start_idx, stream, mr](auto& obj) { return obj.inner(stream, mr); },
    stream);
  // Map from slice to total null processed table
  // Use cub API to handle large arrays (> INT32_MAX)
  cub::DeviceTransform::Transform(
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->size(),
    [left_partition_start_idx] __device__(auto idx) { return left_partition_start_idx + idx; },
    stream.value());
  // Map from total null processed table to unprocessed table
  postprocess_indices(
    preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
  return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
}

}  // namespace detail

sort_merge_join::~sort_merge_join() = default;

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(right, is_right_sorted, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            sorted is_left_sorted,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, is_left_sorted, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           sorted is_left_sorted,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, is_left_sorted, stream, mr);
}

std::unique_ptr<join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left,
  sorted is_left_sorted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, is_left_sorted, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr) const
{
  return _impl->partitioned_inner_join(context, stream, mr);
}

}  // namespace cudf
