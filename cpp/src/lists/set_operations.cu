/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/combine.hpp>
#include <cudf/lists/detail/set_operations.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief Check if two input lists columns are valid input into the list operations.
 * @param lhs The left lists column
 * @param rhs The right lists column
 */
void check_compatibility(lists_column_view const& lhs, lists_column_view const& rhs)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "The input lists column must have the same size.");
  CUDF_EXPECTS(column_types_equal(lhs.child(), rhs.child()),
               "The input lists columns must have children having the same type structure");
}

}  // namespace

std::unique_ptr<column> have_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  check_compatibility(lhs, rhs);

  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {rhs_labels, rhs_child} in the table
  //   {lhs_labels, lhs_child}.
  // - `reduce_by_key` with keys are rhs_labels and `logical_or` reduction on the existence reults
  //   computed in the previous step.

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, lhs_child.size(), stream);
  auto const rhs_labels = generate_labels(rhs, rhs_child.size(), stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  // Check existence for each row of the rhs_table in lhs_table.
  auto const contained =
    cudf::detail::contains(lhs_table, rhs_table, nulls_equal, nans_equal, stream);

  auto const num_rows = lhs.size();

  // This stores the unique label values, used as scatter map.
  auto list_indices = rmm::device_uvector<size_type>(num_rows, stream);

  // Stores the result of checking overlap for non-empty lists.
  auto overlap_results = rmm::device_uvector<bool>(num_rows, stream);

  auto const labels_begin           = rhs_labels->view().begin<size_type>();
  auto const end                    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         labels_begin,  // keys
                                         labels_begin + rhs_labels->size(),  // keys
                                         contained.begin(),  // values to reduce
                                         list_indices.begin(),     // out keys
                                         overlap_results.begin(),  // out values
                                         thrust::equal_to{},  // comp for keys
                                         thrust::logical_or{});  // reduction op for values
  auto const num_non_empty_segments = thrust::distance(overlap_results.begin(), end.second);

  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{lhs.parent(), rhs.parent()}}, stream, mr);
  auto result = make_numeric_column(
    data_type{type_to_id<bool>()}, num_rows, std::move(null_mask), null_count, stream, mr);
  auto const result_begin = result->mutable_view().begin<bool>();

  // `overlap_results` only stores the results of non-empty lists.
  // We need to initialize `false` for the entire output array then scatter these results over.
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), result_begin, result_begin + num_rows, false);
  thrust::scatter(rmm::exec_policy(stream),
                  overlap_results.begin(),
                  overlap_results.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  result_begin);

  result->set_null_count(null_count);

  return result;
}

std::unique_ptr<column> intersect_distinct(lists_column_view const& lhs,
                                           lists_column_view const& rhs,
                                           null_equality nulls_equal,
                                           nan_equality nans_equal,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  check_compatibility(lhs, rhs);

  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {rhs_labels, rhs_child} in the table
  //   {lhs_labels, lhs_child}.
  // - Extract rows of the rhs table using the existence results computed in the previous step.
  // - Remove duplicate rows, and build the output lists.

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, lhs_child.size(), stream);
  auto const rhs_labels = generate_labels(rhs, rhs_child.size(), stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  auto const contained =
    cudf::detail::contains(lhs_table, rhs_table, nulls_equal, nans_equal, stream);

  auto const intersect_table = cudf::detail::copy_if(
    rhs_table,
    [contained = contained.begin()] __device__(auto const idx) { return contained[idx]; },
    stream);

  // A stable algorithm is required to ensure that list labels remain contiguous.
  auto out_table = cudf::detail::stable_distinct(intersect_table->view(),
                                                 {0, 1},  // indices of key columns
                                                 duplicate_keep_option::KEEP_ANY,
                                                 nulls_equal,
                                                 nans_equal,
                                                 stream,
                                                 mr);

  auto const num_rows = lhs.size();
  auto out_offsets    = reconstruct_offsets(out_table->get_column(0).view(), num_rows, stream, mr);
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{lhs.parent(), rhs.parent()}}, stream, mr);
  auto output = make_lists_column(num_rows,
                                  std::move(out_offsets),
                                  std::move(out_table->release().back()),
                                  null_count,
                                  std::move(null_mask),
                                  stream,
                                  mr);

  return null_count == 0 ? std::move(output)
                         : cudf::detail::purge_nonempty_nulls(output->view(), stream, mr);
}

std::unique_ptr<column> union_distinct(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  check_compatibility(lhs, rhs);

  // Algorithm: `return distinct(concatenate_rows(lhs, rhs))`.

  auto const union_col = lists::detail::concatenate_rows(
    table_view{{lhs.parent(), rhs.parent()}}, concatenate_null_policy::NULLIFY_OUTPUT_ROW, stream);

  return cudf::lists::detail::distinct(
    lists_column_view{union_col->view()}, nulls_equal, nans_equal, stream, mr);
}

std::unique_ptr<column> difference_distinct(lists_column_view const& lhs,
                                            lists_column_view const& rhs,
                                            null_equality nulls_equal,
                                            nan_equality nans_equal,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  check_compatibility(lhs, rhs);

  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {lhs_labels, lhs_child} in the table
  //   {rhs_labels, rhs_child}.
  // - Invert the existence results computed in the previous step, resulting in difference results.
  // - Extract rows of the lhs table using that difference results.
  // - Remove duplicate rows, and build the output lists.

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, lhs_child.size(), stream);
  auto const rhs_labels = generate_labels(rhs, rhs_child.size(), stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  auto const contained =
    cudf::detail::contains(rhs_table, lhs_table, nulls_equal, nans_equal, stream);

  auto const difference_table = cudf::detail::copy_if(
    lhs_table,
    [contained = contained.begin()] __device__(auto const idx) { return !contained[idx]; },
    stream);

  // A stable algorithm is required to ensure that list labels remain contiguous.
  auto out_table = cudf::detail::stable_distinct(difference_table->view(),
                                                 {0, 1},  // indices of key columns
                                                 duplicate_keep_option::KEEP_ANY,
                                                 nulls_equal,
                                                 nans_equal,
                                                 stream,
                                                 mr);

  auto const num_rows = lhs.size();
  auto out_offsets    = reconstruct_offsets(out_table->get_column(0).view(), num_rows, stream, mr);
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{lhs.parent(), rhs.parent()}}, stream, mr);

  auto output = make_lists_column(num_rows,
                                  std::move(out_offsets),
                                  std::move(out_table->release().back()),
                                  null_count,
                                  std::move(null_mask),
                                  stream,
                                  mr);

  return null_count == 0 ? std::move(output)
                         : cudf::detail::purge_nonempty_nulls(output->view(), stream, mr);
}

}  // namespace detail

std::unique_ptr<column> have_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::have_overlap(lhs, rhs, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> intersect_distinct(lists_column_view const& lhs,
                                           lists_column_view const& rhs,
                                           null_equality nulls_equal,
                                           nan_equality nans_equal,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::intersect_distinct(
    lhs, rhs, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> union_distinct(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::union_distinct(lhs, rhs, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> difference_distinct(lists_column_view const& lhs,
                                            lists_column_view const& rhs,
                                            null_equality nulls_equal,
                                            nan_equality nans_equal,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::difference_distinct(
    lhs, rhs, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

}  // namespace cudf::lists
