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

#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.cuh>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/combine.hpp>
#include <cudf/lists/detail/set_operations.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

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
  CUDF_EXPECTS(lhs.child().type() == rhs.child().type(),
               "The input lists column must have children having the same data types");
}

/**
 * @brief Generate list labels for elements in the child column of the input lists column.
 *
 * @param input The input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A column containing list labels corresponding to each input list elements
 */
std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        rmm::cuda_stream_view stream)
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), n_elements, cudf::mask_state::UNALLOCATED, stream);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + n_elements, stream);
  return labels;
}

/**
 * @brief Reconstruct an offsets column from the input labels array.
 *
 * @param labels The list labels corresponding to each input list elements
 * @param n_lists The number of lists in the input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The output offsets column
 */
std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_lists,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)

{
  auto out_offsets = make_numeric_column(
    data_type{type_to_id<offset_type>()}, n_lists + 1, mask_state::UNALLOCATED, stream, mr);

  auto const labels_begin  = labels.template begin<size_type>();
  auto const offsets_begin = out_offsets->mutable_view().template begin<size_type>();
  cudf::detail::labels_to_offsets(labels_begin,
                                  labels_begin + labels.size(),
                                  offsets_begin,
                                  offsets_begin + out_offsets->size(),
                                  stream);
  return out_offsets;
}

/**
 * @brief Remove duplicate list elements from a lists column.
 *
 * For an input lists column, the distinct elements from each of its list row are copied (with
 * order preserved) into an output child column to form a set of new lists without duplicates. An
 * output offsets column corresponding to these new lists is also constructed and returned.
 *
 * The input lists column is not given to this function directly. Instead, its child column and a
 * label array containing the corresponding list labels for each element are used to access the
 * input lists.
 *
 * @param n_lists Number of lists in the input lists column
 * @param child_labels Array containing list labels of the list elements
 * @param child The child column of the input lists column
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A pair of output columns `{out_offsets, out_child}`
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> list_distinct_children(
  size_type n_lists,
  column_view const& child_labels,
  column_view const& child,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const input_table = table_view{{child_labels, child}};

  auto const distinct_indices = cudf::detail::get_distinct_indices(
    input_table, duplicate_keep_option::KEEP_ANY, nulls_equal, nans_equal, stream);

  auto const index_markers = [&] {
    auto markers = rmm::device_uvector<bool>(child.size(), stream);
    thrust::uninitialized_fill(rmm::exec_policy(stream), markers.begin(), markers.end(), false);
    thrust::scatter(
      rmm::exec_policy(stream),
      thrust::constant_iterator<size_type>(true, 0),
      thrust::constant_iterator<size_type>(true, static_cast<size_type>(distinct_indices.size())),
      distinct_indices.begin(),
      markers.begin());
    return markers;
  }();

  auto const output_table = cudf::detail::copy_if(
    input_table,
    [index_markers = index_markers.begin()] __device__(auto const idx) {
      return index_markers[idx];
    },
    stream,
    mr);
  auto out_offsets = reconstruct_offsets(output_table->get_column(0).view(), n_lists, stream, mr);

  return std::pair(std::move(out_offsets), std::move(output_table->release().back()));
}

}  // namespace

std::unique_ptr<column> list_distinct(
  lists_column_view const& input,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // Algorithm:
  // - Generate labels for the child elements.
  // - Get indices of distinct rows of the table {labels, child}.
  // - Scatter these indices into a marker array that marks if a row will be copied to the output.
  // - Collect output rows (with order preserved) using the marker array and build the output
  //   lists column.

  auto const child  = input.get_sliced_child(stream);
  auto const labels = generate_labels(input, child.size(), stream);

  auto [out_offsets, out_child] = list_distinct_children(
    input.size(), labels->view(), child, nulls_equal, nans_equal, stream, mr);

  return make_lists_column(input.size(),
                           std::move(out_offsets),
                           std::move(out_child),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

std::unique_ptr<column> list_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {rhs_labels, rhs_child} in the table
  //   {lhs_labels, lhs_child}.
  // - `reduce_by_key` with keys are rhs_labels and `logical_or` reduction on the existence array
  //   computed in the previous step.

  check_compatibility(lhs, rhs);

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, lhs_child.size(), stream);
  auto const rhs_labels = generate_labels(rhs, rhs_child.size(), stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  // Check existence for each row of the rhs_table in the lhs_table.
  auto const contained =
    cudf::detail::contains(lhs_table, rhs_table, nulls_equal, nans_equal, stream);

  // This stores the unique label values, used as scatter map.
  auto list_indices = rmm::device_uvector<size_type>(lhs.size(), stream);

  // Stores the result of checking overlap for non-empty lists.
  auto overlap_results = rmm::device_uvector<bool>(lhs.size(), stream);

  auto const labels_begin           = rhs_labels->view().template begin<size_type>();
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
    data_type{type_to_id<bool>()}, lhs.size(), std::move(null_mask), null_count, stream, mr);
  auto const result_begin = result->mutable_view().template begin<bool>();

  // `overlap_results` only stores the results of non-empty lists.
  // We need to initialize `false` for the entire output array then scatter these results over.
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), result_begin, result_begin + lhs.size(), false);
  thrust::scatter(rmm::exec_policy(stream),
                  overlap_results.begin(),
                  overlap_results.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  result_begin);

  return result;
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {rhs_labels, rhs_child} in the table
  //   {lhs_labels, lhs_child}.
  // - Extract rows of the rhs table using the existence array computed in the previous step.
  // - Remove duplicate rows, and build the output lists.

  check_compatibility(lhs, rhs);

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

  auto [out_offsets, out_child] = list_distinct_children(lhs.size(),
                                                         intersect_table->get_column(0).view(),
                                                         intersect_table->get_column(1).view(),
                                                         nulls_equal,
                                                         nans_equal,
                                                         stream,
                                                         mr);
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{lhs.parent(), rhs.parent()}}, stream, mr);
  auto output = make_lists_column(lhs.size(),
                                  std::move(out_offsets),
                                  std::move(out_child),
                                  null_count,
                                  std::move(null_mask),
                                  stream,
                                  mr);

  return null_count == 0
           ? std::move(output)
           : cudf::detail::purge_nonempty_nulls(lists_column_view{output->view()}, stream, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  check_compatibility(lhs, rhs);

  auto const lhs_distinct = list_distinct(lhs, nulls_equal, nans_equal, stream);

  // The result table from set_different already contains distinct rows.
  auto const diff = lists::detail::set_difference(rhs, lhs, nulls_equal, nans_equal, stream);

  return lists::detail::concatenate_rows(table_view{{lhs_distinct->view(), diff->view()}},
                                         concatenate_null_policy::NULLIFY_OUTPUT_ROW,
                                         stream,
                                         mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  // Algorithm:
  // - Generate labels for lhs and rhs child elements.
  // - Check existence for rows of the table {lhs_labels, lhs_child} in the table
  //   {rhs_labels, rhs_child}.
  // - Invert the existence array computed in the previous step, resulting in a difference array.
  // - Extract rows of the lhs table using that difference array.
  // - Remove duplicate rows, and build the output lists.

  check_compatibility(lhs, rhs);

  auto const lhs_child  = lhs.get_sliced_child(stream);
  auto const rhs_child  = rhs.get_sliced_child(stream);
  auto const lhs_labels = generate_labels(lhs, lhs_child.size(), stream);
  auto const rhs_labels = generate_labels(rhs, rhs_child.size(), stream);
  auto const lhs_table  = table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = table_view{{rhs_labels->view(), rhs_child}};

  auto const inv_contained = [&] {
    auto contained = cudf::detail::contains(rhs_table, lhs_table, nulls_equal, nans_equal, stream);
    thrust::transform(rmm::exec_policy(stream),
                      contained.begin(),
                      contained.end(),
                      contained.begin(),
                      thrust::logical_not{});
    return contained;
  }();

  auto const difference_table = cudf::detail::copy_if(
    lhs_table,
    [inv_contained = inv_contained.begin()] __device__(auto const idx) {
      return inv_contained[idx];
    },
    stream);

  auto [out_offsets, out_child] = list_distinct_children(lhs.size(),
                                                         difference_table->get_column(0).view(),
                                                         difference_table->get_column(1).view(),
                                                         nulls_equal,
                                                         nans_equal,
                                                         stream,
                                                         mr);
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{lhs.parent(), rhs.parent()}}, stream, mr);
  auto output = make_lists_column(lhs.size(),
                                  std::move(out_offsets),
                                  std::move(out_child),
                                  null_count,
                                  std::move(null_mask),
                                  stream,
                                  mr);

  return null_count == 0
           ? std::move(output)
           : cudf::detail::purge_nonempty_nulls(lists_column_view{output->view()}, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> list_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::list_overlap(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      null_equality nulls_equal,
                                      nan_equality nans_equal,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_intersect(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  null_equality nulls_equal,
                                  nan_equality nans_equal,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_union(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_difference(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_difference(lhs, rhs, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
