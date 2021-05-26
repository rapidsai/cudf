/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Concatenate lists within the same row into one list, ignoring any null list during
 * concatenation.
 */
std::unique_ptr<column> concatenate_lists_ignore_null(column_view const& input,
                                                      bool build_null_mask,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.size();

  static_assert(std::is_same_v<offset_type, int32_t> && std::is_same_v<size_type, int32_t>);
  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  // The array of int8_t stores validities for the output list elements.
  auto validities = rmm::device_uvector<int8_t>(build_null_mask ? num_rows : 0, stream);

  auto const d_out_offsets  = out_offsets->mutable_view().template begin<offset_type>();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();
  auto const lists_dv_ptr   = column_device_view::create(lists_column_view(input).child());

  // Concatenating the lists at the same row by converting the entry offsets from the child column
  // into row offsets of the root column. Those entry offsets are subtracted by the first entry
  // offset to output zero-based offsets.
  auto const iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_rows + 1,
                    d_out_offsets,
                    [d_row_offsets,
                     d_list_offsets,
                     lists_dv     = *lists_dv_ptr,
                     d_validities = validities.begin(),
                     build_null_mask,
                     iter] __device__(auto const idx) {
                      if (build_null_mask) {
                        // The output row will be null only if all lists on the input row are null.
                        auto const is_valid = thrust::any_of(thrust::seq,
                                                             iter + d_row_offsets[idx],
                                                             iter + d_row_offsets[idx + 1],
                                                             [&] __device__(auto const list_idx) {
                                                               return lists_dv.is_valid(list_idx);
                                                             });
                        d_validities[idx]   = static_cast<int8_t>(is_valid);
                      }
                      auto const start_offset = d_list_offsets[d_row_offsets[0]];
                      return d_list_offsets[d_row_offsets[idx]] - start_offset;
                    });

  // The child column of the output lists column is just copied from the input column.
  auto out_entries = std::make_unique<column>(
    lists_column_view(lists_column_view(input).get_sliced_child(stream)).get_sliced_child(stream));

  auto [null_mask, null_count] = [&] {
    return build_null_mask
             ? cudf::detail::valid_if(
                 validities.begin(), validities.end(), thrust::identity<int8_t>{}, stream, mr)
             : std::make_pair(cudf::detail::copy_bitmask(input, stream, mr), input.null_count());
  }();

  return make_lists_column(num_rows,
                           std::move(out_offsets),
                           std::move(out_entries),
                           null_count,
                           null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

/**
 * @brief Generate list offsets and list validities for the output lists column.
 *
 * This function is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.size();

  static_assert(std::is_same_v<offset_type, int32_t> && std::is_same_v<size_type, int32_t>);
  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  auto const lists_of_lists_dv_ptr = column_device_view::create(input);
  auto const lists_dv_ptr          = column_device_view::create(lists_column_view(input).child());
  auto const d_out_offsets         = out_offsets->mutable_view().template begin<offset_type>();
  auto const d_row_offsets         = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();

  // The array of int8_t stores validities for the output list elements.
  auto validities = rmm::device_uvector<int8_t>(num_rows, stream);

  // Compute output list sizes and validities.
  auto const iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(
    rmm::exec_policy(stream),
    iter,
    iter + num_rows,
    d_out_offsets,
    [lists_of_lists_dv = *lists_of_lists_dv_ptr,
     lists_dv          = *lists_dv_ptr,
     d_row_offsets,
     d_list_offsets,
     d_validities = validities.begin(),
     iter] __device__(auto const idx) {
      if (d_row_offsets[idx] == d_row_offsets[idx + 1]) {  // This is a null/empty row.
        d_validities[idx] = static_cast<int8_t>(lists_of_lists_dv.is_valid(idx));
        return size_type{0};
      }
      // The output row will not be null only if all lists on the input row are not null.
      auto const is_valid =
        thrust::all_of(thrust::seq,
                       iter + d_row_offsets[idx],
                       iter + d_row_offsets[idx + 1],
                       [&] __device__(auto const list_idx) { return lists_dv.is_valid(list_idx); });
      d_validities[idx] = static_cast<int8_t>(is_valid);
      if (!is_valid) { return size_type{0}; }

      // Compute size of the output list as sum of sizes of all lists in the current input row.
      return d_list_offsets[d_row_offsets[idx + 1]] - d_list_offsets[d_row_offsets[idx]];
    });

  // Compute offsets from sizes.
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + num_rows + 1, d_out_offsets);

  return {std::move(out_offsets), std::move(validities)};
}

/**
 * @brief Gather entries from the input lists column, ignoring rows that have null list elements.
 *
 * This function is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
std::unique_ptr<column> gather_list_entries(column_view const& input,
                                            column_view const& output_list_offsets,
                                            size_type num_rows,
                                            size_type num_output_entries,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  auto const child_col      = lists_column_view(input).child();
  auto const entry_col      = lists_column_view(child_col).child();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(child_col).offsets_begin();
  auto gather_map           = rmm::device_uvector<size_type>(num_output_entries, stream);

  // Fill the gather map with indices of the lists from the child column of the input column.
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    num_rows,
    [d_row_offsets,
     d_list_offsets,
     d_indices = gather_map.begin(),
     d_out_list_offsets =
       output_list_offsets.template begin<offset_type>()] __device__(size_type const idx) {
      // The output row has been identified as a null/empty list during list size computation.
      if (d_out_list_offsets[idx + 1] == d_out_list_offsets[idx]) { return; }

      // The indices of the list elements on the row `idx` of the input column.
      thrust::sequence(thrust::seq,
                       d_indices + d_out_list_offsets[idx],
                       d_indices + d_out_list_offsets[idx + 1],
                       d_list_offsets[d_row_offsets[idx]]);
    });

  auto result = cudf::detail::gather(table_view{{entry_col}},
                                     gather_map.begin(),
                                     gather_map.end(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> concatenate_lists_nullifying_rows(column_view const& input,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::mr::device_memory_resource* mr)
{
  // Generate offsets and validities of the output lists column.
  auto [list_offsets, list_validities] = generate_list_offsets_and_validities(input, stream, mr);
  auto const offsets_view              = list_offsets->view();

  auto const num_rows = input.size();
  auto const num_output_entries =
    cudf::detail::get_value<size_type>(offsets_view, num_rows, stream);

  auto list_entries =
    gather_list_entries(input, offsets_view, num_rows, num_output_entries, stream, mr);
  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(), list_validities.end(), thrust::identity<int8_t>{}, stream, mr);

  return make_lists_column(num_rows,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace

/**
 * @copydoc cudf::lists::concatenate_list_elements
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto type = input.type();  // Column that is lists of lists.
  CUDF_EXPECTS(type.id() == type_id::LIST, "Input column must be a lists column.");

  auto col = lists_column_view(input).child();  // Rows, which are lists.
  type     = col.type();
  CUDF_EXPECTS(type.id() == type_id::LIST, "Rows of the input column must be lists.");

  col  = lists_column_view(col).child();  // The last level entries what we need to check.
  type = col.type();
  CUDF_EXPECTS(type.id() == type_id::LIST || !cudf::is_nested(type),
               "Entry of the input lists column must be of list or non-nested types.");

  if (input.size() == 0) { return cudf::empty_like(input); }

  bool has_null_list = lists_column_view(input).child().has_nulls();

  return (null_policy == concatenate_null_policy::IGNORE || !has_null_list)
           ? concatenate_lists_ignore_null(input, has_null_list, stream, mr)
           : concatenate_lists_nullifying_rows(input, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_list_elements
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(input, null_policy, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
