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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/detail/sorting.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <cudf_test/column_utilities.hpp>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Generate list offsets and list validities for the output lists column from the table_view
 * of the input lists columns.
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(table_view const& input,
                                     bool has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const num_cols  = input.num_columns();
  auto const num_rows  = input.num_rows();
  auto const num_lists = num_rows * num_cols;

  // The output offsets column
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  static_assert(sizeof(size_type) == sizeof(int32_t));
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_lists + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = list_offsets->mutable_view().begin<offset_type>();
  auto const table_dv_ptr  = table_device_view::create(input);

  // The array of int8_t to store element validities
  auto validities = has_null_mask ? rmm::device_uvector<int8_t>(num_lists, stream)
                                  : rmm::device_uvector<int8_t>(0, stream);

  // Compute list sizes
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(num_lists),
                    d_out_offsets,
                    [num_cols,
                     table_dv     = *table_dv_ptr,
                     d_validities = validities.begin(),
                     has_null_mask] __device__(size_type const dst_list_id) {
                      auto const src_col_id  = dst_list_id % num_cols;
                      auto const src_list_id = dst_list_id / num_cols;
                      auto const& src_col    = table_dv.column(src_col_id);
                      auto const is_valid    = src_col.is_valid(src_list_id);

                      if (has_null_mask) {
                        d_validities[dst_list_id] = static_cast<int8_t>(is_valid);
                      }
                      if (not is_valid) { return size_type{0}; }
                      auto const d_offsets =
                        src_col.child(lists_column_view::offsets_column_index).data<size_type>() +
                        src_col.offset();
                      return d_offsets[src_list_id + 1] - d_offsets[src_list_id];
                    });

  // Compute offsets from sizes
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + num_lists + 1, d_out_offsets);

  return {std::move(list_offsets), std::move(validities)};
}

/**
 * @brief Struct used in type_dispatcher to copy entries to the output lists column
 */
struct copy_list_entries_fn {
  template <class T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& list_offsets,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    return nullptr;
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    table_view const& input,
    column_view const& list_offsets,
    bool has_null_mask,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    auto const child_col    = lists_column_view(*input.begin()).child();
    auto const num_cols     = input.num_columns();
    auto const num_rows     = input.num_rows();
    auto const num_lists    = num_rows * num_cols;
    auto const output_size  = cudf::detail::get_value<size_type>(list_offsets, num_lists, stream);
    auto const table_dv_ptr = table_device_view::create(input);

    printf("Line %d\n", __LINE__);

    printf("Line %d\n", __LINE__);

    printf("Line %d\n", __LINE__);

    auto output = allocate_like(child_col, output_size, mask_allocation_policy::NEVER, stream, mr);
    auto output_dv_ptr = mutable_column_device_view::create(*output);

    printf("Line %d\n", __LINE__);

    // The array of int8_t to store element validities
    auto validities = has_null_mask ? rmm::device_uvector<int8_t>(output_size, stream)
                                    : rmm::device_uvector<int8_t>(0, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_lists,
      [num_cols,
       table_dv     = *table_dv_ptr,
       d_validities = validities.begin(),
       dst_offsets  = list_offsets.begin<offset_type>(),
       d_output     = output_dv_ptr->begin<T>(),
       has_null_mask] __device__(size_type const dst_list_id) {
        auto const src_col_id     = dst_list_id % num_cols;
        auto const src_list_id    = dst_list_id / num_cols;
        auto const& src_lists_col = table_dv.column(src_col_id);
        auto const& src_child     = src_lists_col.child(lists_column_view::child_column_index);
        auto const src_child_offsets =
          src_lists_col.child(lists_column_view::offsets_column_index).data<size_type>() +
          src_lists_col.offset();

        auto write_idx = dst_offsets[dst_list_id];
        for (auto read_idx = src_child_offsets[src_list_id],
                  idx_end  = src_child_offsets[src_list_id + 1];
             read_idx < idx_end;
             ++read_idx, ++write_idx) {
          auto const is_valid = src_child.is_valid(read_idx);
          if (has_null_mask) { d_validities[dst_list_id] = static_cast<int8_t>(is_valid); }
          d_output[write_idx] = is_valid ? src_child.element<T>(read_idx) : T{};
        }
      });

    if (has_null_mask) {
      auto [null_mask, null_count] = cudf::detail::valid_if(
        validities.begin(),
        validities.end(),
        [] __device__(auto const valid) { return valid; },
        stream,
        mr);
      if (null_count > 0) { output->set_null_mask(null_mask, null_count); }
    }

    return output;
  }

  template <class T>
  std::enable_if_t<not std::is_same_v<T, cudf::string_view> and not cudf::is_fixed_width<T>(),
                   std::unique_ptr<column>>
  operator()(table_view const&,
             column_view const&,
             bool,
             rmm::cuda_stream_view,
             rmm::mr::device_memory_resource*) const
  {
    // Currently, only support string_view and fixed-width types
    CUDF_FAIL("Called `copy_list_entries_fn()` on non-supported types.");
  }
};

}  // anonymous namespace

/**
 * @copydoc cudf::lists::detail::interleave_columns
 *
 */
std::unique_ptr<column> interleave_columns(table_view const& input,
                                           bool has_null_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto const entry_type = lists_column_view(*input.begin()).child().type();
  for (auto const& col : input) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(not cudf::is_nested(child_col.type()), "Nested types are not supported.");
    CUDF_EXPECTS(entry_type == child_col.type(),
                 "The types of entries in the input columns must be the same.");
  }

  if (input.num_columns() == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }
  if (input.num_rows() == 0) { return cudf::empty_like(input.column(0)); }

  // Generate offsets of the output lists column
  auto [list_offsets, list_validities] =
    generate_list_offsets_and_validities(input, has_null_mask, stream, mr);

  printf("Line %d\n", __LINE__);

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types
  auto const output_size = input.num_rows() * input.num_columns();
  auto list_entries      = type_dispatcher<dispatch_storage_type>(
    entry_type, copy_list_entries_fn{}, input, list_offsets->view(), has_null_mask, stream, mr);

  printf("Line %d\n", __LINE__);

  if (not has_null_mask) {
    return make_lists_column(output_size,
                             std::move(list_offsets),
                             std::move(list_entries),
                             0,
                             rmm::device_buffer{},
                             stream,
                             mr);
  }

  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(),
    list_validities.end(),
    [] __device__(auto const valid) { return valid; },
    stream,
    mr);
  printf("Line %d\n", __LINE__);

  return make_lists_column(output_size,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
