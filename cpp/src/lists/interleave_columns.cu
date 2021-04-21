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

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Generate list offsets for the output lists column from the table_view of the input lists
 * columns.
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(table_view const& input,
                                     bool has_null_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  auto const num_cols    = input.num_columns();
  auto const output_size = input.num_rows() * num_cols + 1;

  // The output offsets column
  static_assert(sizeof(offset_type) == sizeof(int32_t));
  static_assert(sizeof(size_type) == sizeof(int32_t));
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = list_offsets->mutable_view().begin<offset_type>();
  auto const table_dv_ptr  = table_device_view::create(input);

  // The array of int8_t to store element validities
  auto validities = has_null_mask ? rmm::device_uvector<int8_t>(output_size - 1, stream)
                                  : rmm::device_uvector<int8_t>(0, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), validities.begin(), validities.end(), int8_t{1});

  // Compute list sizes
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(output_size - 1),
                    d_out_offsets,
                    [num_cols,
                     table_dv     = *table_dv_ptr,
                     d_validities = validities.begin(),
                     has_null_mask] __device__(size_type const idx) {
                      auto const col_id   = idx % num_cols;
                      auto const list_id  = idx / num_cols;
                      auto const d_column = table_dv.column(col_id);
                      if (d_column.is_null(list_id)) {
                        if (has_null_mask) { d_validities[idx] = 0; }
                        return size_type{0};
                      }
                      auto const d_offsets =
                        d_column.child(lists_column_view::offsets_column_index).data<size_type>() +
                        d_column.offset();
                      return d_offsets[list_id + 1] - d_offsets[list_id];
                    });

  // Compute offsets from sizes
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + output_size, d_out_offsets);

  return {std::move(list_offsets), std::move(validities)};
}

/**
 * @brief Struct used in type_dispatcher to copy entries to the output lists column
 */
struct copy_list_entries_fn {
  template <class T>
  std::enable_if_t<not std::is_same_v<T, cudf::string_view> and not cudf::is_fixed_width<T>(),
                   std::unique_ptr<column>>
  operator()(table_view const& input,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    // Currently, only support string_view and fixed-width types
    CUDF_FAIL("Called `copy_list_entries_fn()` on non-supported types.");
  }

  template <class T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    table_view const& input,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    return nullptr;
  }

  template <class T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, std::unique_ptr<column>> operator()(
    table_view const& input,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const noexcept
  {
    return nullptr;
  }
};

/**
 * @brief Concatenate strings functor.
 *
 * This will concatenate the strings within each list element of the given string lists column
 * and apply the separator. The null-replacement string scalar `string_narep_dv` (if valid) is
 * used in place of any null string.
 */
struct concat_strings_fn {
  column_device_view const lists_dv;
  offset_type const* const list_offsets;
  column_device_view const strings_dv;
  string_scalar_device_view const string_narep_dv;
  string_view const separator;

  offset_type* d_offsets{nullptr};

  // If `d_chars == nullptr`: only compute sizes of the output strings.
  // If `d_chars != nullptr`: only concatenate strings.
  char* d_chars{nullptr};

  __device__ void operator()(size_type idx)
  {
    if (lists_dv.is_null(idx)) {
      if (!d_chars) { d_offsets[idx] = size_type{-1}; }  // negative size means null string
      return;
    }

    // If the string offsets have been computed and the row `idx` is known to be a null string
    if (d_chars && d_offsets[idx] == d_offsets[idx + 1]) { return; }

    auto const separator_size = separator.size_bytes();
    auto size_bytes           = size_type{0};
    bool written              = false;
    char* output_ptr          = d_chars ? d_chars + d_offsets[idx] : nullptr;

    for (size_type str_idx = list_offsets[idx], idx_end = list_offsets[idx + 1]; str_idx < idx_end;
         ++str_idx) {
      if (strings_dv.is_null(str_idx) && !string_narep_dv.is_valid()) {
        if (!d_chars) { d_offsets[idx] = size_type{-1}; }  // negative size means null string
        return;  // early termination: the entire list of strings will result in a null string
      }
      auto const d_str = strings_dv.is_null(str_idx) ? string_narep_dv.value()
                                                     : strings_dv.element<string_view>(str_idx);
      size_bytes += separator_size + d_str.size_bytes();
      if (output_ptr) {
        // Separator is inserted only in between strings
        if (written) { output_ptr = detail::copy_string(output_ptr, separator); }
        output_ptr = detail::copy_string(output_ptr, d_str);
        written    = true;
      }
    }

    // Separator is inserted only in between strings
    if (!d_chars) { d_offsets[idx] = static_cast<size_type>(size_bytes - separator_size); }
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

  // Copy entries from the input lists columns to the output lists column - this needed to be
  // specialized for different types
  auto list_entries = type_dispatcher(entry_type, copy_list_entries_fn{}, input, stream, mr);

  auto const num_output_rows = input.num_rows() * input.num_columns();

  if (not has_null_mask) {
    return make_lists_column(num_output_rows,
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

  return make_lists_column(num_output_rows,
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
