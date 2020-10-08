/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <strings/utilities.cuh>

namespace cudf {
namespace detail {
namespace {
struct interleave_columns_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_fixed_width<T>() and not std::is_same<T, cudf::string_view>::value,
                   std::unique_ptr<cudf::column>>
  operator()(Args&&... args)
  {
    CUDF_FAIL("interleave_columns not supported for dictionary and list types.");
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, cudf::string_view>::value, std::unique_ptr<cudf::column>>
  operator()(table_view const& strings_columns,
             bool create_mask,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream = 0)
  {
    auto num_columns = strings_columns.num_columns();
    if (num_columns == 1)  // Single strings column returns a copy
      return std::make_unique<column>(*(strings_columns.begin()), stream, mr);

    auto strings_count = strings_columns.num_rows();
    if (strings_count == 0)  // All columns have 0 rows
      return strings::detail::make_empty_strings_column(mr, stream);

    // Create device views from the strings columns.
    auto table       = table_device_view::create(strings_columns, stream);
    auto d_table     = *table;
    auto num_strings = num_columns * strings_count;

    std::pair<rmm::device_buffer, size_type> valid_mask{{}, 0};
    if (create_mask) {
      // Create resulting null mask
      valid_mask = cudf::detail::valid_if(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(num_strings),
        [num_columns, d_table] __device__(size_type idx) {
          auto source_row_idx = idx % num_columns;
          auto source_col_idx = idx / num_columns;
          return !d_table.column(source_row_idx).is_null(source_col_idx);
        },
        stream,
        mr);
    }

    auto const null_count = valid_mask.second;

    // Build offsets column by computing sizes of each string in the output
    auto offsets_transformer = [num_columns, d_table] __device__(size_type idx) {
      // First compute the column and the row this item belongs to
      auto source_row_idx = idx % num_columns;
      auto source_col_idx = idx / num_columns;
      return d_table.column(source_row_idx).is_valid(source_col_idx)
               ? d_table.column(source_row_idx).element<string_view>(source_col_idx).size_bytes()
               : 0;
    };
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0), offsets_transformer);
    auto offsets_column = strings::detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + num_strings, mr, stream);
    auto d_results_offsets = offsets_column->view().template data<int32_t>();

    // Create the chars column
    size_type bytes = thrust::device_pointer_cast(d_results_offsets)[num_strings];
    auto chars_column =
      strings::detail::create_chars_child_column(num_strings, null_count, bytes, mr, stream);
    // Fill the chars column
    auto d_results_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<size_type>(0),
      num_strings,
      [num_columns, d_table, d_results_offsets, d_results_chars] __device__(size_type idx) {
        auto source_row_idx = idx % num_columns;
        auto source_col_idx = idx / num_columns;

        // Do not write to buffer if the column value for this row is null
        if (d_table.column(source_row_idx).is_null(source_col_idx)) return;

        size_type offset = d_results_offsets[idx];
        char* d_buffer   = d_results_chars + offset;
        strings::detail::copy_string(
          d_buffer, d_table.column(source_row_idx).element<string_view>(source_col_idx));
      });

    return make_strings_column(num_strings,
                               std::move(offsets_column),
                               std::move(chars_column),
                               null_count,
                               std::move(valid_mask.first),
                               stream,
                               mr);
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>> operator()(
    table_view const& input,
    bool create_mask,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0)
  {
    auto arch_column = input.column(0);
    auto output_size = input.num_columns() * input.num_rows();
    auto output =
      allocate_like(arch_column, output_size, mask_allocation_policy::NEVER, mr, stream);
    auto device_input  = table_device_view::create(input);
    auto device_output = mutable_column_device_view::create(*output);
    auto index_begin   = thrust::make_counting_iterator<size_type>(0);
    auto index_end     = thrust::make_counting_iterator<size_type>(output_size);

    using Type = device_storage_type_t<T>;

    auto func_value = [input   = *device_input,
                       divisor = input.num_columns()] __device__(size_type idx) {
      return input.column(idx % divisor).element<Type>(idx / divisor);
    };

    if (not create_mask) {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        index_begin,
                        index_end,
                        device_output->begin<Type>(),
                        func_value);

      return output;
    }

    auto func_validity = [input   = *device_input,
                          divisor = input.num_columns()] __device__(size_type idx) {
      return input.column(idx % divisor).is_valid(idx / divisor);
    };

    thrust::transform_if(rmm::exec_policy(stream)->on(stream),
                         index_begin,
                         index_end,
                         device_output->begin<Type>(),
                         func_value,
                         func_validity);

    rmm::device_buffer mask;
    size_type null_count;

    std::tie(mask, null_count) = valid_if(index_begin, index_end, func_validity, stream, mr);

    output->set_null_mask(std::move(mask), null_count);

    return output;
  }
};

}  // anonymous namespace
}  // namespace detail

std::unique_ptr<column> interleave_columns(table_view const& input,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input.num_columns() > 0, "input must have at least one column to determine dtype.");

  auto const dtype = input.column(0).type();

  CUDF_EXPECTS(std::all_of(std::cbegin(input),
                           std::cend(input),
                           [dtype](auto const& col) { return dtype == col.type(); }),
               "DTYPE mismatch");

  auto const output_needs_mask = std::any_of(
    std::cbegin(input), std::cend(input), [](auto const& col) { return col.nullable(); });

  return type_dispatcher(dtype, detail::interleave_columns_functor{}, input, output_needs_mask, mr);
}

}  // namespace cudf
