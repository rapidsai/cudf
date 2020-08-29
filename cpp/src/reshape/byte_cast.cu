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

#include <cudf/detail/reshape.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include "cudf/replace.hpp"
#include "cudf/strings/detail/utilities.cuh"
#include "cudf/types.hpp"
#include "cudf/utilities/traits.hpp"
#include "thrust/for_each.h"
// #include <thrust/tabulate.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {
    
std::unique_ptr<column>  byte_cast(column_view const& input_column,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
//   cast(std::unique_ptr<column> &&input_column)
// {
//    // strings would need a specialization but it's also pretty simple
//    int num_output_elements = input_column.size() * size of input_columns type;
//    input_column->type = UINT8;
//    input->column->size = num_output_elements;
//    offsets = make_fixed_width_column_with_the_values(0, num_output_elements);
//    make_lists_column(1, std::move(offsets), std::move(input_column));
// }
// cast(std::unique_ptr<column> &&input_column)
// {
//    offsets = make_fixed_width_column_with_the_values(0, input_column.size());
//    make_lists_column(1, std::move(offsets), std::move(input_column));
// }
size_type num_output_elements = input_column.size() * cudf::size_of(input_column.type());

auto begin = thrust::make_constant_iterator(cudf::size_of(input_column.type()));
auto offsets_column = cudf::strings::detail::make_offsets_child_column(begin, begin + input_column.size(), mr, stream);
auto offsets_view  = offsets_column->view();
auto d_new_offsets = offsets_view.data<int32_t>();

auto byte_column = make_numeric_column(data_type{type_id::UINT8}, num_output_elements, mask_state::UNALLOCATED, stream, mr);
// strings::detail::create_chars_child_column(input_column->size(), 0, input_column->size() * 32, mr, stream);
auto bytes_view = byte_column->mutable_view();
auto d_chars    = bytes_view.data<char>();
//  offsets
//  return column_view{data_type{type_id::LIST}, input_column->size(), std::copy(), input_column}

column_view normalized = input_column;
if(is_floating_point(input_column.type()))
   normalized = normalize_nans_and_zeros(input_column, mr)->view();
   

thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_output_elements),
    [d_chars, data=input_column.data<char>(), mask = (cudf::size_of(input_column.type()) - 1)] __device__(auto byte_index) {
        size_type offset = byte_index & mask;
        int base_index = byte_index - offset;
        d_chars[byte_index] = data[base_index + mask - offset];
    });


thrust::tabulate(
    rmm::exec_policy(stream)->on(stream),
    bytes_view.begin<char>(),
    bytes_view.end<char>(),
    [data=normalized.data<char>(), mask = (cudf::size_of(input_column.type()) - 1)] __device__(auto byte_index) {
        size_type offset = byte_index & mask;
        int base_index = byte_index - offset;
        return data[base_index + mask - offset];\
    });

// actual change of underlying data
rmm::device_buffer null_mask = copy_bitmask(input_column, stream, mr);

return make_lists_column(
input_column.size(),
std::move(offsets_column),
std::move(byte_column),
input_column.null_count(),
std::move(null_mask),
stream,
mr);


//  std::move(chars_column),
//  0,
//  std::move(null_mask),
//  stream,
//  mr);

//  return column_view{type,
//                  input._size,
//                  input._data,
//                  input._null_mask,
//                  input._null_count,
//                  input._offset,
//                  input._children};
}
}
