/*
 * Copyright (c) 2021, Baidu CORPORATION.
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
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/detail/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/optional.h>

namespace cudf {
namespace strings {
namespace detail {

// first call json_to_array kernel for pre-compute number size
struct pre_compute_size {
  size_t num_rows; 
  thrust::optional<size_t> key_output_len;
  thrust::optional<size_t> value_output_len;
  size_t num_lists;
};

struct json_array_output {
  size_t output_max_len;
  thrust::optional<size_t> output_len;
  size_t num_rows;
  char* output; 
  int32_t* offset;
  int32_t* list_offsets;

  __device__ void add_output(const char* str, size_t len)
  {
    if (output != nullptr) {
      // assert output_len + len < output_max_len
      memcpy(output + output_len.value_or(0), str, len);
    }
    output_len = output_len.value_or(0) + len;
    num_rows++;
  }
  __device__ void add_output(string_view const& str) { add_output(str.data(), str.size_bytes()); }

};

__device__ parse_result parse_json_array(json_state& j_state,
                                                json_array_output& key_output,
                                                json_array_output& value_output,
                                                size_t& num_lists
                                                )
{
  parse_result last_result = parse_result::SUCCESS;
  while (1) {
    PARSE_TRY(j_state.next_element());
    json_element_type json_type = j_state.get_cur_el_type();
    switch(json_type) {
      case ARRAY: {
        size_type num_element = 0;
        json_state child(j_state);
        PARSE_TRY(child.child_element(ARRAY));
        // array loop
        while (last_result == parse_result::SUCCESS) {
          PARSE_TRY(child.child_element(OBJECT));
          string_view value;
          num_lists++;
          // obj loop
          while (last_result == parse_result::SUCCESS) {
            if (child.parse_value_extern(value) == parse_result::SUCCESS) {
              string_view key = child.get_cur_el_name();
              key_output.add_output({key.data(), static_cast<size_type>(key.size_bytes())});
              key_output.offset[num_element] = key.size_bytes();
              value_output.add_output({value.data(), static_cast<size_type>(value.size_bytes())});
              value_output.offset[num_element++] = value.size_bytes();
            }
             // next element
            PARSE_TRY(child.next_element());
          }
          if (child.next_obj() == parse_result::SUCCESS){
            last_result = parse_result::SUCCESS;
          }
        }
      } break;
      case OBJECT: {
        num_lists++;
        size_type num_element = 0;
        json_state child(j_state);
        PARSE_TRY(child.child_element(OBJECT));
        while (last_result == parse_result::SUCCESS) {
          string_view value;
          if (child.parse_value_extern(value) == parse_result::SUCCESS) {
            string_view key = child.get_cur_el_name();
            key_output.add_output({key.data(), static_cast<size_type>(key.size_bytes())});
            key_output.offset[num_element] = key.size_bytes();
            value_output.add_output({value.data(), static_cast<size_type>(value.size_bytes())});
            value_output.offset[num_element++] = value.size_bytes();
          }
          // next element
          PARSE_TRY(child.next_element());
        }
      } break;
      default: return parse_result::ERROR;
    }
    if (last_result == parse_result::EMPTY) break;
  }
  return parse_result::SUCCESS;
}

__device__ pre_compute_size json_to_array_kernel_impl(
  char const* input,
  size_t input_len,
  char* key_buffer,
  char* value_buffer,
  int32_t* key_offset,
  int32_t* value_offset,
  int32_t* lists_offset,
  size_t key_buf_size,
  size_t value_buf_size)
{

    size_t key_num_rows = 0;
    size_t value_num_rows = 0;
    size_t num_lists = 0;
    json_array_output key{key_buf_size, 0, key_num_rows, key_buffer, key_offset, lists_offset};
    json_array_output value{value_buf_size, 0, value_num_rows, value_buffer, value_offset, lists_offset};
    json_state j_state(input, input_len);
    
    parse_json_array(j_state, key, value, num_lists);
    return {key.num_rows, key.output_len.value_or(0), value.output_len.value_or(0), num_lists};
}

__global__ void json_to_array_kernel(char const* chars,
                                     int32_t const* offsets,
                                     char* key_buf,
                                     char* value_buf,
                                     int32_t* key_offset,
                                     int32_t* value_offset,
                                     int32_t* lists_offset,
                                     size_t key_buf_size,
                                     size_t value_buf_size,
                                     int32_t* output_offsets)
{
  uint64_t const tid = threadIdx.x + (blockDim.x * blockIdx.x);
  pre_compute_size for_pre_compute = json_to_array_kernel_impl(
                                                    chars + offsets[tid], // 第tid个数据
                                                    offsets[tid + 1] - offsets[tid],
                                                    key_buf,
                                                    value_buf,
                                                    key_offset,
                                                    value_offset,
                                                    lists_offset,
                                                    key_buf_size,
                                                    value_buf_size);
  if (output_offsets != nullptr) { 
    output_offsets[tid] = static_cast<int32_t>(for_pre_compute.num_rows); 
    key_offset[tid] = static_cast<int32_t>(for_pre_compute.key_output_len.value_or(0));
    value_offset[tid] = static_cast<int32_t>(for_pre_compute.value_output_len.value_or(0));
    lists_offset[tid] = static_cast<int32_t>(for_pre_compute.num_lists);
  }
}

std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // for num_rows
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view offsets_view(*offsets);
  // for key ouput size
  auto key_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view key_offsets_view(*key_offsets_column);
  // for value output size
  auto value_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view value_offsets_view(*value_offsets_column);
  // for lists size
  auto list_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view lists_offsets_view(*list_offsets_column);

  cudf::detail::grid_1d const grid{1, col.size()};

  // pre-compute for calculate the key column size and value column size
  json_to_array_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      col.chars().head<char>(),
      col.offsets().head<int32_t>(),
      nullptr,
      nullptr,
      key_offsets_view.head<int32_t>(),
      value_offsets_view.head<int32_t>(),
      lists_offsets_view.head<int32_t>(),
      0,
      0,
      offsets_view.head<int32_t>());

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for num_rows
                         offsets_view.head<int32_t>(),
                         offsets_view.head<int32_t>() + col.size() + 1,
                         offsets_view.head<int32_t>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for key column size and key offset column  
                         key_offsets_view.head<int32_t>(),
                         key_offsets_view.head<int32_t>() + col.size() + 1,
                         key_offsets_view.head<int32_t>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for value column size and value offset column
                         value_offsets_view.head<int32_t>(),
                         value_offsets_view.head<int32_t>() + col.size() + 1,
                         value_offsets_view.head<int32_t>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for lists column size and lists offset
                         lists_offsets_view.head<int32_t>(),
                         lists_offsets_view.head<int32_t>() + col.size() + 1,
                         lists_offsets_view.head<int32_t>(),
                         0);
                                
  cudf::size_type num_rows = cudf::detail::get_value<int32_t>(offsets_view, col.size(), stream);
  cudf::size_type key_output_size = cudf::detail::get_value<int32_t>(key_offsets_view, col.size(), stream);
  cudf::size_type value_output_size = cudf::detail::get_value<int32_t>(value_offsets_view, col.size(), stream);
  cudf::size_type lists_size = cudf::detail::get_value<int32_t>(lists_offsets_view, col.size(), stream);

  auto key_chars_column = cudf::make_numeric_column(data_type{type_id::INT8}, 
                            key_output_size, mask_state::UNALLOCATED, stream, mr);

  auto value_chars_column = cudf::make_numeric_column(data_type{type_id::INT8}, 
                            value_output_size, mask_state::UNALLOCATED, stream, mr);

  key_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  value_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  list_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, lists_size + 1, mask_state::UNALLOCATED, stream, mr);

  // compute results
  
  json_to_array_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      col.chars().head<char>(),
      col.offsets().head<int32_t>(),
      key_chars_column->mutable_view().head<char>(),
      value_chars_column->mutable_view().head<char>(),
      key_offsets_column->mutable_view().head<int32_t>(),
      value_offsets_column->mutable_view().head<int32_t>(),
      list_offsets_column->mutable_view().head<int32_t>(),
      key_output_size,
      value_output_size,
      nullptr);

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),  // list column offset
                         list_offsets_column->mutable_view().head<int32_t>(),
                         list_offsets_column->mutable_view().head<int32_t>() + lists_size + 1,
                         list_offsets_column->mutable_view().head<int32_t>(),
                         0);

  thrust::exclusive_scan(rmm::exec_policy(stream),  // key column offset
                         key_offsets_column->mutable_view().head<int32_t>(),
                         key_offsets_column->mutable_view().head<int32_t>() + num_rows + 1,
                         key_offsets_column->mutable_view().head<int32_t>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // value column offset
                         value_offsets_column->mutable_view().head<int32_t>(),
                         value_offsets_column->mutable_view().head<int32_t>() + num_rows + 1,
                         value_offsets_column->mutable_view().head<int32_t>(),
                         0);
  
  auto key_col = cudf::make_strings_column(num_rows,
                                           std::move(key_offsets_column),
                                           std::move(key_chars_column),
                                           UNKNOWN_NULL_COUNT,
                                           rmm::device_buffer{},
                                           stream,
                                           mr);
  auto value_col = cudf::make_strings_column(num_rows,
                                             std::move(value_offsets_column),
                                             std::move(value_chars_column),
                                             UNKNOWN_NULL_COUNT,
                                             rmm::device_buffer{},
                                             stream,
                                             mr);

  std::vector<std::unique_ptr<cudf::column>> col_output;
  col_output.push_back(std::move(key_col));
  col_output.push_back(std::move(value_col));

  auto out_struct = make_structs_column(num_rows,
                                       std::move(col_output),
                                       0,
                                       rmm::device_buffer{},
                                       stream,
                                       mr);

  return make_lists_column(lists_size,
                           std::move(list_offsets_column),
                           std::move(out_struct), 
                           0, 
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}

}  // namespace detail

/**
 * @copydoc cudf::strings::json_to_array
 */
std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                            rmm::mr::device_memory_resource* mr)
{
  return detail::json_to_array(col, 0, mr);
}

}  // namespace strings
}  // namespace cudf 
