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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Result of calling a parse function.
 *
 * The primary use of this is to distinguish between "success" and
 * "success but no data" return cases.  For example, if you are reading the
 * values of an array you might call a parse function in a while loop. You
 * would want to continue doing this until you either encounter an error (parse_result::ERROR)
 * or you get nothing back (parse_result::EMPTY)
 */
enum class parse_result {
  ERROR,    // failure
  SUCCESS,  // success
  EMPTY,    // success, but no data
};

/**
 * @brief Base parser class inherited by the (device-side) json_state class...
 * 
 * reused get_json_object parser: https://github.com/rapidsai/cudf/pull/7286
 */
class parser {
  protected:
  CUDA_HOST_DEVICE_CALLABLE parser() : input(nullptr), input_len(0), pos(nullptr) {}
  CUDA_HOST_DEVICE_CALLABLE parser(const char* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }
  CUDA_HOST_DEVICE_CALLABLE bool eof(const char* p) { return p - input >= input_len; }
  CUDA_HOST_DEVICE_CALLABLE bool eof() { return eof(pos); }

  CUDA_HOST_DEVICE_CALLABLE bool parse_whitespace()
  {
    while (!eof()) {
      if (is_whitespace(*pos)) {
        pos++;
      } else {
        return true;
      }
    }
    return false;
  }

  CUDA_HOST_DEVICE_CALLABLE parse_result parse_string(string_view& str, bool can_be_empty, char quote)
  {
    str = string_view(nullptr, 0);

    if (parse_whitespace() && *pos == quote) {
        const char* start = ++pos;
        while (!eof()) {
          if (*pos == quote) {
            str = string_view(start, pos - start);
            pos++;
            return parse_result::SUCCESS;
          }
          pos++;
        }
    }
    return can_be_empty ? parse_result::EMPTY : parse_result::ERROR;
  }

  // a name means:
  // - a string followed by a :
  // - no string
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_name(string_view& name,
                                                    bool can_be_empty,
                                                    char quote)
  {
    if (parse_string(name, can_be_empty, quote) == parse_result::ERROR) {
      return parse_result::ERROR;
    }

    // if we got a real string, the next char must be a :
    if (name.size_bytes() > 0) {
      if (!parse_whitespace()) { return parse_result::ERROR; }
      if (*pos == ':') {
        pos++;
        return parse_result::SUCCESS;
      }
    }
    return parse_result::EMPTY;
  }

  protected:
  char const* input;
  int64_t input_len;
  char const* pos;

  private:
  CUDA_HOST_DEVICE_CALLABLE bool is_whitespace(char c)
  {
    return (c == ' ' || c == '\r' || c == '\n' || c == '\t') ;
  }
};

// first call json_to_array kernel for pre-compute number size
struct pre_compute_size {
  size_t num_rows; 
  thrust::optional<size_t> key_output_len;
  thrust::optional<size_t> value_output_len;
  size_t num_lists;
};

enum first_operator_type { NONE, OBJECT, ARRAY };

struct json_first_operator {
  first_operator_type type;
  string_view name;
};

struct json_array_output {
  size_t output_max_len;
  thrust::optional<size_t> output_len;
  size_t num_rows;
  char* output; 
  offset_type* offset;
  offset_type* list_offsets;

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

class json_state : private parser {
 public:
  __device__ json_state()
    : parser(), element(first_operator_type::NONE), cur_el_start(nullptr), num_element(0)
  {
  }
  __device__ json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len), element(first_operator_type::NONE), cur_el_start(nullptr), num_element(0)
  {
  }
  __device__ json_first_operator get_first_character()
  {
    char c = *pos++;
    switch (c) {
      case '{': {
        json_first_operator op{OBJECT, {"", 0}};
        string_view j_obj{"{", 1};
        op.name = j_obj;
        op.type = OBJECT;
        element = OBJECT;
        return op;
      } break;
      case '[': {
        json_first_operator op{ARRAY, {"", 0}};
        string_view j_arr{"[", 1};
        op.name = j_arr;
        op.type = ARRAY;
        element = ARRAY;
        return op;
      } break;
      default: break;
    }
    return {NONE};
  }
  __device__ char const* find_char_pos(const char ch)
  {
    char const* idx = pos;
    while (*idx != ch && !eof(idx))  idx++;
    return idx;
  }

  // value's last validity position forward 1 byte position
  __device__ char const* find_last_validity_pos()
  {
    char const* last_validity = pos;
    while ((*last_validity >= '0' && *last_validity <= '9') 
          || *last_validity == '.' || *last_validity == '+' 
          || *last_validity == '-')  
      last_validity++;
    return last_validity;
  }

  // extract key string and add to ouput 
  __device__ parse_result extract_key(json_array_output& output)
  {
    string_view key;
    if (element == ARRAY && parse_whitespace()) {
      if (*pos == '{')  
        ++pos;
    }
    if (parse_name(key, false, '\"') == parse_result::SUCCESS) {
      output.add_output({key.data(), static_cast<size_type>(key.size_bytes())});
      output.offset[num_element] = key.size_bytes();
      return parse_result::SUCCESS;
    }
    return parse_result::ERROR;
  }

  // extract value string and add to output 
  __device__ parse_result extract_value(json_array_output& output, 
                                       bool& finish_obj, bool& finish_arr)
  {
    string_view value;
    parse_whitespace();
    char c = *pos;
    if (c == '\"') {
      auto start = ++pos;
      char const* end = find_char_pos('\"');
      value = string_view(start, end - start);
      pos = ++end;
      parse_whitespace();
      char const* comma_pos = find_char_pos(',');
      if (*pos == '}' && !eof(pos))
      {
        finish_obj = true;
        finish_arr = false;
      }
      else if (eof(pos))
      {
        finish_obj = true;
        finish_arr = true;
      }
      else{
        finish_obj = false;
        finish_arr = false;
        pos = comma_pos;
      }      
      pos++;
    }
    else if (c == '{' || c == '[') {
      auto tmp_pos = pos;
      int obj_count = 0;
      int arr_count = 0;
      while (!eof(pos)) {
        char c = *pos++;
        switch (c) {
          case '{': obj_count++; break;
          case '}': obj_count--; break;
          case '[': arr_count++; break;
          case ']': arr_count--; break;
          default: break;
        }
        if (obj_count == 0 && arr_count == 0) break;
      }
      value = string_view(tmp_pos, pos - tmp_pos);
      parse_whitespace();
      char const* comma_pos = find_char_pos(',');
      pos = comma_pos;
      if (eof(pos))
        finish_obj = true;
      pos++;
    }
    else if ((c >= '0' && c <= '9') || c == '+' || c == '-') {
      char const* comma_pos = find_char_pos(',');
      char const* last_value_pos = find_last_validity_pos();
      value = string_view(pos, last_value_pos - pos);
      pos = last_value_pos;
      parse_whitespace();
      if (*pos == ',' && comma_pos == pos) { //obj inner
        finish_obj = false;
        finish_arr = false;
        pos++;
      }
      else if (*pos == '}' && eof(comma_pos)) { //eof end
        finish_obj = true;
        finish_arr = true;
      }
      else if (*pos == '}'){ // obj end
        pos = comma_pos;
        pos++;
        finish_obj = true;
        finish_arr = false;
      }
    }
    else {
      return parse_result::ERROR;
    }
    output.add_output({value.data(), static_cast<size_type>(value.size_bytes())});
    output.offset[num_element++] = value.size_bytes();
    return parse_result::SUCCESS;
  }

  first_operator_type element;

  private:

  const char* cur_el_start;
  size_t num_element;
};

#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)

__device__ parse_result parse_json_array(json_state& j_state,
                                                json_array_output& key_output,
                                                json_array_output& value_output,
                                                size_t& num_lists
                                                )
{
  json_first_operator op = j_state.get_first_character();
  switch(op.type) {
    case OBJECT: {
      for (size_t i = 0 ; ; i++)
      {
        bool finish_obj = false;
        bool finish_arr = false; // not use in this case
        parse_result last_result = parse_result::SUCCESS;
        PARSE_TRY(j_state.extract_key(key_output));
        PARSE_TRY(j_state.extract_value(value_output, finish_obj, finish_arr));
        num_lists = 1;
        if (finish_obj) {
          key_output.list_offsets[0] = key_output.num_rows;
          break;
        }
      } 
    } break;
    case ARRAY: {
      for (size_t i = 0 ;  ; i++)
      {
        bool finish_arr = false;
        size_t j = 0;
        for ( ;  ; j++)
        {
          bool finish_obj = false;
          parse_result last_result = parse_result::SUCCESS;
          PARSE_TRY(j_state.extract_key(key_output));
          PARSE_TRY(j_state.extract_value(value_output, finish_obj, finish_arr));
          if (finish_obj) break;
        }
        num_lists++;
        key_output.list_offsets[i] = j + 1;
        if (finish_arr) break;
      }
    } break;
    case NONE: {
      return parse_result::ERROR;
    } break;
    default: break;
  }
  return parse_result::SUCCESS;
}

__device__ pre_compute_size json_to_array_kernel_impl(
  char const* input,
  size_t input_len,
  char* key_buffer,
  char* value_buffer,
  offset_type* key_offset,
  offset_type* value_offset,
  offset_type* lists_offset,
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
                                     offset_type const* offsets,
                                     char* key_buf,
                                     char* value_buf,
                                     offset_type* key_offset,
                                     offset_type* value_offset,
                                     offset_type* lists_offset,
                                     size_t key_buf_size,
                                     size_t value_buf_size,
                                     offset_type* output_offsets)
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
    output_offsets[tid] = static_cast<offset_type>(for_pre_compute.num_rows); 
    key_offset[tid] = static_cast<offset_type>(for_pre_compute.key_output_len.value_or(0));
    value_offset[tid] = static_cast<offset_type>(for_pre_compute.value_output_len.value_or(0));
    lists_offset[tid] = static_cast<offset_type>(for_pre_compute.num_lists);
  }
}

std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  size_t stack_size;
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, 4096);

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
      col.offsets().head<offset_type>(),
      nullptr,
      nullptr,
      key_offsets_view.head<offset_type>(),
      value_offsets_view.head<offset_type>(),
      lists_offsets_view.head<offset_type>(),
      0,
      0,
      offsets_view.head<offset_type>());

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for num_rows
                         offsets_view.head<offset_type>(),
                         offsets_view.head<offset_type>() + col.size() + 1,
                         offsets_view.head<offset_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for key column size and key offset column  
                         key_offsets_view.head<offset_type>(),
                         key_offsets_view.head<offset_type>() + col.size() + 1,
                         key_offsets_view.head<offset_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for value column size and value offset column
                         value_offsets_view.head<offset_type>(),
                         value_offsets_view.head<offset_type>() + col.size() + 1,
                         value_offsets_view.head<offset_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for lists column size and lists offset
                         lists_offsets_view.head<offset_type>(),
                         lists_offsets_view.head<offset_type>() + col.size() + 1,
                         lists_offsets_view.head<offset_type>(),
                         0);
                                
  cudf::size_type num_rows = cudf::detail::get_value<offset_type>(offsets_view, col.size(), stream);
  cudf::size_type key_output_size = cudf::detail::get_value<offset_type>(key_offsets_view, col.size(), stream);
  cudf::size_type value_output_size = cudf::detail::get_value<offset_type>(value_offsets_view, col.size(), stream);
  cudf::size_type lists_size = cudf::detail::get_value<offset_type>(lists_offsets_view, col.size(), stream);

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
      col.offsets().head<offset_type>(),
      key_chars_column->mutable_view().head<char>(),
      value_chars_column->mutable_view().head<char>(),
      key_offsets_column->mutable_view().head<offset_type>(),
      value_offsets_column->mutable_view().head<offset_type>(),
      list_offsets_column->mutable_view().head<offset_type>(),
      key_output_size,
      value_output_size,
      nullptr);

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),  // list column offset
                         list_offsets_column->mutable_view().head<offset_type>(),
                         list_offsets_column->mutable_view().head<offset_type>() + lists_size + 1,
                         list_offsets_column->mutable_view().head<offset_type>(),
                         0);

  thrust::exclusive_scan(rmm::exec_policy(stream),  // key column offset
                         key_offsets_column->mutable_view().head<offset_type>(),
                         key_offsets_column->mutable_view().head<offset_type>() + num_rows + 1,
                         key_offsets_column->mutable_view().head<offset_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // value column offset
                         value_offsets_column->mutable_view().head<offset_type>(),
                         value_offsets_column->mutable_view().head<offset_type>() + num_rows + 1,
                         value_offsets_column->mutable_view().head<offset_type>(),
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
  
  cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

  return make_lists_column(lists_size,
                           std::move(list_offsets_column),
                           std::move(out_struct), 
                           0, 
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}

}  // namespace
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