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
  EOOBJ,
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

  CUDA_HOST_DEVICE_CALLABLE parser(parser const& p)
    : input(p.input), input_len(p.input_len), pos(p.pos)
  {
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
  // parse string for neted type such as object or array
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_string_nested(string_view& str, bool can_be_empty, char quote)
  {
    str = string_view(nullptr, 0);

    const char* start = pos;
    int obj_count = 0;
    int arr_count = 0;
    while (!eof()) {
        char c = *pos++;
        switch (c) {
          case '{': obj_count++; break;
          case '}': obj_count--; break;
          case '[': arr_count++; break;
          case ']': arr_count--; break;
          default: break;
        }
        if (obj_count == 0 && arr_count == 0) {
          str = string_view(start, pos - start);
          pos++;
          return parse_result::SUCCESS;
        }
    }
    return parse_result::EMPTY;
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

  // numbers, true, false, null.
  // this function is not particularly strong. badly formed values will get
  // consumed without throwing any errors
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_non_string_value(string_view& val)
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // parse to the end of the value
    char const* start = pos;
    char const* end   = start;
    while (!eof(end)) {
      char const c = *end;
      if (c == ',' || c == '}' || c == ']' || is_whitespace(c)) { break; }

      // illegal chars
      if (c == '[' || c == '{' || c == ':' || c == '\"') { return parse_result::ERROR; }
      end++;
    }
    pos = end;

    val = string_view(start, end - start);

    return parse_result::SUCCESS;
  }

  protected:
  char const* input;
  int64_t input_len;
  char const* pos;

  private:
  CUDA_HOST_DEVICE_CALLABLE bool is_whitespace(char c) { return c == ' '; }
};

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

struct json_output {
  size_t output_max_len;
  char* output;
  thrust::optional<size_t> output_len;

  __device__ void add_output(const char* str, size_t len)
  {
    if (output != nullptr) { memcpy(output + output_len.value_or(0), str, len); }
    output_len = output_len.value_or(0) + len;
  }

  __device__ void add_output(string_view const& str) { add_output(str.data(), str.size_bytes()); }
};

enum json_element_type { NONE, OBJECT, ARRAY, VALUE };

/**
 * @brief Parsing class that holds the current state of the json to be parse and provides
 * functions for navigating through it.
 */
class json_state : private parser {
 public:
  __device__ json_state()
    : parser(),
      cur_el_start(nullptr),
      cur_el_type(json_element_type::NONE),
      parent_el_type(json_element_type::NONE)
  {
  }
  __device__ json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len),
      cur_el_start(nullptr),
      cur_el_type(json_element_type::NONE),
      parent_el_type(json_element_type::NONE)
  {
  }

  __device__ json_state(json_state const& j)
    : parser(j),
      cur_el_start(j.cur_el_start),
      cur_el_type(j.cur_el_type),
      parent_el_type(j.parent_el_type)
  {
  }

  // retrieve the entire current element into the output
  __device__ parse_result extract_element(json_output* output, bool list_element)
  {
    char const* start = cur_el_start;
    char const* end   = start;
    // if we're a value type, do a simple value parse.
    if (cur_el_type == VALUE) {
      pos = cur_el_start;
      if (parse_value() != parse_result::SUCCESS) { return parse_result::ERROR; }
      end = pos;

      // SPARK-specific behavior.  if this is a non-list-element wrapped in quotes,
      // strip them. we may need to make this behavior configurable in some way
      // later on.
      if (!list_element && *start == '\"' && *(end - 1) == '\"') {
        start++;
        end--;
      }
    }
    // otherwise, march through everything inside
    else {
      int obj_count = 0;
      int arr_count = 0;

      while (!eof(end)) {
        // could do some additional checks here. we know our current
        // element type, so we could be more strict on what kinds of
        // characters we expect to see.
        switch (*end++) {
          case '{': obj_count++; break;
          case '}': obj_count--; break;
          case '[': arr_count++; break;
          case ']': arr_count--; break;
          default: break;
        }
        if (obj_count == 0 && arr_count == 0) { break; }
      }
      if (obj_count > 0 || arr_count > 0) { return parse_result::ERROR; }
      pos = end;
    }
    // parse trailing ,
    if (parse_whitespace()) {
      if (*pos == ',') { pos++; }
    }
    if (output != nullptr) { output->add_output({start, static_cast<size_type>(end - start)}); }
    return parse_result::SUCCESS;
  }

  // skip the next element
  __device__ parse_result skip_element() { return extract_element(nullptr, false); }

  // advance to the next element
  __device__ parse_result next_element() { return next_element_internal(false); }

  // advance inside the current element
  __device__ parse_result child_element(json_element_type expected_type)
  {
    
    if (expected_type != NONE && cur_el_type != expected_type) { cur_el_type);return parse_result::ERROR; }
    // if we succeed, record our parent element type.
    auto const prev_el_type = cur_el_type;
    auto const result       = next_element_internal(true);
    
    if (result == parse_result::SUCCESS) { parent_el_type = prev_el_type; }
    return result;
  }

  // return the next element that matches the specified name.
  __device__ parse_result next_matching_element(string_view const& name, bool inclusive)
  {
    // if we're not including the current element, skip it
    if (!inclusive) {
      parse_result result = next_element_internal(false);
      if (result != parse_result::SUCCESS) { return result; }
    }
    // loop until we find a match or there's nothing left
    do {
      // wildcard matches anything
      if (name.size_bytes() == 1 && name.data()[0] == '*') {
        return parse_result::SUCCESS;
      } else if (cur_el_name == name) {
        return parse_result::SUCCESS;
      }

      // next
      parse_result result = next_element_internal(false);
      if (result != parse_result::SUCCESS) { return result; }
    } while (1);

    return parse_result::ERROR;
  }

  __device__ json_element_type get_cur_el_type() { return cur_el_type; }
  __device__ string_view get_cur_el_name() { return cur_el_name; }
  __device__ parse_result next_obj() {
    while(!eof()) {
      pos++;
      if (*pos == '\"') { 
        cur_el_type = OBJECT;
        return parse_result::SUCCESS;
      }
    }
    return parse_result::EMPTY;
  }
    // parse a value - either a string or a number/null/bool
  __device__ parse_result parse_value_extern(string_view& str)
  {
    parse_result result = parse_result::SUCCESS;
    if (!parse_whitespace()) { return parse_result::ERROR; }
    // back one valid character, may include multiples whitespaces.
    do{
      --pos;
    }while (*pos == ' ');
    if (*pos == '+' || *pos == '-' || (*pos >= '0' && *pos <= '9')){
      result = parse_non_string_value(str);
    }
    else if (*pos == '\"')
      result = parse_string(str, false, '\"');
    else if (*pos == '{')
      result = parse_string_nested(str, false, '{');
    else if (*pos == '[')
      result = parse_string_nested(str, false, '[');
    else
      result = parse_result::ERROR;
    
    return result;
  }

 private:

  // parse a value - either a string or a number/null/bool
  __device__ parse_result parse_value()
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // string or number?
    string_view unused;
    return *pos == '\"' ? parse_string(unused, false, '\"') : parse_non_string_value(unused);
  }

  __device__ parse_result next_element_internal(bool child)
  {
    // if we're not getting a child element, skip the current element.
    // this will leave pos as the first character -after- the close of
    // the current element
    if (!child && cur_el_start != nullptr) {
      if (skip_element() == parse_result::ERROR) { return parse_result::ERROR; }
      cur_el_start = nullptr;
    }
    // otherwise pos will be at the first character within the current element

    // can only get the child of an object or array.
    // this could theoretically be handled as an error, but the evaluators I've found
    // seem to treat this as "it's nothing"
    if (child && (cur_el_type == VALUE || cur_el_type == NONE)) { return parse_result::EMPTY; }

    // what's next
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    // if we're closing off a parent element, we're done
    char const c = *pos;
    if (c == ']' || c == '}') { return parse_result::EMPTY; }

    // if we're not accessing elements of an array, check for name.
    bool const array_access =
      (cur_el_type == ARRAY && child) || (parent_el_type == ARRAY && !child);
    
    if (!array_access && parse_name(cur_el_name, true, '\"') == parse_result::ERROR) {
      return parse_result::ERROR;
    }
    // element type
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    switch (*pos++) {
      case '[': cur_el_type = ARRAY; break;
      case '{': cur_el_type = OBJECT; break;

      case ',':
      case ':':
      case '\'': return parse_result::ERROR;

      // value type
      default: cur_el_type = VALUE; break;
    }
    
    // the start of the current element is always at the value, not the name
    cur_el_start = pos - 1;
    return parse_result::SUCCESS;
  }

  const char* cur_el_start;          // pointer to the first character of the -value- of the current
                                     // element - not the name
  string_view cur_el_name;           // name of the current element (if applicable)
  json_element_type cur_el_type;     // type of the current element
  json_element_type parent_el_type;  // parent element type
};


#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)

__device__ parse_result get_array_result(json_state& j_state,
                                                json_array_output& key_output,
                                                json_array_output& value_output,
                                                size_t& num_lists)
{
  parse_result last_result = parse_result::SUCCESS;
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
}

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
