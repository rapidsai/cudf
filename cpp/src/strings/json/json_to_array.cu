#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf {

class parser {
  protected:
  CUDA_HOST_DEVICE_CALLABLE parser() : input(nullptr), input_len(0), pos(nullptr) {}
  CUDA_HOST_DEVICE_CALLABLE parser(const char* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }
  CUDA_HOST_DEVICE_CALLABLE bool parse_whitespace()
  {
    while (!eof()) {
      char c = *pos;
      if (c == ' ' || c == '\r' || c == '\n' || c == '\t') {
        pos++;
      } else {
        return true;
      }
    }
    return false;
  }

  CUDA_HOST_DEVICE_CALLABLE bool eof(const char* p) { return p - input >= input_len; }

  CUDA_HOST_DEVICE_CALLABLE bool eof() { return eof(pos); }

  protected:
  char const* input;
  int64_t input_len;
  char const* pos;
};

CUDA_HOST_DEVICE_CALLABLE bool device_strncmp(const char* str1, const char* str2, size_t num_chars)
{
  for (size_t idx = 0; idx < num_chars; idx++) {
    if (str1[idx] != str2[idx]) { return false; }
  }
  return true;
}


CUDA_HOST_DEVICE_CALLABLE bool is_digit(char c)
{
  if (c >= '0' && c <= '9') return true;
  return false;
}

struct json_string {
  const char* str;
  int64_t len;

  CUDA_HOST_DEVICE_CALLABLE bool operator==(json_string const& cmp)
  {
    return len == cmp.len && str != nullptr && cmp.str != nullptr &&
           device_strncmp(str, cmp.str, static_cast<size_t>(len));
  }
};

struct pre_compute_size {
  size_t num_rows; 
  size_t key_output_len;
  size_t value_output_len;
  size_t num_lists;
};

enum first_operator_type { NONE, OBJECT, ARRAY };
struct json_first_operator {
  first_operator_type type;
  json_string name;
  int index;
};



struct json_array_output {
  size_t output_max_len;
  size_t output_len;
  size_t num_rows;
  char* output; 
  size_type* offset;
  size_type* list_offsets;

  CUDA_HOST_DEVICE_CALLABLE void add_output(const char* str, size_t len)
  {
    if (output != nullptr) {
      // assert output_len + len < output_max_len
      memcpy(output + output_len, str, len);
    }
    output_len += len;
    num_rows++;
  }

  CUDA_HOST_DEVICE_CALLABLE void add_output(json_string str) { add_output(str.str, str.len); }
};

class json_state : private parser {
 public:
  CUDA_HOST_DEVICE_CALLABLE json_state()
    : parser(), element(first_operator_type::NONE), cur_el_start(nullptr)
  {
  }
  CUDA_HOST_DEVICE_CALLABLE json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len), element(first_operator_type::NONE), cur_el_start(nullptr)
  {
  }
  CUDA_HOST_DEVICE_CALLABLE json_first_operator get_first_character()
  {
    char c = parse_char();
    switch (c) {
      case '{': {
        json_first_operator op;
        json_string j_obj{"{", 1};
        op.name = j_obj;
        op.type = OBJECT;
        element = OBJECT;
        return op;
      } break;
      case '[': {
        json_first_operator op;
        json_string j_arr{"[", 1};
        op.name = j_arr;
        op.type = ARRAY;
        element = ARRAY;
        return op;
      }
      default: break;
    }
    return {NONE};
  }

  CUDA_HOST_DEVICE_CALLABLE json_string extract_key()
  {
    json_string key;
    if (element == ARRAY) {
      parse_whitespace();
      if (*pos == '{')
        pos++;
    }
    parse_string(key, true);
    return key;
  }

  CUDA_HOST_DEVICE_CALLABLE char const* find_char_pos(const char ch)
  {
    char const* idx = pos;
    while (*idx != ch && !eof(idx))  idx++;
    return idx;
  }
  CUDA_HOST_DEVICE_CALLABLE char const* find_last_validity_pos()
  {
    char const* last_validity = pos;
    while (is_digit(*last_validity) || *last_validity == '.')  last_validity++;
    return last_validity;
  }
  CUDA_HOST_DEVICE_CALLABLE json_string extract_value(bool& finish_obj, bool& finish_arr)
  {
    json_string value;
    parse_whitespace();
    if (*pos != ':') {
      printf("Error json format is't colon character.\n");
    }
    pos++;
    parse_whitespace();
    char c = *pos;
    if (c == '\"') {
      value.str = ++pos;
      char const* quo_pos = find_char_pos('\"');
      value.len = quo_pos - value.str;
      pos = ++quo_pos;
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
      value.str = pos;
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
      value.len = pos - value.str;
      parse_whitespace();
      char const* comma_pos = find_char_pos(',');
      pos = comma_pos;
      if (eof(pos))
        finish_obj = true;
      pos++;
    }
    else if (is_digit(c)) {
      value.str = pos;
      char const* comma_pos = find_char_pos(',');
      char const* last_value_pos = find_last_validity_pos();
      value.len = last_value_pos - pos;
      pos = ++last_value_pos;
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
      printf("Error when extract value from json foramt.\n");
    }
    return value;
  }
  CUDA_HOST_DEVICE_CALLABLE char parse_char() { return *pos++; }

  first_operator_type element;

  private:
  CUDA_HOST_DEVICE_CALLABLE bool parse_string(json_string& str, bool can_be_empty)
  {
    str.str = nullptr;
    str.len = 0;

    if (parse_whitespace()) {
      if (*pos == '\"') {
        const char* start = ++pos;
        while (!eof()) {
          if (*pos == '\"') {
            str.str = start;
            str.len = pos - start;
            pos++;
            return true;
          }
          pos++;
        }
      }
    }
    return can_be_empty ? true : false;
  }

  const char* cur_el_start;
};

namespace strings {
namespace detail {

namespace {
using namespace cudf;

CUDA_HOST_DEVICE_CALLABLE void parse_json_array(json_state& j_state,
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

        json_string key = j_state.extract_key();
        json_string value = j_state.extract_value(finish_obj, finish_arr);

        key_output.add_output(key);
        value_output.add_output(value);

        key_output.offset[i] = key.len;
        value_output.offset[i] = value.len;

        num_lists = 1;
        if (finish_obj) {
          key_output.list_offsets[0] = key_output.num_rows;
          break;
        }
      } 
    } break;
    case ARRAY: {
      size_t idx = 0;
      for (size_t i = 0 ;  ; i++)
      {
        bool finish_arr = false;
        size_t j = 0;
        for ( ;  ; j++)
        {
          bool finish_obj = false;
          json_string key = j_state.extract_key();
          json_string value = j_state.extract_value(finish_obj, finish_arr);

          key_output.add_output(key);
          value_output.add_output(value);

          key_output.offset[idx] = key.len;
          value_output.offset[idx] = value.len;

          idx++;
          if (finish_obj) break;
        }
        num_lists++;
        key_output.list_offsets[i] = j + 1;
        if (finish_arr) break;
      }
    } break;
    case NONE: {

    } break;

    default: break;
  }
}

CUDA_HOST_DEVICE_CALLABLE pre_compute_size json_to_array_kernel_impl(
  char const* input,
  size_t input_len,
  char* key_buffer,
  char* value_buffer,
  size_type* key_offset,
  size_type* value_offset,
  size_type* lists_offset,
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
    return {key.num_rows, key.output_len, value.output_len, num_lists};
}

__global__ void json_to_array_kernel(char const* chars,
                                     size_type const* offsets,
                                     char* key_buf,
                                     char* value_buf,
                                     size_type* key_offset,
                                     size_type* value_offset,
                                     size_type* lists_offset,
                                     size_t key_buf_size,
                                     size_t value_buf_size,
                                     size_type* output_offsets)
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
    output_offsets[tid] = static_cast<size_type>(for_pre_compute.num_rows); 
    key_offset[tid] = static_cast<size_type>(for_pre_compute.key_output_len);
    value_offset[tid] = static_cast<size_type>(for_pre_compute.value_output_len);
    lists_offset[tid] = static_cast<size_type>(for_pre_compute.num_lists);
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
      col.offsets().head<size_type>(),
      nullptr,
      nullptr,
      key_offsets_view.head<size_type>(),
      value_offsets_view.head<size_type>(),
      lists_offsets_view.head<size_type>(),
      0,
      0,
      offsets_view.head<size_type>());

  thrust::exclusive_scan(rmm::exec_policy(stream),  // for num_rows
                         offsets_view.head<size_type>(),
                         offsets_view.head<size_type>() + col.size() + 1,
                         offsets_view.head<size_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for key column size and key offset column  
                         key_offsets_view.head<size_type>(),
                         key_offsets_view.head<size_type>() + col.size() + 1,
                         key_offsets_view.head<size_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for value column size and value offset column
                         value_offsets_view.head<size_type>(),
                         value_offsets_view.head<size_type>() + col.size() + 1,
                         value_offsets_view.head<size_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // for lists column size and lists offset
                         lists_offsets_view.head<size_type>(),
                         lists_offsets_view.head<size_type>() + col.size() + 1,
                         lists_offsets_view.head<size_type>(),
                         0);
                                
  cudf::size_type num_rows = cudf::detail::get_value<size_type>(offsets_view, col.size(), stream);
  cudf::size_type key_output_size = cudf::detail::get_value<size_type>(key_offsets_view, col.size(), stream);
  cudf::size_type value_output_size = cudf::detail::get_value<size_type>(value_offsets_view, col.size(), stream);
  cudf::size_type lists_size = cudf::detail::get_value<size_type>(lists_offsets_view, col.size(), stream);

  auto key_chars_column = cudf::make_numeric_column(data_type{type_id::INT8}, key_output_size, mask_state::UNALLOCATED, stream, mr);

  auto value_chars_column = cudf::make_numeric_column(data_type{type_id::INT8}, value_output_size, mask_state::UNALLOCATED, stream, mr);

  key_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  value_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  list_offsets_column = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, lists_size + 1, mask_state::UNALLOCATED, stream, mr);

  json_to_array_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      col.chars().head<char>(),
      col.offsets().head<size_type>(),
      key_chars_column->mutable_view().head<char>(),
      value_chars_column->mutable_view().head<char>(),
      key_offsets_column->mutable_view().head<size_type>(),
      value_offsets_column->mutable_view().head<size_type>(),
      list_offsets_column->mutable_view().head<size_type>(),
      key_output_size,
      value_output_size,
      nullptr);

  thrust::exclusive_scan(rmm::exec_policy(stream),  // list column offset
                         list_offsets_column->mutable_view().head<size_type>(),
                         list_offsets_column->mutable_view().head<size_type>() + lists_size + 1,
                         list_offsets_column->mutable_view().head<size_type>(),
                         0);

  thrust::exclusive_scan(rmm::exec_policy(stream),  // key column offset
                         key_offsets_column->mutable_view().head<size_type>(),
                         key_offsets_column->mutable_view().head<size_type>() + num_rows + 1,
                         key_offsets_column->mutable_view().head<size_type>(),
                         0);
  thrust::exclusive_scan(rmm::exec_policy(stream),  // value column offset
                         value_offsets_column->mutable_view().head<size_type>(),
                         value_offsets_column->mutable_view().head<size_type>() + num_rows + 1,
                         value_offsets_column->mutable_view().head<size_type>(),
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

std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                            rmm::mr::device_memory_resource* mr)
{
  return detail::json_to_array(col, 0, mr);
}

}  // namespace strings
}  // namespace cudf 
