#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cudf_test/column_wrapper.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

using namespace cudf;

CUDA_HOST_DEVICE_CALLABLE bool device_strncmp(const char* str1, const char* str2, size_t num_chars)
{
  for (size_t idx = 0; idx < num_chars; idx++) {
    if (str1[idx] != str2[idx]) { return false; }
  }
  return true;
}

CUDA_HOST_DEVICE_CALLABLE char const* device_strpbrk(const char* str,
                                                     size_t str_size,
                                                     const char* tok,
                                                     size_t tok_size)
{
  size_t pos = 0;
  while (pos < str_size) {
    size_t tpos = 0;
    char c      = str[pos];
    while (tpos < tok_size) {
      if (c == tok[tpos]) { return str + pos; }
      tpos++;
    }
    pos++;
  }
  return nullptr;
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

enum json_element_type {
  NONE,
  OBJECT,
  ARRAY,
};

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

  CUDA_HOST_DEVICE_CALLABLE bool parse_name(json_string& name, json_string& terminators)
  {
    char c = *pos;
    switch (c) {
      case '*':
        name.str = pos;
        name.len = 1;
        pos++;
        return true;

      default: {
        size_t const chars_left = input_len - (pos - input);
        char const* end         = device_strpbrk(pos, chars_left, terminators.str, terminators.len);
        if (end) {
          name.str = pos;
          name.len = end - pos;
          pos      = end;
        } else {
          name.str = pos;
          name.len = chars_left;
          pos      = input + input_len;
        }
        return true;
      } break;
    }

    return false;
  }

 protected:
  char const* input;
  int64_t input_len;
  char const* pos;
};

class json_state : private parser {
 public:
  CUDA_HOST_DEVICE_CALLABLE json_state()
    : parser(), element(json_element_type::NONE), cur_el_start(nullptr)
  {
  }
  CUDA_HOST_DEVICE_CALLABLE json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len), element(json_element_type::NONE), cur_el_start(nullptr)
  {
  }

  CUDA_HOST_DEVICE_CALLABLE bool next_match(json_string& str, json_state& child)
  {
    json_string name;
    if (!parse_string(name, true)) { return false; }
    if ((str.len == 1 && str.str[0] == '*') || str == name) {
      // if this isn't an empty string, parse out the :
      if (name.len > 0) {
        if (!parse_whitespace() || *pos != ':') { return false; }
        pos++;
      }

      // we have a match on the name, so advance to the beginning of the next element
      if (parse_whitespace()) {
        switch (*pos) {
          case '[': element = ARRAY; break;

          case '{': element = OBJECT; break;

          default: return false;
        }
        cur_el_start = pos++;

        // success
        child = *this;
        return true;
      }
    }
    return false;
  }

  CUDA_HOST_DEVICE_CALLABLE json_string extract_element()
  {
    // collapse the current element into a json_string
    int obj_count = 0;
    int arr_count = 0;

    char const* start = cur_el_start;
    char const* end   = start;
    while (!eof(end)) {
      char c = *end++;
      switch (c) {
        case '{': obj_count++; break;
        case '}': obj_count--; break;
        case '[': arr_count++; break;
        case ']': arr_count--; break;
        default: break;
      }
      if (obj_count == 0 && arr_count == 0) { break; }
    }
    pos = end;

    return {start, end - start};
  }

  json_element_type element;

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

enum path_operator_type { ROOT, CHILD, CHILD_WILDCARD, CHILD_INDEX, ERROR, END };

// constexpr max_name_len    (63)
struct path_operator {
  path_operator_type type;
  json_string name;
  int index;
};

// current state of the JSONPath
class path_state : private parser {
 public:
  CUDA_HOST_DEVICE_CALLABLE path_state() : parser() {}
  CUDA_HOST_DEVICE_CALLABLE path_state(const char* _path, size_t _path_len)
    : parser(_path, _path_len)
  {
  }

  CUDA_HOST_DEVICE_CALLABLE path_operator get_next_operator()
  {
    if (eof()) { return {END}; }

    char c = parse_char();
    switch (c) {
      case '$': return {ROOT};

      case '.': {
        path_operator op;
        json_string term{".[", 2};
        if (parse_name(op.name, term)) {
          if (op.name.len == 1 && op.name.str[0] == '*') {
            op.type = CHILD_WILDCARD;
          } else {
            op.type = CHILD;
          }
          return op;
        }
      } break;

      // 3 ways this can be used
      // indices:   [0]
      // name:      ['book']
      // wildcard:  [*]
      case '[': {
        path_operator op;
        json_string term{"]", 1};
        if (parse_name(op.name, term)) {
          pos++;
          if (op.name.len == 1 && op.name.str[0] == '*') {
            op.type = CHILD_WILDCARD;
          } else {
            // unhandled cases
            break;
          }
          return op;
        }
      } break;

      default: break;
    }
    return {ERROR};
  }

 private:
  CUDA_HOST_DEVICE_CALLABLE char parse_char() { return *pos++; }
};

struct json_output {
  size_t output_max_len;
  size_t output_len;
  char* output;

  CUDA_HOST_DEVICE_CALLABLE void add_output(const char* str, size_t len)
  {
    if (output != nullptr) {
      // assert output_len + len < output_max_len
      memcpy(output + output_len, str, len);
    }
    output_len += len;
  }

  CUDA_HOST_DEVICE_CALLABLE void add_output(json_string str) { add_output(str.str, str.len); }
};

CUDA_HOST_DEVICE_CALLABLE void parse_json_path(json_state& j_state,
                                               path_state p_state,
                                               json_output& output)
{
  path_operator op = p_state.get_next_operator();

  switch (op.type) {
    // whatever the first object is
    case ROOT: {
      json_state child;
      json_string wildcard{"*", 1};
      if (j_state.next_match(wildcard, child)) { parse_json_path(child, p_state, output); }
    } break;

    // .name
    // ['name']
    // [1]
    // will return a single thing
    case CHILD: {
      json_state child;
      if (j_state.next_match(op.name, child)) { parse_json_path(child, p_state, output); }
    } break;

    // .*
    // [*]
    // will return an array of things
    case CHILD_WILDCARD: {
      output.add_output("[\n", 2);

      json_state child;
      int count = 0;
      while (j_state.next_match(op.name, child)) {
        if (count > 0) { output.add_output(",\n", 2); }
        parse_json_path(child, p_state, output);
        j_state = child;
        count++;
      }
      output.add_output("]\n", 2);
    } break;

    // some sort of error.
    case ERROR: break;

    // END case
    default: output.add_output(j_state.extract_element()); break;
  }
}

CUDA_HOST_DEVICE_CALLABLE json_output get_json_object_single(char const* input,
                                                             size_t input_len,
                                                             char const* path,
                                                             size_t path_len,
                                                             char* out_buf,
                                                             size_t out_buf_size)
{
  // TODO: add host-side code to verify path is a valid string.
  json_state j_state(input, input_len);
  path_state p_state(path, path_len);
  json_output output{out_buf_size, 0, out_buf};

  parse_json_path(j_state, p_state, output);

  return output;
}

__global__ void get_json_object_kernel(char const* chars,
                                       size_type const* offsets,
                                       char const* json_path,
                                       size_t json_path_len,
                                       size_type* output_offsets,
                                       char* out_buf,
                                       size_t out_buf_size)
{
  uint64_t const tid = threadIdx.x + (blockDim.x * blockIdx.x);

  json_output out = get_json_object_single(chars + offsets[tid],
                                           offsets[tid + 1] - offsets[tid],
                                           json_path,
                                           json_path_len,
                                           out_buf,
                                           out_buf_size);

  // filled in only during the precompute step
  if (output_offsets != nullptr) { output_offsets[tid] = static_cast<size_type>(out.output_len); }
}

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  size_t stack_size;
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, 2048);

  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view offsets_view(*offsets);

  cudf::detail::grid_1d const grid{1, col.size()};

  // preprocess sizes
  get_json_object_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    col.chars().head<char>(),
    col.offsets().head<size_type>(),
    json_path.data(),
    json_path.size(),
    offsets_view.head<size_type>(),
    nullptr,
    0);

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets_view.head<size_type>(),
                         offsets_view.head<size_type>() + col.size() + 1,
                         offsets_view.head<size_type>(),
                         0);
  size_type output_size = cudf::detail::get_value<size_type>(offsets_view, col.size(), stream);

  // allocate output string column
  auto chars = cudf::make_fixed_width_column(
    data_type{type_id::INT8}, output_size, mask_state::UNALLOCATED, stream, mr);

  // compute results
  cudf::mutable_column_view chars_view(*chars);
  get_json_object_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    col.chars().head<char>(),
    col.offsets().head<size_type>(),
    json_path.data(),
    json_path.size(),
    nullptr,
    chars_view.head<char>(),
    output_size);

  // reset back to original stack size
  cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

  return make_strings_column(col.size(),
                             std::move(offsets),
                             std::move(chars),
                             UNKNOWN_NULL_COUNT,
                             rmm::device_buffer{},
                             stream,
                             mr);
}

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::get_json_object(col, json_path, 0, mr);
}

}  // namespace strings
}  // namespace cudf