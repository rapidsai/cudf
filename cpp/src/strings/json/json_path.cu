#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <io/utilities/column_type_histogram.hpp>
#include <io/utilities/parsing_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf_test/column_wrapper.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

// temporary. for debugging purposes
#define DEBUG_NEWLINE
#define DEBUG_NEWLINE_LEN (0)

// #define DEBUG_NEWLINE  "\n"
// #define DEBUG_NEWLINE_LEN (1)

// temporary. spark doesn't strictly follow the JSONPath spec.
// I think this probably should be a configurable enum to control
// the kind of output you get and what features are supported.
//
// Current known differences:
// - When returning a string value as a single element, Spark strips the quotes.
//   standard:   "whee"
//   spark:      whee
//
// - Spark only supports the wildcard operator when in a subscript, eg  [*]
//   It does not handle .*
//
#define __SPARK_BEHAVIORS

// Other, non-spark known differences:
//
// - In jsonpath_ng, name subscripts can use double quotes instead of the standard
//   single quotes in the query string.
//   standard:      $.thing['subscript']
//   jsonpath_ng:   $.thing["subscript"]
//
//  Currently, this code only allows single-quotes but that can be easily expanded.
//

using namespace cudf;

CUDA_HOST_DEVICE_CALLABLE char to_lower(char const c)
{
  return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
CUDA_HOST_DEVICE_CALLABLE uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;

  *valid_flag = false;
  return 0;
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
CUDA_HOST_DEVICE_CALLABLE uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';

  *valid_flag = false;
  return 0;
}

CUDA_HOST_DEVICE_CALLABLE bool is_infinity(char const* begin, char const* end)
{
  if (*begin == '-' || *begin == '+') begin++;
  char const* cinf = "infinity";
  auto index       = begin;
  while (index < end) {
    if (*cinf != to_lower(*index)) break;
    index++;
    cinf++;
  }
  return ((index == begin + 3 || index == begin + 8) && index >= end);
}

template <typename T, int base = 10>
CUDA_HOST_DEVICE_CALLABLE T parse_numeric(const char* begin,
                                          const char* end,
                                          cudf::io::parse_options_view const& opts)
{
  T value{};
  bool all_digits_valid = true;

  // Handle negative values if necessary
  int32_t sign = (*begin == '-') ? -1 : 1;

  // Handle infinity
  if (std::is_floating_point<T>::value && is_infinity(begin, end)) {
    return sign * std::numeric_limits<T>::infinity();
  }
  if (*begin == '-' || *begin == '+') begin++;

  // Skip over the "0x" prefix for hex notation
  if (base == 16 && begin + 2 < end && *begin == '0' && *(begin + 1) == 'x') { begin += 2; }

  // Handle the whole part of the number
  // auto index = begin;
  while (begin < end) {
    if (*begin == opts.decimal) {
      ++begin;
      break;
    } else if (base == 10 && (*begin == 'e' || *begin == 'E')) {
      break;
    } else if (*begin != opts.thousands && *begin != '+') {
      value = (value * base) + decode_digit<T>(*begin, &all_digits_valid);
    }
    ++begin;
  }

  if (std::is_floating_point<T>::value) {
    // Handle fractional part of the number if necessary
    double divisor = 1;
    while (begin < end) {
      if (*begin == 'e' || *begin == 'E') {
        ++begin;
        break;
      } else if (*begin != opts.thousands && *begin != '+') {
        divisor /= base;
        value += decode_digit<T>(*begin, &all_digits_valid) * divisor;
      }
      ++begin;
    }

    // Handle exponential part of the number if necessary
    if (begin < end) {
      const int32_t exponent_sign = *begin == '-' ? -1 : 1;
      if (*begin == '-' || *begin == '+') { ++begin; }
      int32_t exponent = 0;
      while (begin < end) {
        exponent = (exponent * 10) + decode_digit<T>(*(begin++), &all_digits_valid);
      }
      if (exponent != 0) { value *= exp10(double(exponent * exponent_sign)); }
    }
  }
  if (!all_digits_valid) { return std::numeric_limits<T>::quiet_NaN(); }

  return value * sign;
}

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

  CUDA_HOST_DEVICE_CALLABLE json_string() : str(nullptr), len(-1) {}
  CUDA_HOST_DEVICE_CALLABLE json_string(const char* _str, int64_t _len) : str(_str), len(_len) {}

  CUDA_HOST_DEVICE_CALLABLE bool operator==(json_string const& cmp)
  {
    return len == cmp.len && str != nullptr && cmp.str != nullptr &&
           device_strncmp(str, cmp.str, static_cast<size_t>(len));
  }
};

enum class parse_result {
  ERROR,
  SUCCESS,
  EMPTY,
};

enum json_element_type { NONE, OBJECT, ARRAY, VALUE };

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

CUDA_HOST_DEVICE_CALLABLE bool is_whitespace(char c)
{
  return c == ' ' || c == '\r' || c == '\n' || c == '\t' ? true : false;
}

class parser {
 protected:
  CUDA_HOST_DEVICE_CALLABLE parser() : input(nullptr), input_len(0), pos(nullptr) {}
  CUDA_HOST_DEVICE_CALLABLE parser(const char* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }

  CUDA_HOST_DEVICE_CALLABLE parser(parser const& p) { *this = p; }

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

  CUDA_HOST_DEVICE_CALLABLE parse_result parse_string(json_string& str,
                                                      bool can_be_empty,
                                                      char quote)
  {
    str.str = nullptr;
    str.len = 0;

    if (parse_whitespace()) {
      if (*pos == quote) {
        const char* start = ++pos;
        while (!eof()) {
          if (*pos == quote) {
            str.str = start;
            str.len = pos - start;
            pos++;
            return parse_result::SUCCESS;
          }
          pos++;
        }
      }
    }

    return can_be_empty ? parse_result::EMPTY : parse_result::ERROR;
  }

  // a name means:
  // - a string followed by a :
  // - no string
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_name(json_string& name,
                                                    bool can_be_empty,
                                                    char quote)
  {
    if (parse_string(name, can_be_empty, quote) == parse_result::ERROR) {
      return parse_result::ERROR;
    }

    // if we got a real string, the next char must be a :
    if (name.len > 0) {
      if (!parse_whitespace()) { return parse_result::ERROR; }
      if (*pos == ':') {
        pos++;
        return parse_result::SUCCESS;
      }
    }
    return parse_result::EMPTY;
  }

  // this function is not particularly strong
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_number(json_string& val)
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // parse to the end of the number (does not do any error checking on whether
    // the number is reasonably formed or not)
    char const* start = pos;
    char const* end   = start;
    while (!eof(end)) {
      char c = *end;
      if (c == ',' || is_whitespace(c)) { break; }
      end++;
    }
    pos = end;

    val.str = start;
    val.len = {end - start};

    return parse_result::SUCCESS;
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

  CUDA_HOST_DEVICE_CALLABLE json_state(json_state const& j) : parser(j) { *this = j; }

  CUDA_HOST_DEVICE_CALLABLE parse_result extract_element(json_output* output, bool list_element)
  {
    // collapse the current element into a json_string

    char const* start = cur_el_start;
    char const* end   = start;

    // if we're a value type, do a simple value parse.
    if (cur_el_type == VALUE) {
      pos = cur_el_start;
      if (parse_value() != parse_result::SUCCESS) { return parse_result::ERROR; }
      end = pos;

#if defined(__SPARK_BEHAVIORS)
      // spark/hive-specific behavior.  if this is a non-list-element wrapped in quotes,
      // strip them
      if (!list_element && *start == '\"' && *(end - 1) == '\"') {
        start++;
        end--;
      }
#endif
    }
    // otherwise, march through everything inside
    else {
      int obj_count = 0;
      int arr_count = 0;

      while (!eof(end)) {
        char c = *end++;
        // could do some additional checks here. we know our current
        // element type, so we could be more strict on what kinds of
        // characters we expect to see.
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
    }

    // parse trailing ,
    if (parse_whitespace()) {
      if (*pos == ',') { pos++; }
    }

    if (output != nullptr) {
      // seems like names are never included with JSONPath unless
      // they are nested within the element being returned.
      /*
      if(cur_el_name.len > 0){
        output->add_output({"\"", 1});
        output->add_output(cur_el_name);
        output->add_output({"\"", 1});
        output->add_output({":", 1});
      }
      */
      output->add_output({start, end - start});
    }
    return parse_result::SUCCESS;
  }

  CUDA_HOST_DEVICE_CALLABLE parse_result skip_element() { return extract_element(nullptr, false); }

  json_element_type element;

  CUDA_HOST_DEVICE_CALLABLE parse_result next_element() { return next_element_internal(false); }

  CUDA_HOST_DEVICE_CALLABLE parse_result child_element() { return next_element_internal(true); }

  CUDA_HOST_DEVICE_CALLABLE parse_result next_matching_element(json_string const& name,
                                                               bool inclusive)
  {
    // if we're not including the current element, skip it
    if (!inclusive) {
      parse_result result = next_element_internal(false);
      if (result != parse_result::SUCCESS) { return result; }
    }
    // loop until we find a match or there's nothing left
    do {
      // wildcard matches anything
      if (name.len == 1 && name.str[0] == '*') {
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

 private:
  CUDA_HOST_DEVICE_CALLABLE parse_result parse_value()
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // string or number?
    json_string unused;
    return *pos == '\"' ? parse_string(unused, false, '\"') : parse_number(unused);
  }

  CUDA_HOST_DEVICE_CALLABLE parse_result next_element_internal(bool child)
  {
    // if we're not getting a child element, skip the current element.
    // this will leave pos as the first character -after- the close of
    // the current element
    if (!child && cur_el_start != nullptr) {
      if (skip_element() == parse_result::ERROR) { return parse_result::ERROR; }
      cur_el_start = nullptr;
    }
    // otherwise pos will be at the first character within the current element

    // what's next
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    // if we're closing off a parent element, we're done
    char c = *pos;
    if (c == ']' || c == '}') { return parse_result::EMPTY; }

    // element name, if any
    if (parse_name(cur_el_name, true, '\"') == parse_result::ERROR) { return parse_result::ERROR; }

    // element type
    if (!parse_whitespace()) { return parse_result::EMPTY; }
    switch (*pos) {
      case '[': cur_el_type = ARRAY; break;
      case '{': cur_el_type = OBJECT; break;

      case ',':
      case ':':
      case '\'': return parse_result::ERROR;

      // value type
      default: cur_el_type = VALUE; break;
    }
    pos++;

    // the start of the current element is always at the value, not the name
    cur_el_start = pos - 1;
    return parse_result::SUCCESS;
  }

  const char* cur_el_start;
  json_string cur_el_name;
  json_element_type cur_el_type;
};

enum class path_operator_type { ROOT, CHILD, CHILD_WILDCARD, CHILD_INDEX, ERROR, END };

struct path_operator {
  CUDA_HOST_DEVICE_CALLABLE path_operator() : type(path_operator_type::ERROR), index(-1) {}
  CUDA_HOST_DEVICE_CALLABLE path_operator(path_operator_type _type) : type(_type), index(-1) {}

  path_operator_type type;
  json_string name;
  int index;
};
struct command_buffer {
  rmm::device_uvector<path_operator> commands;
  // used as backing memory for the name fields inside the
  // path_operator objects
  string_scalar json_path;
};

// current state of the JSONPath
class path_state : private parser {
 public:
  CUDA_HOST_DEVICE_CALLABLE path_state(const char* _path, size_t _path_len)
    : parser(_path, _path_len)
  {
  }

  CUDA_HOST_DEVICE_CALLABLE path_operator get_next_operator()
  {
    if (eof()) { return {path_operator_type::END}; }

    char c = *pos++;
    switch (c) {
      case '$': return {path_operator_type::ROOT};

      case '.': {
        path_operator op;
        json_string term{".[", 2};
        if (parse_path_name(op.name, term)) {
          // this is another potential use case for __SPARK_BEHAVIORS / configurability
          // Spark currently only handles the wildcard operator inside [*], it does
          // not handle .*
          if (op.name.len == 1 && op.name.str[0] == '*') {
            op.type = path_operator_type::CHILD_WILDCARD;
          } else {
            op.type = path_operator_type::CHILD;
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
        bool is_string = *pos == '\'' ? true : false;
        if (parse_path_name(op.name, term)) {
          pos++;
          if (op.name.len == 1 && op.name.str[0] == '*') {
            op.type = path_operator_type::CHILD_WILDCARD;
          } else {
            if (is_string) {
              op.type = path_operator_type::CHILD;
            } else {
              op.type  = path_operator_type::CHILD_INDEX;
              op.index = parse_numeric<int>(op.name.str, op.name.str + op.name.len, json_opts);
            }
          }
          return op;
        }
      } break;

      default: break;
    }
    return {path_operator_type::ERROR};
  }

 private:
  cudf::io::parse_options_view json_opts{',', '\n', '\"', '.'};

  CUDA_HOST_DEVICE_CALLABLE bool parse_path_name(json_string& name, json_string& terminators)
  {
    char c = *pos;
    switch (c) {
      case '*':
        name.str = pos;
        name.len = 1;
        pos++;
        break;

      case '\'':
        if (parse_string(name, false, '\'') != parse_result::SUCCESS) { return false; }
        break;

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
      }
    }

    return true;
  }
};

std::pair<rmm::device_uvector<path_operator>, int> build_command_buffer(
  cudf::string_scalar const& json_path, rmm::cuda_stream_view stream)
{
  std::string h_json_path = json_path.to_string(stream);
  path_state p_state(h_json_path.data(), static_cast<size_type>(h_json_path.size()));

  std::vector<path_operator> h_operators;

  path_operator op;
  int max_stack_depth = 1;
  do {
    op = p_state.get_next_operator();
    if (op.type == path_operator_type::ERROR) {
      CUDF_FAIL("Encountered invalid JSONPath input string");
    }
    if (op.type == path_operator_type::CHILD_WILDCARD) { max_stack_depth++; }
    // convert pointer to device pointer
    if (op.name.len > 0) { op.name.str = json_path.data() + (op.name.str - h_json_path.data()); }
    h_operators.push_back(op);
  } while (op.type != path_operator_type::END);

  rmm::device_uvector<path_operator> d_operators(h_operators.size(), stream);
  cudaMemcpyAsync(d_operators.data(),
                  h_operators.data(),
                  sizeof(path_operator) * h_operators.size(),
                  cudaMemcpyHostToDevice,
                  stream.value());

  return {std::move(d_operators), max_stack_depth};
}

#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)

template <int max_command_stack_depth>
CUDA_HOST_DEVICE_CALLABLE parse_result parse_json_path(json_state& _j_state,
                                                       path_operator const* _commands,
                                                       json_output& output,
                                                       bool _list_element = false)
{
  // manually maintained context stack in lieu of calling parse_json_path recursively.
  struct context {
    json_state j_state;
    path_operator const* commands;
    bool list_element;
    int element_count;
    bool state_flag;
  };
  context stack[max_command_stack_depth];
  int stack_pos     = 0;
  auto push_context = [&stack, &stack_pos](json_state const& _j_state,
                                           path_operator const* _commands,
                                           bool _list_element = false,
                                           int _element_count = 0,
                                           bool _state_flag   = false) {
    if (stack_pos == max_command_stack_depth - 1) { return false; }
    stack[stack_pos++] = context{_j_state, _commands, _list_element, _element_count, _state_flag};
    return true;
  };
  auto pop_context = [&stack, &stack_pos](context& c) {
    if (stack_pos > 0) {
      c = stack[--stack_pos];
      return true;
    }
    return false;
  };
  push_context(_j_state, _commands, _list_element);

  parse_result last_result = parse_result::SUCCESS;
  context ctx;
  while (pop_context(ctx)) {
    path_operator op = *ctx.commands;

    switch (op.type) {
      // whatever the first object is
      case path_operator_type::ROOT:
        PARSE_TRY(ctx.j_state.next_element());
        push_context(ctx.j_state, ctx.commands + 1);
        break;

      // .name
      // ['name']
      // [1]
      // will return a single thing
      case path_operator_type::CHILD: {
        PARSE_TRY(ctx.j_state.child_element());
        if (last_result == parse_result::SUCCESS) {
          PARSE_TRY(ctx.j_state.next_matching_element(op.name, true));
          if (last_result == parse_result::SUCCESS) {
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element, ctx.element_count);
          }
        }
      } break;

      // .*
      // [*]
      // will return an array of things
      case path_operator_type::CHILD_WILDCARD: {
        // if we're on the first element of this wildcard
        if (!ctx.state_flag) {
          output.add_output("[" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN);

          // step into the child element
          PARSE_TRY(ctx.j_state.child_element());
          if (last_result == parse_result::EMPTY) {
            output.add_output("]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN);
            last_result = parse_result::SUCCESS;
            break;
          }

          // first element
          PARSE_TRY(ctx.j_state.next_matching_element(op.name, true));
          if (last_result == parse_result::EMPTY) {
            output.add_output("]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN);
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, false, 0, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true, 0);
        } else {
          // if we actually processed something to the output, increment count
          if (last_result != parse_result::EMPTY) { ctx.element_count++; }

          // next element
          PARSE_TRY(ctx.j_state.next_matching_element(op.name, false));
          if (last_result == parse_result::EMPTY) {
            output.add_output("]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN);
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, false, 0, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true, ctx.element_count);
        }
      } break;

      // [0]
      // [1]
      // etc
      // returns a single thing
      case path_operator_type::CHILD_INDEX: {
        PARSE_TRY(ctx.j_state.child_element());
        if (last_result == parse_result::SUCCESS) {
          json_string any{"*", 1};
          PARSE_TRY(ctx.j_state.next_matching_element(any, true));
          if (last_result == parse_result::SUCCESS) {
            for (int idx = 1; idx <= op.index; idx++) {
              PARSE_TRY(ctx.j_state.next_matching_element(any, false));
              if (last_result == parse_result::EMPTY) { break; }
            }
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element, ctx.element_count);
          }
        }
      } break;

      // some sort of error.
      case path_operator_type::ERROR: return parse_result::ERROR; break;

      // END case
      default: {
        if (ctx.list_element && ctx.element_count > 0) {
          output.add_output({"," DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
        }
        PARSE_TRY(ctx.j_state.extract_element(&output, ctx.list_element));
      } break;
    }
  }
  return parse_result::SUCCESS;
}

// hardcoding this for now. to reach a stack depth of 8 would require
// a jsonpath containing 7 nested wildcards so this is probably reasonable.
constexpr int max_command_stack_depth = 8;

CUDA_HOST_DEVICE_CALLABLE json_output get_json_object_single(char const* input,
                                                             size_t input_len,
                                                             path_operator const* const commands,
                                                             char* out_buf,
                                                             size_t out_buf_size)
{
  if (input_len == 0) { return json_output{0, 0, out_buf}; }

  json_state j_state(input, input_len);
  json_output output{out_buf_size, 0, out_buf};

  parse_json_path<max_command_stack_depth>(j_state, commands, output);

  return output;
}

__global__ void get_json_object_kernel(char const* chars,
                                       size_type const* offsets,
                                       path_operator const* const commands,
                                       size_type* output_offsets,
                                       char* out_buf,
                                       size_type num_rows)
{
  uint64_t const tid = threadIdx.x + (blockDim.x * blockIdx.x);

  if (tid >= num_rows) { return; }

  char* dst       = out_buf ? out_buf + output_offsets[tid] : nullptr;
  size_t dst_size = out_buf ? output_offsets[tid + 1] - output_offsets[tid] : 0;

  json_output out = get_json_object_single(
    chars + offsets[tid], offsets[tid + 1] - offsets[tid], commands, dst, dst_size);

  // filled in only during the precompute step
  if (!out_buf) { output_offsets[tid] = static_cast<size_type>(out.output_len); }
}

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // preprocess the json_path into a command buffer
  std::pair<rmm::device_uvector<path_operator>, int> preprocess =
    build_command_buffer(json_path, stream);
  CUDF_EXPECTS(preprocess.second <= max_command_stack_depth,
               "Encountered json_path string that is too complex");

  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view offsets_view(*offsets);

  cudf::detail::grid_1d const grid{col.size(), 512};

  // preprocess sizes
  get_json_object_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    col.chars().head<char>(),
    col.offsets().head<size_type>(),
    preprocess.first.data(),
    offsets_view.head<size_type>(),
    nullptr,
    col.size());

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
    preprocess.first.data(),
    offsets_view.head<size_type>(),
    chars_view.head<char>(),
    col.size());

  return make_strings_column(col.size(),
                             std::move(offsets),
                             std::move(chars),
                             col.null_count(),
                             cudf::detail::copy_bitmask(col.parent(), stream, mr),
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
