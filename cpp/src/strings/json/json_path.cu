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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <io/utilities/column_type_histogram.hpp>
#include <io/utilities/parsing_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

// debug accessibility

// change to "\n" and 1 to make output more readable
#define DEBUG_NEWLINE
constexpr int DEBUG_NEWLINE_LEN = 0;

// temporary? spark doesn't strictly follow the JSONPath spec.
// I think this probably could be a configurable enum to control
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
// Other, non-spark known differences:
//
// - In jsonpath_ng, name subscripts can use double quotes instead of the standard
//   single quotes in the query string.
//   standard:      $.thing['subscript']
//   jsonpath_ng:   $.thing["subscript"]
//
//  Currently, this code only allows single-quotes but that could be expanded if necessary.
//
#define SPARK_BEHAVIORS

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
 * @brief A struct which represents a string.
 *
 * Typically used to point into a substring of a larger string, such as
 * the input json itself.
 *
 * @code
 * // where cur_pos is a pointer to the beginning of a name string in the
 * // input json and name_size is the computed size.
 * json_string name{cur_pos, name_size};
 * @endcode
 *
 * Also used for parameter passing in a few cases:
 *
 * @code
 * json_string wildcard{"*", 1};
 * func(wildcard);
 * @endcode
 */
struct json_string {
  const char* str;
  int64_t len;

  constexpr json_string() : str(nullptr), len(-1) {}
  constexpr json_string(const char* _str, int64_t _len) : str(_str), len(_len) {}

  constexpr bool operator==(json_string const& cmp)
  {
    return len == cmp.len && str != nullptr && cmp.str != nullptr &&
           thrust::equal(thrust::seq, str, str + len, cmp.str);
  }
};

/**
 * @brief Base parser class inherited by the (device-side) json_state class and
 * (host-side) path_state class.
 *
 * Contains a number of useful utility functions common to parsing json and
 * JSONPath strings.
 */
class parser {
 protected:
  constexpr parser() : input(nullptr), input_len(0), pos(nullptr) {}
  constexpr parser(const char* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }

  constexpr parser(parser const& p) : input(p.input), input_len(p.input_len), pos(p.pos) {}

  constexpr bool eof(const char* p) { return p - input >= input_len; }
  constexpr bool eof() { return eof(pos); }

  constexpr bool parse_whitespace()
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

  constexpr parse_result parse_string(json_string& str, bool can_be_empty, char quote)
  {
    str.str = nullptr;
    str.len = 0;

    if (parse_whitespace() && *pos == quote) {
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

    return can_be_empty ? parse_result::EMPTY : parse_result::ERROR;
  }

  // a name means:
  // - a string followed by a :
  // - no string
  constexpr parse_result parse_name(json_string& name, bool can_be_empty, char quote)
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

  // numbers, true, false, null.
  // this function is not particularly strong. badly formed values will get
  // consumed without throwing any errors
  constexpr parse_result parse_non_string_value(json_string& val)
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

    val.str = start;
    val.len = {end - start};

    return parse_result::SUCCESS;
  }

 protected:
  char const* input;
  int64_t input_len;
  char const* pos;

 private:
  constexpr bool is_whitespace(char c)
  {
    return c == ' ' || c == '\r' || c == '\n' || c == '\t' ? true : false;
  }
};

/**
 * @brief Output buffer object.  Used during the preprocess/size-computation step
 * and the actual output step.
 */
struct json_output {
  size_t output_max_len;
  size_t output_len;
  int element_count;
  char* output;

  constexpr void add_output(const char* str, size_t len)
  {
    if (output != nullptr) { memcpy(output + output_len, str, len); }
    output_len += len;
  }

  constexpr void add_output(json_string str) { add_output(str.str, str.len); }
};

enum json_element_type { NONE, OBJECT, ARRAY, VALUE };

/**
 * @brief Parsing class that holds the current state of the json to be parse and provides
 * functions for navigating through it.
 */
class json_state : private parser {
 public:
  constexpr json_state() : parser(), cur_el_start(nullptr), cur_el_type(json_element_type::NONE) {}
  constexpr json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len), cur_el_start(nullptr), cur_el_type(json_element_type::NONE)
  {
  }

  constexpr json_state(json_state const& j)
    : parser(j), cur_el_start(j.cur_el_start), cur_el_type(j.cur_el_type)
  {
  }

  // retrieve the entire current element as a json_string
  constexpr parse_result extract_element(json_output* output, bool list_element)
  {
    char const* start = cur_el_start;
    char const* end   = start;

    // if we're a value type, do a simple value parse.
    if (cur_el_type == VALUE) {
      pos = cur_el_start;
      if (parse_value() != parse_result::SUCCESS) { return parse_result::ERROR; }
      end = pos;

#if defined(SPARK_BEHAVIORS)
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

    if (output != nullptr) {
      output->add_output({start, end - start});
      output->element_count++;
    }
    return parse_result::SUCCESS;
  }

  // skip the next element
  constexpr parse_result skip_element() { return extract_element(nullptr, false); }

  // advance to the next element
  constexpr parse_result next_element() { return next_element_internal(false); }

  // advance inside the current element
  constexpr parse_result child_element(bool as_field = false)
  {
    // cannot retrieve a field from an array
    if (as_field && cur_el_type == json_element_type::ARRAY) { return parse_result::ERROR; }
    return next_element_internal(true);
  }

  // return the next element that matches the specified name.
  constexpr parse_result next_matching_element(json_string const& name, bool inclusive)
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
  // parse a value - either a string or a number/null/bool
  constexpr parse_result parse_value()
  {
    if (!parse_whitespace()) { return parse_result::ERROR; }

    // string or number?
    json_string unused;
    return *pos == '\"' ? parse_string(unused, false, '\"') : parse_non_string_value(unused);
  }

  constexpr parse_result next_element_internal(bool child)
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

    // element name, if any
    if (parse_name(cur_el_name, true, '\"') == parse_result::ERROR) { return parse_result::ERROR; }

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

  const char* cur_el_start;       // pointer to the first character of the -value- of the current
                                  // element - not the name
  json_string cur_el_name;        // name of the current element (if applicable)
  json_element_type cur_el_type;  // type of the current element
};

enum class path_operator_type { ROOT, CHILD, CHILD_WILDCARD, CHILD_INDEX, ERROR, END };

/**
 * @brief A "command" operator used to query a json string.  A full query is
 * an array of these operators applied to the incoming json string,
 */
struct path_operator {
  constexpr path_operator() : type(path_operator_type::ERROR), index(-1) {}
  constexpr path_operator(path_operator_type _type) : type(_type), index(-1) {}

  path_operator_type type;  // operator type
  json_string name;         // name to match against (if applicable)
  int index;                // index for subscript operator
};

/**
 * @brief Parsing class that holds the current state of the JSONPath string to be parsed
 * and provides functions for navigating through it. This is only called on the host
 * during the preprocess step which builds a command buffer that the gpu uses.
 */
class path_state : private parser {
 public:
  path_state(const char* _path, size_t _path_len) : parser(_path, _path_len) {}

  // get the next operator in the JSONPath string
  path_operator get_next_operator()
  {
    if (eof()) { return {path_operator_type::END}; }

    switch (*pos++) {
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
        bool const is_string = *pos == '\'' ? true : false;
        if (parse_path_name(op.name, term)) {
          pos++;
          if (op.name.len == 1 && op.name.str[0] == '*') {
            op.type = path_operator_type::CHILD_WILDCARD;
          } else {
            if (is_string) {
              op.type = path_operator_type::CHILD;
            } else {
              op.type = path_operator_type::CHILD_INDEX;
              op.index =
                cudf::io::parse_numeric<int>(op.name.str, op.name.str + op.name.len, json_opts, -1);
              CUDF_EXPECTS(op.index >= 0, "Invalid numeric index specified in JSONPath");
            }
          }
          return op;
        }
      } break;

      // wildcard operator
      case '*': {
        pos++;
        return path_operator{path_operator_type::CHILD_WILDCARD};
      } break;

      default: CUDF_FAIL("Unrecognized JSONPath operator"); break;
    }
    return {path_operator_type::ERROR};
  }

 private:
  cudf::io::parse_options_view json_opts{',', '\n', '\"', '.'};

  bool parse_path_name(json_string& name, json_string const& terminators)
  {
    switch (*pos) {
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
        char const* end         = std::find_first_of(
          pos, pos + chars_left, terminators.str, terminators.str + terminators.len);
        if (end) {
          name.str = pos;
          name.len = end - pos;
          pos      = end;
        } else {
          name.str = pos;
          name.len = chars_left;
          pos      = input + input_len;
        }
        break;
      }
    }

    // an empty name is not valid
    CUDF_EXPECTS(name.len > 0, "Invalid empty name in JSONpath query string");

    return true;
  }
};

/**
 * @brief Preprocess the incoming JSONPath string on the host to generate a
 * command buffer for use by the GPU.
 *
 * @param json_path The incoming json path
 * @param stream Cuda stream to perform any gpu actions on
 * @returns A tuple containing the command buffer, the maximum stack depth required and whether or
 * not the command buffer is empty.
 */
std::tuple<rmm::device_uvector<path_operator>, int, bool> build_command_buffer(
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
    if (op.type == path_operator_type::ROOT) {
      CUDF_EXPECTS(h_operators.size() == 0, "Root operator ($) can only exist at the root");
    }
    // if we havent' gotten a root operator to start, and we're not empty, quietly push a
    // root operator now.
    if (h_operators.size() == 0 && op.type != path_operator_type::ROOT &&
        op.type != path_operator_type::END) {
      h_operators.push_back(path_operator{path_operator_type::ROOT});
    }
    h_operators.push_back(op);
  } while (op.type != path_operator_type::END);

  rmm::device_uvector<path_operator> d_operators(h_operators.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(d_operators.data(),
                           h_operators.data(),
                           sizeof(path_operator) * h_operators.size(),
                           cudaMemcpyHostToDevice,
                           stream.value()));
  stream.synchronize();

  return {std::move(d_operators),
          max_stack_depth,
          h_operators.size() == 1 && h_operators[0].type == path_operator_type::END ? true : false};
}

#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * @param j_state The incoming json string and associated parser
 * @param commands The command buffer to be applied to the string. Always ends with a
 * path_operator_type::END
 * @param output Buffer user to store the results of the query
 * @returns A result code indicating success/fail/empty.
 */
template <int max_command_stack_depth>
__device__ parse_result parse_json_path(json_state& j_state,
                                        path_operator const* commands,
                                        json_output& output)
{
  // manually maintained context stack in lieu of calling parse_json_path recursively.
  struct context {
    json_state j_state;
    path_operator const* commands;
    bool list_element;
    bool state_flag;
  };
  context stack[max_command_stack_depth];
  int stack_pos     = 0;
  auto push_context = [&stack, &stack_pos](json_state const& _j_state,
                                           path_operator const* _commands,
                                           bool _list_element = false,
                                           bool _state_flag   = false) {
    if (stack_pos == max_command_stack_depth - 1) { return false; }
    stack[stack_pos++] = context{_j_state, _commands, _list_element, _state_flag};
    return true;
  };
  auto pop_context = [&stack, &stack_pos](context& c) {
    if (stack_pos > 0) {
      c = stack[--stack_pos];
      return true;
    }
    return false;
  };
  push_context(j_state, commands, false);

  parse_result last_result = parse_result::SUCCESS;
  context ctx;
  int element_count = 0;
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
        PARSE_TRY(ctx.j_state.child_element(true));
        if (last_result == parse_result::SUCCESS) {
          PARSE_TRY(ctx.j_state.next_matching_element(op.name, true));
          if (last_result == parse_result::SUCCESS) {
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element);
          }
        }
      } break;

      // .*
      // [*]
      // will return an array of things
      case path_operator_type::CHILD_WILDCARD: {
        // if we're on the first element of this wildcard
        if (!ctx.state_flag) {
          // we will only ever be returning 1 array
          if (!ctx.list_element) { output.add_output({"[" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN}); }

          // step into the child element
          PARSE_TRY(ctx.j_state.child_element());
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // first element
          PARSE_TRY(ctx.j_state.next_matching_element({"*", 1}, true));
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, ctx.list_element, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true);
        } else {
          // next element
          PARSE_TRY(ctx.j_state.next_matching_element({"*", 1}, false));
          if (last_result == parse_result::EMPTY) {
            if (!ctx.list_element) {
              output.add_output({"]" DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
            }
            last_result = parse_result::SUCCESS;
            break;
          }

          // re-push ourselves
          push_context(ctx.j_state, ctx.commands, ctx.list_element, true);
          // push the next command
          push_context(ctx.j_state, ctx.commands + 1, true);
        }
      } break;

      // [0]
      // [1]
      // etc
      // returns a single thing
      case path_operator_type::CHILD_INDEX: {
        PARSE_TRY(ctx.j_state.child_element());
        if (last_result == parse_result::SUCCESS) {
          json_string const any{"*", 1};
          PARSE_TRY(ctx.j_state.next_matching_element(any, true));
          if (last_result == parse_result::SUCCESS) {
            int idx;
            for (idx = 1; idx <= op.index; idx++) {
              PARSE_TRY(ctx.j_state.next_matching_element(any, false));
              if (last_result == parse_result::EMPTY) { break; }
            }
            // if we didn't end up at the index we requested, this is an invalid index
            if (idx - 1 != op.index) { return parse_result::ERROR; }
            push_context(ctx.j_state, ctx.commands + 1, ctx.list_element);
          }
        }
      } break;

      // some sort of error.
      case path_operator_type::ERROR: return parse_result::ERROR; break;

      // END case
      default: {
        if (ctx.list_element && element_count > 0) {
          output.add_output({"," DEBUG_NEWLINE, 1 + DEBUG_NEWLINE_LEN});
        }
        PARSE_TRY(ctx.j_state.extract_element(&output, ctx.list_element));
        if (ctx.list_element && last_result != parse_result::EMPTY) { element_count++; }
      } break;
    }
  }

  return parse_result::SUCCESS;
}

// hardcoding this for now. to reach a stack depth of 8 would require
// a jsonpath containing 7 nested wildcards so this is probably reasonable.
constexpr int max_command_stack_depth = 8;

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * This function exists primarily as a shim for debugging purposes.
 *
 * @param input The incoming json string
 * @param input_len Size of the incoming json string
 * @param commands The command buffer to be applied to the string. Always ends with a
 * path_operator_type::END
 * @param out_buf Buffer user to store the results of the query (nullptr in the size computation
 * step)
 * @param out_buf_size Size of the output buffer
 * @returns A pair containing the result code the output buffer.
 */
__device__ thrust::pair<parse_result, json_output> get_json_object_single(
  char const* input,
  size_t input_len,
  path_operator const* const commands,
  char* out_buf,
  size_t out_buf_size)
{
  json_state j_state(input, input_len);
  json_output output{out_buf_size, 0, 0, out_buf};

  auto const result = parse_json_path<max_command_stack_depth>(j_state, commands, output);

  return {result, output};
}

/**
 * @brief Kernel for running the JSONPath query.
 *
 * This kernel operates in a 2-pass way.  On the first pass, it computes
 * output sizes.  On the second pass it fills in the provided output buffers
 * (chars and validity)
 *
 * @param col Device view of the incoming string
 * @param commands JSONPath command buffer
 * @param output_offsets Buffer used to store the string offsets for the results of the query
 * (nullptr in the size computation step)
 * @param out_buf Buffer used to store the results of the query (nullptr in the size computation
 * step)
 * @param out_validity Output validity buffer (nullptr in the size computation step)
 */
__global__ void get_json_object_kernel(column_device_view col,
                                       path_operator const* const commands,
                                       size_type* output_offsets,
                                       char* out_buf,
                                       bitmask_type* out_validity)
{
  uint64_t const tid = threadIdx.x + (blockDim.x * blockIdx.x);

  bool is_valid = false;
  if (tid < col.size()) {
    string_view const str = col.element<string_view>(tid);
    size_type output_size = 0;
    if (str.size_bytes() > 0) {
      char* dst             = out_buf ? out_buf + output_offsets[tid] : nullptr;
      size_t const dst_size = out_buf ? output_offsets[tid + 1] - output_offsets[tid] : 0;

      parse_result result;
      json_output out;
      thrust::tie(result, out) =
        get_json_object_single(str.data(), str.size_bytes(), commands, dst, dst_size);
      output_size = out.output_len;
      if (out.element_count > 0 && result == parse_result::SUCCESS) { is_valid = true; }
    }

    // filled in only during the precompute step
    if (!out_buf) { output_offsets[tid] = static_cast<size_type>(output_size); }
  }

  // validity filled in only during the output step
  if (out_validity) {
    uint32_t mask = __ballot_sync(0xffffffff, is_valid);
    // 0th lane of the warp writes the validity
    if (!(tid % cudf::detail::warp_size) && tid < col.size()) {
      out_validity[cudf::word_index(tid)] = mask;
    }
  }
}

/**
 * @copydoc cudf::strings::detail::get_json_object
 */
std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // preprocess the json_path into a command buffer
  std::tuple<rmm::device_uvector<path_operator>, int, bool> preprocess =
    build_command_buffer(json_path, stream);
  CUDF_EXPECTS(std::get<1>(preprocess) <= max_command_stack_depth,
               "Encountered json_path string that is too complex");

  // allocate output offsets buffer.
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view offsets_view(*offsets);

  // if the query is empty, return a string column containing all nulls
  if (std::get<2>(preprocess)) {
    return std::make_unique<column>(
      data_type{type_id::STRING},
      col.size(),
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr),
      col.size());  // null count
  }

  cudf::detail::grid_1d const grid{col.size(), 512};

  auto cdv = column_device_view::create(col.parent(), stream);

  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *cdv, std::get<0>(preprocess).data(), offsets_view.head<size_type>(), nullptr, nullptr);

  // convert sizes to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets_view.head<size_type>(),
                         offsets_view.head<size_type>() + col.size() + 1,
                         offsets_view.head<size_type>(),
                         0);
  size_type const output_size =
    cudf::detail::get_value<size_type>(offsets_view, col.size(), stream);

  // allocate output string column
  auto chars = cudf::make_fixed_width_column(
    data_type{type_id::INT8}, output_size, mask_state::UNALLOCATED, stream, mr);

  // potential optimization : if we know that all outputs are valid, we could skip creating
  // the validity mask altogether
  rmm::device_buffer validity =
    cudf::detail::create_null_mask(col.size(), mask_state::UNINITIALIZED, stream, mr);

  // compute results
  cudf::mutable_column_view chars_view(*chars);
  get_json_object_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *cdv,
    std::get<0>(preprocess).data(),
    offsets_view.head<size_type>(),
    chars_view.head<char>(),
    static_cast<bitmask_type*>(validity.data()));

  return make_strings_column(col.size(),
                             std::move(offsets),
                             std::move(chars),
                             UNKNOWN_NULL_COUNT,
                             std::move(validity),
                             stream,
                             mr);
}

}  // namespace
}  // namespace detail

/**
 * @copydoc cudf::strings::get_json_object
 */
std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::get_json_object(col, json_path, 0, mr);
}

}  // namespace strings
}  // namespace cudf
