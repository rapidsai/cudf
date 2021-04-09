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

#pragma once

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/optional.h>

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

#define PARSE_TRY(_x)                                                       \
  do {                                                                      \
    last_result = _x;                                                       \
    if (last_result == parse_result::ERROR) { return parse_result::ERROR; } \
  } while (0)
  
/**
 * @brief Base parser class inherited by the (device-side) json_state class and
 * (host-side) path_state class.
 *
 * Contains a number of useful utility functions common to parsing json and
 * JSONPath strings.
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

  CUDA_HOST_DEVICE_CALLABLE parse_result parse_string(string_view& str,
                                                      bool can_be_empty,
                                                      char quote)
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
  CUDA_HOST_DEVICE_CALLABLE bool is_whitespace(char c) { return c <= ' '; }
};

/**
 * @brief Output buffer object.  Used during the preprocess/size-computation step
 * and the actual output step.
 *
 * There is an important distinction between two cases:
 *
 * - producing no output at all. that is, the query matched nothing in the input.
 * - producing empty output. the query matched something in the input, but the
 *   value of the result is an empty string.
 *
 * The `has_output` field is the flag which indicates whether or not the output
 * from the query should be considered empty or null.
 *
 */
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
    if (expected_type != NONE && cur_el_type != expected_type) { return parse_result::ERROR; }

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
}  // namespace

/**
 * @copydoc cudf::strings::get_json_object
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::strings::json_to_array
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> json_to_array(
  cudf::strings_column_view const& col,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf

