
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "io/utilities/trie.cuh"
#include "nested_json.hpp"
#include "tabulate_output_iterator.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/tokenize_json.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform_scan.h>

namespace cudf::io::json {
namespace detail {

struct write_if {
  using token_t   = cudf::io::json::token_t;
  using scan_type = thrust::pair<token_t, bool>;
  PdaTokenT* tokens;
  size_t n;
  // Index, value
  __device__ void operator()(size_type i, scan_type x)
  {
    if (i == n - 1 or tokens[i + 1] == token_t::LineEnd) {
      if (x.first == token_t::ErrorBegin and tokens[i] != token_t::ErrorBegin) {
        tokens[i] = token_t::ErrorBegin;
      }
    }
  }
};

enum class number_state {
  START = 0,
  SAW_NEG,  // not a complete state
  LEADING_ZERO,
  WHOLE,
  SAW_RADIX,  // not a complete state
  FRACTION,
  START_EXPONENT,       // not a complete state
  AFTER_SIGN_EXPONENT,  // not a complete state
  EXPONENT
};

enum class string_state {
  NORMAL = 0,
  ESCAPED,   // not a complete state
  ESCAPED_U  // not a complete state
};

__device__ inline bool substr_eq(const char* data,
                                 SymbolOffsetT const start,
                                 SymbolOffsetT const end,
                                 SymbolOffsetT const expected_len,
                                 const char* expected)
{
  if (end - start != expected_len) { return false; }
  for (auto idx = 0; idx < expected_len; idx++) {
    if (data[start + idx] != expected[idx]) { return false; }
  }
  return true;
}

void validate_token_stream(device_span<char const> d_input,
                           device_span<PdaTokenT> tokens,
                           device_span<SymbolOffsetT> token_indices,
                           cudf::io::json_reader_options const& options,
                           rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (!options.is_strict_validation()) { return; }
  using token_t = cudf::io::json::token_t;
  cudf::detail::optional_trie trie_na =
    cudf::detail::create_serialized_trie(options.get_na_values(), stream);
  auto trie_na_view    = cudf::detail::make_trie_view(trie_na);
  auto validate_values = cuda::proclaim_return_type<bool>(
    [data                        = d_input.data(),
     trie_na                     = trie_na_view,
     allow_numeric_leading_zeros = options.is_allowed_numeric_leading_zeros(),
     allow_nonnumeric =
       options.is_allowed_nonnumeric_numbers()] __device__(SymbolOffsetT start,
                                                           SymbolOffsetT end) -> bool {
      // This validates an unquoted value. A value must match https://www.json.org/json-en.html
      // but the leading and training whitespace should already have been removed, and is not
      // a string
      auto c               = data[start];
      auto is_null_literal = serialized_trie_contains(trie_na, {data + start, end - start});
      if (is_null_literal) {
        return true;
      } else if ('n' == c) {
        return substr_eq(data, start, end, 4, "null");
      } else if ('t' == c) {
        return substr_eq(data, start, end, 4, "true");
      } else if ('f' == c) {
        return substr_eq(data, start, end, 5, "false");
      } else if (allow_nonnumeric && c == 'N') {
        return substr_eq(data, start, end, 3, "NaN");
      } else if (allow_nonnumeric && c == 'I') {
        return substr_eq(data, start, end, 8, "Infinity");
      } else if (allow_nonnumeric && c == '+') {
        return substr_eq(data, start, end, 4, "+INF") ||
               substr_eq(data, start, end, 9, "+Infinity");
      } else if ('-' == c || c <= '9' && 'c' >= '0') {
        // number
        auto num_state = number_state::START;
        for (auto at = start; at < end; at++) {
          c = data[at];
          switch (num_state) {
            case number_state::START:
              if ('-' == c) {
                num_state = number_state::SAW_NEG;
              } else if ('0' == c) {
                num_state = number_state::LEADING_ZERO;
              } else if (c >= '1' && c <= '9') {
                num_state = number_state::WHOLE;
              } else {
                return false;
              }
              break;
            case number_state::SAW_NEG:
              if ('0' == c) {
                num_state = number_state::LEADING_ZERO;
              } else if (c >= '1' && c <= '9') {
                num_state = number_state::WHOLE;
              } else if (allow_nonnumeric && 'I' == c) {
                return substr_eq(data, start, end, 4, "-INF") ||
                       substr_eq(data, start, end, 9, "-Infinity");
              } else {
                return false;
              }
              break;
            case number_state::LEADING_ZERO:
              if (allow_numeric_leading_zeros && c >= '0' && c <= '9') {
                num_state = number_state::WHOLE;
              } else if ('.' == c) {
                num_state = number_state::SAW_RADIX;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::START_EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::WHOLE:
              if (c >= '0' && c <= '9') {
                num_state = number_state::WHOLE;
              } else if ('.' == c) {
                num_state = number_state::SAW_RADIX;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::START_EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::SAW_RADIX:
              if (c >= '0' && c <= '9') {
                num_state = number_state::FRACTION;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::START_EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::FRACTION:
              if (c >= '0' && c <= '9') {
                num_state = number_state::FRACTION;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::START_EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::START_EXPONENT:
              if ('+' == c || '-' == c) {
                num_state = number_state::AFTER_SIGN_EXPONENT;
              } else if (c >= '0' && c <= '9') {
                num_state = number_state::EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::AFTER_SIGN_EXPONENT:
              if (c >= '0' && c <= '9') {
                num_state = number_state::EXPONENT;
              } else {
                return false;
              }
              break;
            case number_state::EXPONENT:
              if (c >= '0' && c <= '9') {
                num_state = number_state::EXPONENT;
              } else {
                return false;
              }
              break;
          }
        }
        return num_state != number_state::AFTER_SIGN_EXPONENT &&
               num_state != number_state::START_EXPONENT && num_state != number_state::SAW_NEG &&
               num_state != number_state::SAW_RADIX;
      } else {
        return false;
      }
    });

  auto validate_strings = cuda::proclaim_return_type<bool>(
    [data = d_input.data(),
     allow_unquoted_control_chars =
       options.is_allowed_unquoted_control_chars()] __device__(SymbolOffsetT start,
                                                               SymbolOffsetT end) -> bool {
      // This validates a quoted string. A string must match https://www.json.org/json-en.html
      // but we already know that it has a starting and ending " and all white space has been
      // stripped out. Also the base CUDF validation makes sure escaped chars are correct
      // so we only need to worry about unquoted control chars

      auto state   = string_state::NORMAL;
      auto u_count = 0;
      for (SymbolOffsetT idx = start + 1; idx < end; idx++) {
        auto c = data[idx];
        if (!allow_unquoted_control_chars && static_cast<int>(c) >= 0 && static_cast<int>(c) < 32) {
          return false;
        }

        switch (state) {
          case string_state::NORMAL:
            if (c == '\\') { state = string_state::ESCAPED; }
            break;
          case string_state::ESCAPED:
            // in Spark you can allow any char to be escaped, but CUDF
            // validates it in some cases so we need to also validate it.
            if (c == 'u') {
              state   = string_state::ESCAPED_U;
              u_count = 0;
            } else if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                       c == 'r' || c == 't') {
              state = string_state::NORMAL;
            } else {
              return false;
            }
            break;
          case string_state::ESCAPED_U:
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
              u_count++;
              if (u_count == 4) {
                state   = string_state::NORMAL;
                u_count = 0;
              }
            } else {
              return false;
            }
            break;
        }
      }
      return string_state::NORMAL == state;
    });

  auto num_tokens = tokens.size();
  auto count_it   = thrust::make_counting_iterator(0);
  auto predicate  = [tokens        = tokens.begin(),
                    token_indices = token_indices.begin(),
                    validate_values,
                    validate_strings] __device__(auto i) -> bool {
    if (tokens[i] == token_t::ValueEnd) {
      return !validate_values(token_indices[i - 1], token_indices[i]);
    } else if (tokens[i] == token_t::FieldNameEnd || tokens[i] == token_t::StringEnd) {
      return !validate_strings(token_indices[i - 1], token_indices[i]);
    }
    return false;
  };

  using scan_type            = write_if::scan_type;
  auto conditional_write     = write_if{tokens.begin(), num_tokens};
  auto conditional_output_it = cudf::detail::make_tabulate_output_iterator(conditional_write);
  auto transform_op          = cuda::proclaim_return_type<scan_type>(
    [predicate, tokens = tokens.begin()] __device__(auto i) -> scan_type {
      if (predicate(i)) return {token_t::ErrorBegin, tokens[i] == token_t::LineEnd};
      return {static_cast<token_t>(tokens[i]), tokens[i] == token_t::LineEnd};
    });
  auto binary_op = cuda::proclaim_return_type<scan_type>(
    [] __device__(scan_type prev, scan_type curr) -> scan_type {
      auto op_result = (prev.first == token_t::ErrorBegin ? prev.first : curr.first);
      return scan_type((curr.second ? curr.first : op_result), prev.second | curr.second);
    });

  thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                   count_it,
                                   count_it + num_tokens,
                                   conditional_output_it,
                                   transform_op,
                                   binary_op);  // in-place scan
}
}  // namespace detail
}  // namespace cudf::io::json
