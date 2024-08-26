
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
        // printf("writing\n");
      }
    }
  }
};

enum class number_state {
  start = 0,
  saw_neg,  // not a complete state
  leading_zero,
  whole,
  saw_radix,  // not a complete state
  fraction,
  start_exponent,       // not a complete state
  after_sign_exponent,  // not a complete state
  exponent
};

enum class string_state {
  normal = 0,
  escaped,   // not a complete state
  escaped_u  // not a complete state
};

__device__ inline bool substr_eq(const char* data,
                                 SymbolOffsetT const start,
                                 SymbolOffsetT const end,
                                 SymbolOffsetT const expected_len,
                                 const char* expected)
{
  if (end - start != expected_len) {
    return false;
  } else {
    for (auto idx = 0; idx < expected_len; idx++) {
      if (data[start + idx] != expected[idx]) { return false; }
    }
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
  if (options.is_strict_validation()) {
    using token_t = cudf::io::json::token_t;
    cudf::detail::optional_trie trie_na =
      cudf::detail::create_serialized_trie(options.get_na_values(), stream);
    auto trie_na_view = cudf::detail::make_trie_view(trie_na);
    auto validate_values =
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
        auto num_state = number_state::start;
        for (auto at = start; at < end; at++) {
          c = data[at];
          switch (num_state) {
            case number_state::start:
              if ('-' == c) {
                num_state = number_state::saw_neg;
              } else if ('0' == c) {
                num_state = number_state::leading_zero;
              } else if (c >= '1' && c <= '9') {
                num_state = number_state::whole;
              } else {
                return false;
              }
              break;
            case number_state::saw_neg:
              if ('0' == c) {
                num_state = number_state::leading_zero;
              } else if (c >= '1' && c <= '9') {
                num_state = number_state::whole;
              } else if (allow_nonnumeric && 'I' == c) {
                return substr_eq(data, start, end, 4, "-INF") ||
                       substr_eq(data, start, end, 9, "-Infinity");
              } else {
                return false;
              }
              break;
            case number_state::leading_zero:
              if (allow_numeric_leading_zeros && c >= '0' && c <= '9') {
                num_state = number_state::whole;
              } else if ('.' == c) {
                num_state = number_state::saw_radix;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::start_exponent;
              } else {
                return false;
              }
              break;
            case number_state::whole:
              if (c >= '0' && c <= '9') {
                num_state = number_state::whole;
              } else if ('.' == c) {
                num_state = number_state::saw_radix;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::start_exponent;
              } else {
                return false;
              }
              break;
            case number_state::saw_radix:
              if (c >= '0' && c <= '9') {
                num_state = number_state::fraction;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::start_exponent;
              } else {
                return false;
              }
              break;
            case number_state::fraction:
              if (c >= '0' && c <= '9') {
                num_state = number_state::fraction;
              } else if ('e' == c || 'E' == c) {
                num_state = number_state::start_exponent;
              } else {
                return false;
              }
              break;
            case number_state::start_exponent:
              if ('+' == c || '-' == c) {
                num_state = number_state::after_sign_exponent;
              } else if (c >= '0' && c <= '9') {
                num_state = number_state::exponent;
              } else {
                return false;
              }
              break;
            case number_state::after_sign_exponent:
              if (c >= '0' && c <= '9') {
                num_state = number_state::exponent;
              } else {
                return false;
              }
              break;
            case number_state::exponent:
              if (c >= '0' && c <= '9') {
                num_state = number_state::exponent;
              } else {
                return false;
              }
              break;
          }
        }
        return num_state != number_state::after_sign_exponent &&
               num_state != number_state::start_exponent && num_state != number_state::saw_neg &&
               num_state != number_state::saw_radix;
      } else {
        return false;
      }
    };

    auto validate_strings =
      [data = d_input.data(),
       allow_unquoted_control_chars =
         options.is_allowed_unquoted_control_chars()] __device__(SymbolOffsetT start,
                                                                 SymbolOffsetT end) -> bool {
      // This validates a quoted string. A string must match https://www.json.org/json-en.html
      // but we already know that it has a starting and ending " and all white space has been
      // stripped out. Also the base CUDF validation makes sure escaped chars are correct
      // so we only need to worry about unquoted control chars

      auto state   = string_state::normal;
      auto u_count = 0;
      for (SymbolOffsetT idx = start + 1; idx < end; idx++) {
        auto c = data[idx];
        if (!allow_unquoted_control_chars && c < 32) { return false; }

        switch (state) {
          case string_state::normal:
            if (c == '\\') { state = string_state::escaped; }
            break;
          case string_state::escaped:
            // in Spark you can allow any char to be escaped, but CUDF
            // validates it in some cases so we need to also validate it.
            if (c == 'u') {
              state   = string_state::escaped_u;
              u_count = 0;
            } else if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                       c == 'r' || c == 't') {
              state = string_state::normal;
            } else {
              return false;
            }
            break;
          case string_state::escaped_u:
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
              u_count++;
              if (u_count == 4) {
                state   = string_state::normal;
                u_count = 0;
              }
            } else {
              return false;
            }
            break;
        }
      }
      return string_state::normal == state;
    };

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
    auto conditional_output_it = thrust::make_tabulate_output_iterator(conditional_write);
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
    rmm::device_uvector<int8_t> error(num_tokens, stream);
    thrust::transform(rmm::exec_policy(stream),
                      count_it,
                      count_it + num_tokens,
                      error.begin(),
                      predicate);  // in-place scan
    // printf("error:");
    // for (auto tk : cudf::detail::make_std_vector_sync(error, stream))
    //   printf("%d ", tk);
    // printf("\n");

    thrust::transform_inclusive_scan(rmm::exec_policy(stream),
                                     count_it,
                                     count_it + num_tokens,
                                     conditional_output_it,
                                     transform_op,
                                     binary_op);  // in-place scan
  }
  // printf("pre_process_token:");
  // for (auto tk : cudf::detail::make_std_vector_sync(device_span<PdaTokenT const>(tokens),
  // stream))
  //   printf("%d ", tk);
  // printf("\n");

  // LE SB FB FE VB VE SE LE SB ER LE SB LB VB VE SE LE LE
  // 1   0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  1  1
  // 1   1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  3  4
  // auto unary_op = [] __device__ (auto) -> token_t { return token_t::ErrorBegin; };
  // thrust::transform_if(rmm::exec_policy(stream), count_it, count_it + num_tokens, tokens.begin(),
  // unary_op, predicate); auto num_rows = thrust::count(rmm::exec_policy(stream), tokens.begin(),
  // tokens.end(), token_t::LineEnd); rmm::device_uvector<bool> row_is_error(num_rows, stream);
  // rmm::device_uvector<SymbolOffsetT> row_index(num_tokens, stream);
  // auto is_LineEnd = [] __device__ (auto token) -> SymbolOffsetT { return token ==
  // token_t::LineEnd; }; thrust::transform_inclusive_scan(rmm::exec_policy(stream),
  //   tokens.begin(), tokens.end(), row_index.begin(), is_LineEnd, thrust::plus<SymbolOffsetT>{});
  // auto is_error_it = thrust::make_transform_iterator(tokens.begin(), [] __device__ (auto token)
  // -> bool { return token == token_t::ErrorBegin; });
  // thrust::reduce_by_key(rmm::exec_policy(stream), row_index.begin(), row_index.end(),
  // is_error_it, thrust::make_discard_iterator(), row_is_error.begin());

  // if current == ErrorBegin and tokens[i+1]==LE or i==n-1) then write ErrorBegin to tokens[i],
  // else nothing. if VB, or SB, then if validate(token[i], token[i+1])==false,
  //
  // Transform_if to errors tokens
  // Count LE (num_rows)
  // create int vector [num_rows], bool[num_rows]
  // TODO: fuse them together with single scan algorithm.
  // LE==1, to scan to row_index.
  // reduce_by_key, to check if it has any error, and number of non-errors tokens.
  // reduce them to get output tokens count.
  // copy_if -> use cub cub::DeviceSelect::If(d_data, d_num_selected_out, num_items, select_op)
}

// corner cases: empty LE,
// alternate Decoupled look back idea:
// count LineEnd, allocate bool[num_rows] as is_error.
// decoupled look back for LineEnd tokens for row indices,
//  transform_if to error tokens and write to bool[row_index] atomically (reduce and write within
//  warp/block?)
// decoupled look back for LineEnd tokens for row indices,
//  (if not error row & not lineEnd token) -> decoupled look back for output indices,
//  CopyIf (if not error row & not lineEnd token) write to output.
}  // namespace detail
}  // namespace cudf::io::json
