/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::json {

/// Type used to represent the atomic symbol type used within the finite-state machine
using SymbolT = char;

/// Type used to represent the stack alphabet (i.e.: empty-stack, struct, list)
using StackSymbolT = char;

/// Type used to index into the symbols within the JSON input
using SymbolOffsetT = uint32_t;

/// Type large enough to support indexing up to max nesting level (must be signed)
using StackLevelT = int8_t;

/// Type used to represent a symbol group id of the input alphabet in the pushdown automaton
using PdaInputSymbolGroupIdT = char;

/// Type used to represent a symbol group id of the stack alphabet in the pushdown automaton
using PdaStackSymbolGroupIdT = char;

/// Type used to represent a (input-symbol, stack-symbol)-tuple in stack-symbol-major order
using PdaSymbolGroupIdT = char;

/// Type being emitted by the pushdown automaton transducer
using PdaTokenT = char;

/**
 * @brief Tokens emitted while parsing a JSON input
 */
enum token_t : PdaTokenT {
  /// Beginning-of-struct token (on encounter of semantic '{')
  StructBegin,
  /// End-of-struct token (on encounter of semantic '}')
  StructEnd,
  /// Beginning-of-list token (on encounter of semantic '[')
  ListBegin,
  /// End-of-list token (on encounter of semantic ']')
  ListEnd,
  /// Beginning-of-field-name token (on encounter of first quote)
  FieldNameBegin,
  /// End-of-field-name token (on encounter of a field name's second quote)
  FieldNameEnd,
  /// Beginning-of-string-value token (on encounter of the string's first quote)
  StringBegin,
  /// End-of-string token (on encounter of a string's second quote)
  StringEnd,
  /// Beginning-of-value token (first character of literal or numeric)
  ValueBegin,
  /// Post-value token (first character after a literal or numeric string)
  ValueEnd,
  /// Beginning-of-error token (on first encounter of a parsing error)
  ErrorBegin,
  /// Total number of tokens
  NUM_TOKENS
};

namespace detail {
/**
 * @brief Identifies the stack context for each character from a JSON input. Specifically, we
 * identify brackets and braces outside of quoted fields (e.g., field names, strings).
 * At this stage, we do not perform bracket matching, i.e., we do not verify whether a closing
 * bracket would actually pop a the corresponding opening brace.
 *
 * @param[in] d_json_in The string of input characters
 * @param[out] d_top_of_stack Will be populated with what-is-on-top-of-the-stack for any given input
 * character of \p d_json_in, where a '{' represents that the corresponding input character is
 * within the context of a struct, a '[' represents that it is within the context of an array, and a
 * '_' symbol that it is at the root of the JSON.
 * @param[in] stream The cuda stream to dispatch GPU kernels to
 */
void get_stack_context(device_span<SymbolT const> d_json_in,
                       SymbolT* d_top_of_stack,
                       rmm::cuda_stream_view stream);

/**
 * @brief Parses the given JSON string and emits a sequence of tokens that demarcate relevant
 * sections from the input.
 *
 * @param[in] d_json_in The JSON input
 * @param[out] d_tokens Device memory to which the parsed tokens are written
 * @param[out] d_tokens_indices Device memory to which the indices are written, where each index
 * represents the offset within \p d_json_in that cause the input being written
 * @param[out] d_num_written_tokens The total number of tokens that were parsed
 * @param[in] stream The CUDA stream to which kernels are dispatched
 */
void get_token_stream(device_span<SymbolT const> d_json_in,
                      PdaTokenT* d_tokens,
                      SymbolOffsetT* d_tokens_indices,
                      SymbolOffsetT* d_num_written_tokens,
                      rmm::cuda_stream_view stream);
}  // namespace detail

}  // namespace cudf::io::json
