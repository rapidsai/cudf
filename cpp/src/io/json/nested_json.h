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

namespace cudf {
namespace io {
namespace json {
namespace gpu {

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

/// Type used to represent a (input-symbol, stack-symbole)-tuple in stack-symbole-major order
using PdaSymbolGroupIdT = char;

/// Type being emitted by the pushdown automaton transducer
using PdaTokenT = char;

/**
 * @brief Tokens emitted while parsing a JSON input
 */
enum token_t : PdaTokenT {
  /// Beginning-of-struct token (on encounter of semantic '{')
  TK_BOS,
  /// Beginning-of-list token (on encounter of semantic '[')
  TK_BOL,
  /// Beginning-of-error token (on first encounter of a parsing error)
  TK_ERR,
  /// Beginning-of-string-value token (on encounter of the string's first quote)
  TK_BST,
  /// Beginning-of-value token (first character of literal or numeric)
  TK_BOV,
  /// End-of-list token (on encounter of semantic ']')
  TK_EOL,
  /// End-of-struct token (on encounter of semantic '}')
  TK_EOS,
  /// Beginning-of-field-name token (on encounter of first quote)
  TK_BFN,
  /// Post-value token (first character after a literal or numeric string)
  TK_POV,
  /// End-of-string token (on encounter of a string's second quote)
  TK_EST,
  /// End-of-field-name token (on encounter of a field name's second quote)
  TK_EFN,
  /// Total number of tokens
  NUM_TOKENS
};

/**
 * @brief Identifies the stack context for each character from a JSON input. Specifically, we
 * identify brackets and braces outside of quoted fields (e.g., field names, strings).
 * At this stage, we do not perform bracket matching, i.e., we do not verify whether a closing
 * bracket would actually pop a the corresponding opening brace.
 *
 * @param d_json_in The string of input characters
 * @param d_top_of_stack
 * @param stream The cuda stream to dispatch GPU kernels to
 */
void get_stack_context(device_span<SymbolT const> d_json_in,
                       device_span<SymbolT> d_top_of_stack,
                       rmm::cuda_stream_view stream);

/**
 * @brief Parses the given JSON string and emits a sequence of tokens that demarcate relevant
 * sections from the input.
 *
 * @param d_json_in The JSON input
 * @param d_tokens_out Device memory to which the parsed tokens are written
 * @param d_tokens_indices Device memory to which the indices are written, where each index
 * represents the offset within \p d_json_in that cause the input being written
 * @param d_num_written_tokens The total number of tokens that were parsed
 * @param stream The CUDA stream to which kernels are dispatched
 */
void get_token_stream(device_span<SymbolT const> d_json_in,
                      device_span<PdaTokenT> d_tokens,
                      device_span<SymbolOffsetT> d_tokens_indices,
                      SymbolOffsetT* d_num_written_tokens,
                      rmm::cuda_stream_view stream);

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
