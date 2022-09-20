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

#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <map>
#include <vector>

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

/// Type used to represent the class of a node (or a node "category") within the tree representation
using NodeT = char;

/// Type used to index into the nodes within the tree of structs, lists, field names, and value
/// nodes
using NodeIndexT = uint32_t;

/// Type large enough to represent tree depth from [0, max-tree-depth); may be an unsigned type
using TreeDepthT = StackLevelT;

/**
 * @brief Struct that encapsulate all information of a columnar tree representation.
 */
struct tree_meta_t {
  rmm::device_uvector<NodeT> node_categories;
  rmm::device_uvector<NodeIndexT> parent_node_ids;
  rmm::device_uvector<TreeDepthT> node_levels;
  rmm::device_uvector<SymbolOffsetT> node_range_begin;
  rmm::device_uvector<SymbolOffsetT> node_range_end;
};

constexpr NodeIndexT parent_node_sentinel = std::numeric_limits<NodeIndexT>::max();

/**
 * @brief Class of a node (or a node "category") within the tree representation
 */
enum node_t : NodeT {
  /// A node representing a struct
  NC_STRUCT,
  /// A node representing a list
  NC_LIST,
  /// A node representing a field name
  NC_FN,
  /// A node representing a string value
  NC_STR,
  /// A node representing a numeric or literal value (e.g., true, false, null)
  NC_VAL,
  /// A node representing a parser error
  NC_ERR,
  /// Total number of node classes
  NUM_NODE_CLASSES
};

/**
 * @brief A column type
 */
enum class json_col_t : char { ListColumn, StructColumn, StringColumn, Unknown };

/**
 * @brief Intermediate representation of data from a nested JSON input
 */
struct json_column {
  // Type used to count number of rows
  using row_offset_t = size_type;

  // The inferred type of this column (list, struct, or value/string column)
  json_col_t type = json_col_t::Unknown;

  std::vector<row_offset_t> string_offsets;
  std::vector<row_offset_t> string_lengths;

  // Row offsets
  std::vector<row_offset_t> child_offsets;

  // Validity bitmap
  std::vector<bitmask_type> validity;
  row_offset_t valid_count = 0;

  // Map of child columns, if applicable.
  // Following "items" as the default child column's name of a list column
  // Using the struct's field names
  std::map<std::string, json_column> child_columns;
  std::vector<std::string> column_order;

  // Counting the current number of items in this column
  row_offset_t current_offset = 0;

  json_column()                    = default;
  json_column(json_column&& other) = default;
  json_column& operator=(json_column&&) = default;
  json_column(const json_column&)       = delete;
  json_column& operator=(const json_column&) = delete;

  /**
   * @brief Fills the rows up to the given \p up_to_row_offset with nulls.
   *
   * @param up_to_row_offset The row offset up to which to fill with nulls.
   */
  void null_fill(row_offset_t up_to_row_offset);

  /**
   * @brief Recursively iterates through the tree of columns making sure that all child columns of a
   * struct column have the same row count, filling missing rows with nulls.
   *
   * @param min_row_count The minimum number of rows to be filled.
   */
  void level_child_cols_recursively(row_offset_t min_row_count);

  /**
   * @brief Appends the row at the given index to the column, filling all rows between the column's
   * current offset and the given \p row_index with null items.
   *
   * @param row_index The row index at which to insert the given row
   * @param row_type The row's type
   * @param string_offset The string offset within the original JSON input of this item
   * @param string_end The one-past-the-last-char offset within the original JSON input of this item
   * @param child_count In case of a list column, this row's number of children is used to compute
   * the offsets
   */
  void append_row(uint32_t row_index,
                  json_col_t row_type,
                  uint32_t string_offset,
                  uint32_t string_end,
                  uint32_t child_count);
};

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
  // Beginning-of-struct-member token
  StructMemberBegin,
  // End-of-struct-member token
  StructMemberEnd,
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

// TODO: return device_uvector instead of passing pre-allocated memory
/**
 * @brief Identifies the stack context for each character from a JSON input. Specifically, we
 * identify brackets and braces outside of quoted fields (e.g., field names, strings).
 * At this stage, we do not perform bracket matching, i.e., we do not verify whether a closing
 * bracket would actually pop a the corresponding opening brace.
 *
 * @param[in] json_in The string of input characters
 * @param[out] d_top_of_stack Will be populated with what-is-on-top-of-the-stack for any given input
 * character of \p d_json_in, where a '{' represents that the corresponding input character is
 * within the context of a struct, a '[' represents that it is within the context of an array, and a
 * '_' symbol that it is at the root of the JSON.
 * @param[in] stream The cuda stream to dispatch GPU kernels to
 */
void get_stack_context(device_span<SymbolT const> json_in,
                       SymbolT* d_top_of_stack,
                       rmm::cuda_stream_view stream);

/**
 * @brief Parses the given JSON string and emits a sequence of tokens that demarcate relevant
 * sections from the input.
 *
 * @param json_in The JSON input
 * @param options Parsing options specifying the parsing behaviour
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return Pair of device vectors, where the first vector represents the token types and the second
 * vector represents the index within the input corresponding to each token
 */
std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> get_token_stream(
  device_span<SymbolT const> json_in,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Parses the given JSON string and generates a tree representation of the given input.
 *
 * @param tokens Vector of token types in the json string
 * @param token_indices The indices within the input string corresponding to each token
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return A tree representation of the input JSON string as vectors of node type, parent index,
 * level, begin index, and end index in the input JSON string
 */
tree_meta_t get_tree_representation(
  device_span<PdaTokenT const> tokens,
  device_span<SymbolOffsetT const> token_indices,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Parses the given JSON string and generates table from the given input.
 *
 * @param input The JSON input
 * @param options Parsing options specifying the parsing behaviour
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return The data parsed from the given JSON input
 */
table_with_metadata parse_nested_json(
  host_span<SymbolT const> input,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail

}  // namespace cudf::io::json
