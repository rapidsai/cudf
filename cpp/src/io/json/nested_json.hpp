/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/resource_ref.hpp>

#include <map>
#include <vector>

// Forward declaration of parse_options from parsing_utils.cuh
namespace cudf::io {
struct parse_options;
}
namespace cudf::io::json {

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

/**
 * @brief A column type
 */
enum class json_col_t : char { ListColumn, StructColumn, StringColumn, Unknown };

/**
 * @brief Enum class to specify whether we just push onto and pop from the stack or whether we also
 * reset to an empty stack on a newline character.
 */
enum class stack_behavior_t : char {
  /// Opening brackets and braces, [, {, push onto the stack, closing brackets and braces, ], }, pop
  /// from the stack
  PushPopWithoutReset,

  /// Opening brackets and braces, [, {, push onto the stack, closing brackets and braces, ], }, pop
  /// from the stack. Delimiter characters are passed when the stack context is constructed to
  /// reset to an empty stack.
  ResetOnDelimiter
};

// Default name for a list's child column
constexpr auto list_child_name{"element"};

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

  json_column()                              = default;
  json_column(json_column&& other)           = default;
  json_column& operator=(json_column&&)      = default;
  json_column(json_column const&)            = delete;
  json_column& operator=(json_column const&) = delete;

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
 * @brief Intermediate representation of data from a nested JSON input, in device memory.
 * Device memory equivalent of `json_column`.
 */
struct device_json_column {
  // Type used to count number of rows
  using row_offset_t = size_type;

  // The inferred type of this column (list, struct, or value/string column)
  json_col_t type = json_col_t::Unknown;

  rmm::device_uvector<row_offset_t> string_offsets;
  rmm::device_uvector<row_offset_t> string_lengths;

  // Row offsets
  rmm::device_uvector<row_offset_t> child_offsets;

  // Validity bitmap
  rmm::device_buffer validity;

  // Map of child columns, if applicable.
  // Following "element" as the default child column's name of a list column
  // Using the struct's field names
  std::map<std::string, device_json_column> child_columns;
  std::vector<std::string> column_order;
  // Counting the current number of items in this column
  row_offset_t num_rows = 0;
  // Force as string column
  bool forced_as_string_column{false};

  /**
   * @brief Construct a new d json column object
   *
   * @note `mr` is used for allocating the device memory for child_offsets, and validity
   * since it will moved into cudf::column later.
   *
   * @param stream The CUDA stream to which kernels are dispatched
   * @param mr Optional, resource with which to allocate
   */
  device_json_column(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : string_offsets(0, stream),
      string_lengths(0, stream),
      child_offsets(0, stream, mr),
      validity(0, stream, mr)
  {
  }
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
 * @param[in] stack_behavior Specifies the stack's behavior
 * @param[in] delimiter Specifies the delimiter to use as separator for JSON lines input
 * @param[in] stream The cuda stream to dispatch GPU kernels to
 */
void get_stack_context(device_span<SymbolT const> json_in,
                       SymbolT* d_top_of_stack,
                       stack_behavior_t stack_behavior,
                       SymbolT delimiter,
                       rmm::cuda_stream_view stream);

/**
 * @brief Post-processes a token stream that may contain tokens from invalid lines. Expects that the
 * token stream begins with a LineEnd token.
 *
 * @param tokens The tokens to be post-processed
 * @param token_indices The tokens' corresponding indices that are post-processed
 * @param stream The cuda stream to dispatch GPU kernels to
 * @return Returns the post-processed token stream
 */
std::pair<rmm::device_uvector<PdaTokenT>, rmm::device_uvector<SymbolOffsetT>> process_token_stream(
  device_span<PdaTokenT const> tokens,
  device_span<SymbolOffsetT const> token_indices,
  rmm::cuda_stream_view stream);

/**
 * @brief Parses the given JSON string and generates a tree representation of the given input.
 *
 * @param tokens Vector of token types in the json string
 * @param token_indices The indices within the input string corresponding to each token
 * @param is_strict_nested_boundaries Whether to extract node end of nested types strictly
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return A tree representation of the input JSON string as vectors of node type, parent index,
 * level, begin index, and end index in the input JSON string
 */
tree_meta_t get_tree_representation(device_span<PdaTokenT const> tokens,
                                    device_span<SymbolOffsetT const> token_indices,
                                    bool is_strict_nested_boundaries,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @brief Traverse the tree representation of the JSON input in records orient format and populate
 * the output columns indices and row offsets within that column.
 *
 * @param d_input The JSON input
 * @param d_tree A tree representation of the input JSON string as vectors of node type, parent
 * index, level, begin index, and end index in the input JSON string
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param is_enabled_lines Whether the input is a line-delimited JSON
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return A tuple of the output column indices and the row offsets within each column for each node
 */
std::tuple<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
records_orient_tree_traversal(device_span<SymbolT const> d_input,
                              tree_meta_t const& d_tree,
                              bool is_array_of_arrays,
                              bool is_enabled_lines,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Searches for and selects nodes at level `row_array_children_level`. For each selected
 * node, the function outputs the original index of that node (i.e., the nodes index within
 * `node_levels`) and also generates the child index of that node relative to other children of the
 * same parent. E.g., the child indices of the following string nodes relative to their respective
 * list parents are: `[["a", "b", "c"], ["d", "e"]]`: `"a": 0, "b": 1, "c": 2, "d": 0, "e": 1`.
 *
 * @param row_array_children_level Level of the nodes to search for
 * @param node_levels Levels of each node in the tree
 * @param parent_node_ids Parent node ids of each node in the tree
 * @param stream The CUDA stream to which kernels are dispatched
 * @return A pair of device_uvector containing the original node indices and their corresponding
 * child index
 */
std::pair<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<NodeIndexT>>
get_array_children_indices(TreeDepthT row_array_children_level,
                           device_span<TreeDepthT const> node_levels,
                           device_span<NodeIndexT const> parent_node_ids,
                           rmm::cuda_stream_view stream);
/**
 * @brief Reduce node tree into column tree by aggregating each property of column.
 *
 * @param tree json node tree to reduce (modified in-place, but restored to original state)
 * @param col_ids column ids of each node (modified in-place, but restored to original state)
 * @param row_offsets row offsets of each node (modified in-place, but restored to original state)
 * @param stream The CUDA stream to which kernels are dispatched
 * @return A tuple containing the column tree, identifier for each column and the maximum row index
 * in each column
 */
std::tuple<tree_meta_t, rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
reduce_to_column_tree(tree_meta_t& tree,
                      device_span<NodeIndexT> col_ids,
                      device_span<size_type> row_offsets,
                      rmm::cuda_stream_view stream);

/**
 * @brief Retrieves the parse_options to be used for type inference and type casting
 *
 * @param options The reader options to influence the relevant type inference and type casting
 * options
 * @param stream The CUDA stream to which kernels are dispatched
 */
cudf::io::parse_options parsing_options(cudf::io::json_reader_options const& options,
                                        rmm::cuda_stream_view stream);

/**
 * @brief Parses the given JSON string and generates table from the given input.
 *
 * All processing is done in device memory.
 *
 * @param input The JSON input
 * @param options Parsing options specifying the parsing behaviour
 * @param stream The CUDA stream to which kernels are dispatched
 * @param mr Optional, resource with which to allocate
 * @return The data parsed from the given JSON input
 */
table_with_metadata device_parse_nested_json(device_span<SymbolT const> input,
                                             cudf::io::json_reader_options const& options,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @brief Get the path data type of a column by path if present in input schema
 *
 * @param path path of the column
 * @param options json reader options which holds schema
 * @return data type of the column if present
 */
std::optional<data_type> get_path_data_type(
  host_span<std::pair<std::string, cudf::io::json::NodeT> const> path,
  cudf::io::json_reader_options const& options);

/**
 * @brief Helper class to get path of a column by column id from reduced column tree
 *
 */
struct path_from_tree {
  host_span<NodeT const> column_categories;
  host_span<NodeIndexT const> column_parent_ids;
  host_span<std::string const> column_names;
  bool is_array_of_arrays;
  NodeIndexT const row_array_parent_col_id;

  using path_rep = std::pair<std::string, cudf::io::json::NodeT>;
  std::vector<path_rep> get_path(NodeIndexT this_col_id);
};

}  // namespace detail

}  // namespace cudf::io::json
