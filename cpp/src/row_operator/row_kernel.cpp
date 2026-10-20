/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "row_kernel.hpp"

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_fragments.hpp>
#include <row_fragments.hpp>
#include <runtime/context.hpp>

#include <algorithm>
#include <string_view>
#include <vector>
namespace cudf::jit {
namespace {
std::span<uint8_t const> fragment(std::size_t id)
{
  auto const range = cudf_fragments::file_ranges[id];
  return cudf_fragments::files.subspan(range[0], range[1]);
}
}  // namespace
void validate_row_column(column_view column)
{
  if (column.type().id() == type_id::DICTIONARY32) {
    CUDF_EXPECTS(is_index_type(column.child(dictionary_indices_column_index).type()),
                 "Unsupported dictionary index width",
                 std::invalid_argument);
    column = column.child(dictionary_keys_column_index);
  }
  if (column.type().id() == type_id::LIST) {
    CUDF_EXPECTS(column.child(offsets_column_index).type().id() == type_id::INT32,
                 "List offsets must be INT32",
                 std::invalid_argument);
    column = column.child(lists_column_view::child_column_index);
  }
  CUDF_EXPECTS(is_fixed_width(column.type()) || column.type().id() == type_id::STRING,
               "Row operators support T, list<T>, dictionary<T>, and dictionary<list<T>>",
               std::invalid_argument);
}
kernel get_row_kernel(std::string const& source,
                      row_execution execution,
                      std::span<rtcx::memory_fragment const> dependencies,
                      std::optional<std::span<std::string const>> required_symbols)
{
  char const* names[]{"cudf/detail/operation_udf.cuh"};
  char const* headers[]{source.c_str()};
  auto row = get_kernel_fragment(
    execution == row_execution::TRANSFORM ? "transform_operator" : "predicate_operator",
    "cudf/cpp/src/jit/row_operator.cu",
    names,
    headers,
    execution == row_execution::TRANSFORM ? "transform_operator" : "predicate_operator");
  return link_row_kernel(row->view(), execution, source, dependencies, required_symbols);
}
kernel link_row_kernel(std::span<uint8_t const> row,
                       row_execution execution,
                       std::string_view source,
                       std::span<rtcx::memory_fragment const> dependencies,
                       std::optional<std::span<std::string const>> required_symbols)
{
  std::vector<rtcx::memory_fragment> fragments;
  auto add = [&](std::size_t id) {
    fragments.push_back({.data = fragment(id),
                         .type = rtcx::binary_type::FATBIN,
                         .name = cudf_fragments::file_ids[id]});
  };
  add(execution == row_execution::TRANSFORM ? cudf_fragments::row_transform
                                            : cudf_fragments::row_join);
  if (required_symbols.has_value()) {
    for (auto const& symbol : *required_symbols) {
      auto const found = std::ranges::lower_bound(row_fragments, symbol, {}, &row_fragment::symbol);
      CUDF_EXPECTS(found != row_fragments.end() && found->symbol == symbol,
                   "Unknown row ABI symbol: " + symbol,
                   std::invalid_argument);
      add(found->id);
    }
  } else {
    // Opaque CUDA/PTX/LTO dependencies do not expose their ABI imports to RowIR.
    add(cudf_fragments::row_access);
    if (source.find("cudf_row_op_") != std::string_view::npos || !dependencies.empty()) {
      add(cudf_fragments::row_operators);
    }
  }
  if (!row.empty()) {
    fragments.push_back({.data = row, .type = rtcx::binary_type::LTO_IR, .name = "operator"});
  }
  fragments.insert(fragments.end(), dependencies.begin(), dependencies.end());
  return get_lto_linked_kernel(
    execution == row_execution::TRANSFORM ? "row_transform_v1" : "row_join_v1", {}, fragments);
}
}  // namespace cudf::jit
