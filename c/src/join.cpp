/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/exceptions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.h>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
std::unique_ptr<cudf::table>& get_table_owner(cudfTable_t table)
{
  return *reinterpret_cast<std::unique_ptr<cudf::table>*>(table->addr);
}

cudf::table& get_table(cudfTable_t table) { return *get_table_owner(table); }

cudf::table_view select_keys(cudf::table_view table, int32_t const* indices, int32_t num_keys)
{
  std::vector<cudf::column_view> selected;
  selected.reserve(num_keys);
  for (int32_t i = 0; i < num_keys; ++i) {
    selected.push_back(table.column(indices[i]));
  }
  return cudf::table_view{selected};
}

std::unique_ptr<cudf::column> make_gather_map(
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices)
{
  auto const size = static_cast<cudf::size_type>(indices->size());
  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, size, indices->release(), rmm::device_buffer{}, 0);
}
}  // namespace

extern "C" cudfError_t cudfTableJoin(cudfTable_t left,
                                     cudfTable_t right,
                                     const int32_t* left_on,
                                     const int32_t* right_on,
                                     int32_t num_keys,
                                     cudfJoinType_t join_type,
                                     cudaStream_t stream,
                                     cudfTable_t* result)
{
  return cudf::c::translate_exceptions([=] {
    if (join_type == CUDF_JOIN_LEFT || join_type == CUDF_JOIN_FULL) {
      CUDF_FAIL("cudfTableJoin: LEFT and FULL join types are not yet implemented");
    }

    CUDF_EXPECTS(result != nullptr, "output cudfTable_t pointer cannot be null");
    CUDF_EXPECTS(left_on != nullptr, "left_on pointer cannot be null");
    CUDF_EXPECTS(right_on != nullptr, "right_on pointer cannot be null");
    CUDF_EXPECTS(num_keys >= 0, "num_keys must be non-negative");

    auto stream_view = rmm::cuda_stream_view{stream};
    auto left_view   = get_table(left).view();
    auto right_view  = get_table(right).view();

    auto left_selected  = select_keys(left_view, left_on, num_keys);
    auto right_selected = select_keys(right_view, right_on, num_keys);

    auto [left_indices, right_indices] =
      cudf::inner_join(left_selected, right_selected, cudf::null_equality::EQUAL, stream_view);

    auto left_map    = make_gather_map(std::move(left_indices));
    auto right_map   = make_gather_map(std::move(right_indices));
    auto left_result = cudf::gather(
      left_view, left_map->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream_view);
    auto right_result = cudf::gather(
      right_view, right_map->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream_view);

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(left_result->num_columns() + right_result->num_columns());

    for (auto& column : left_result->release()) {
      columns.push_back(std::move(column));
    }
    for (auto& column : right_result->release()) {
      columns.push_back(std::move(column));
    }

    auto owner =
      new std::unique_ptr<cudf::table>(std::make_unique<cudf::table>(std::move(columns)));
    *result = new cudfTable{reinterpret_cast<uintptr_t>(owner)};
  });
}
