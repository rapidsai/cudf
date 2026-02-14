/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <vector>

void jit_filter_join_indices_inner_join(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  // Create test tables with integer columns
  auto left_col0  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto left_col1  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto right_col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto right_col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);

  std::vector<std::unique_ptr<cudf::column>> left_columns;
  left_columns.push_back(std::move(left_col0));
  left_columns.push_back(std::move(left_col1));
  auto left_table = cudf::table(std::move(left_columns));

  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.push_back(std::move(right_col0));
  right_columns.push_back(std::move(right_col1));
  auto right_table = cudf::table(std::move(right_columns));

  // Create join indices (simulate all pairs matching from equality join)
  std::vector<cudf::size_type> indices_h(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    indices_h[i] = i;
  }
  auto left_indices_d = cudf::detail::make_device_uvector_async(
    indices_h, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_d = cudf::detail::make_device_uvector_async(
    indices_h, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  cudf::device_span<cudf::size_type const> left_span{left_indices_d.data(), left_indices_d.size()};
  cudf::device_span<cudf::size_type const> right_span{right_indices_d.data(),
                                                      right_indices_d.size()};

  // Predicate: left.col1 > right.col1 (receives output pointer, then all columns: left cols, then
  // right cols)
  std::string predicate_code = R"(
    __device__ void predicate(bool* output, int32_t left_col0, int32_t left_col1,
                              int32_t right_col0, int32_t right_col1) {
      *output = left_col1 > right_col1;
    }
  )";

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto [filtered_left, filtered_right] =
      cudf::jit_filter_join_indices(left_table.view(),
                                    right_table.view(),
                                    left_span,
                                    right_span,
                                    predicate_code,
                                    cudf::join_kind::INNER_JOIN);
  });
}

void jit_filter_join_indices_left_join(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto left_col0  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto left_col1  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto right_col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
  auto right_col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);

  std::vector<std::unique_ptr<cudf::column>> left_columns;
  left_columns.push_back(std::move(left_col0));
  left_columns.push_back(std::move(left_col1));
  auto left_table = cudf::table(std::move(left_columns));

  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.push_back(std::move(right_col0));
  right_columns.push_back(std::move(right_col1));
  auto right_table = cudf::table(std::move(right_columns));

  std::vector<cudf::size_type> indices_h(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    indices_h[i] = i;
  }
  auto left_indices_d = cudf::detail::make_device_uvector_async(
    indices_h, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto right_indices_d = cudf::detail::make_device_uvector_async(
    indices_h, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  cudf::device_span<cudf::size_type const> left_span{left_indices_d.data(), left_indices_d.size()};
  cudf::device_span<cudf::size_type const> right_span{right_indices_d.data(),
                                                      right_indices_d.size()};

  std::string predicate_code = R"(
    __device__ void predicate(bool* output, int32_t left_col0, int32_t left_col1,
                              int32_t right_col0, int32_t right_col1) {
      *output = left_col1 > right_col1;
    }
  )";

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto [filtered_left, filtered_right] =
      cudf::jit_filter_join_indices(left_table.view(),
                                    right_table.view(),
                                    left_span,
                                    right_span,
                                    predicate_code,
                                    cudf::join_kind::LEFT_JOIN);
  });
}

NVBENCH_BENCH(jit_filter_join_indices_inner_join)
  .set_name("jit_filter_join_indices_inner")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000});

NVBENCH_BENCH(jit_filter_join_indices_left_join)
  .set_name("jit_filter_join_indices_left")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000});
