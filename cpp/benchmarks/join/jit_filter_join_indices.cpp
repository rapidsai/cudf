/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <random>
#include <vector>

class JitFilterJoinIndicesBench : public cudf::benchmark {
 private:
  std::unique_ptr<cudf::table> left_table;
  std::unique_ptr<cudf::table> right_table;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices;

 public:
  void SetUp(int64_t num_rows, double selectivity) 
  {
    // Create test tables with integer columns
    auto left_col0_data = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<int32_t>(i); });
    auto left_col1_data = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<int32_t>(i * 2); });

    auto left_col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
    auto left_col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
    
    thrust::copy(left_col0_data, left_col0_data + num_rows, 
                 left_col0->mutable_view().data<int32_t>());
    thrust::copy(left_col1_data, left_col1_data + num_rows,
                 left_col1->mutable_view().data<int32_t>());

    std::vector<std::unique_ptr<cudf::column>> left_columns;
    left_columns.push_back(std::move(left_col0));
    left_columns.push_back(std::move(left_col1));
    left_table = std::make_unique<cudf::table>(std::move(left_columns));

    // Create right table with similar pattern
    auto right_col0_data = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<int32_t>(i); });
    auto right_col1_data = cudf::detail::make_counting_transform_iterator(
      0, [selectivity](auto i) { 
        return static_cast<int32_t>(i * 2 - (selectivity > 0.5 ? 1 : 10)); 
      });

    auto right_col0 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
    auto right_col1 = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, num_rows);
    
    thrust::copy(right_col0_data, right_col0_data + num_rows,
                 right_col0->mutable_view().data<int32_t>());
    thrust::copy(right_col1_data, right_col1_data + num_rows,
                 right_col1->mutable_view().data<int32_t>());

    std::vector<std::unique_ptr<cudf::column>> right_columns;
    right_columns.push_back(std::move(right_col0));
    right_columns.push_back(std::move(right_col1));
    right_table = std::make_unique<cudf::table>(std::move(right_columns));

    // Create join indices (simulate all pairs matching from equality join)
    left_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(num_rows, cudf::get_default_stream());
    right_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(num_rows, cudf::get_default_stream());

    auto counting_iter = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<cudf::size_type>(i); });
    thrust::copy(counting_iter, counting_iter + num_rows, left_indices->begin());
    thrust::copy(counting_iter, counting_iter + num_rows, right_indices->begin());
  }

  void BenchmarkJitFilterJoinIndices(nvbench::state& state,
                                     cudf::join_kind join_kind,
                                     std::string const& predicate_code)
  {
    auto const num_rows = static_cast<int64_t>(state.get_int64("num_rows"));
    auto const selectivity = state.get_float64("selectivity");
    
    SetUp(num_rows, selectivity);

    cudf::device_span<cudf::size_type const> left_span{left_indices->data(), left_indices->size()};
    cudf::device_span<cudf::size_type const> right_span{right_indices->data(), right_indices->size()};

    state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto [filtered_left, filtered_right] = cudf::jit_filter_join_indices(
        left_table->view(),
        right_table->view(),
        left_span,
        right_span,
        predicate_code,
        join_kind);
    });

    state.add_buffer_size(num_rows * sizeof(cudf::size_type), "input_indices");
    state.add_buffer_size(filtered_left->size() * sizeof(cudf::size_type), "output_indices"); 
  }
};

void jit_filter_join_indices_inner_join(nvbench::state& state)
{
  JitFilterJoinIndicesBench benchmark;
  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";
  benchmark.BenchmarkJitFilterJoinIndices(state, cudf::join_kind::INNER_JOIN, predicate_code);
}

void jit_filter_join_indices_left_join(nvbench::state& state)
{
  JitFilterJoinIndicesBench benchmark;
  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";
  benchmark.BenchmarkJitFilterJoinIndices(state, cudf::join_kind::LEFT_JOIN, predicate_code);
}

void jit_filter_join_indices_full_join(nvbench::state& state)
{
  JitFilterJoinIndicesBench benchmark;
  std::string predicate_code = R"(
    __device__ bool predicate(int32_t left_val, int32_t right_val) {
      return left_val > right_val;
    }
  )";
  benchmark.BenchmarkJitFilterJoinIndices(state, cudf::join_kind::FULL_JOIN, predicate_code);
}

NVBENCH_BENCH(jit_filter_join_indices_inner_join)
  .set_name("jit_filter_join_indices_inner")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000})
  .add_float64_axis("selectivity", {0.1, 0.5, 0.9});

NVBENCH_BENCH(jit_filter_join_indices_left_join)
  .set_name("jit_filter_join_indices_left") 
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000})
  .add_float64_axis("selectivity", {0.1, 0.5, 0.9});

NVBENCH_BENCH(jit_filter_join_indices_full_join)
  .set_name("jit_filter_join_indices_full")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000})
  .add_float64_axis("selectivity", {0.1, 0.5, 0.9});
