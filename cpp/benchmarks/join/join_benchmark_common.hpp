/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>
#include <nvbench/nvbench.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/ast/expressions.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <vector>

#include "generate_input_tables.cuh"

template <typename key_type,
          typename payload_type,
          bool Nullable,
          bool is_conditional = false,
          typename state_type,
          typename Join>
static void BM_join(state_type& state, Join JoinFunc)
{
  auto const build_table_size = [&]() {
    if constexpr (std::is_same_v<state_type, benchmark::State>) {
      return static_cast<cudf::size_type>(state.range(0));
    }
    if constexpr (std::is_same_v<state_type, nvbench::state>) {
      return static_cast<cudf::size_type>(state.get_int64("Build Table Size"));
    }
  }();
  auto const probe_table_size = [&]() {
    if constexpr (std::is_same_v<state_type, benchmark::State>) {
      return static_cast<cudf::size_type>(state.range(1));
    }
    if constexpr (std::is_same_v<state_type, nvbench::state>) {
      return static_cast<cudf::size_type>(state.get_int64("Probe Table Size"));
    }
  }();

  const double selectivity = 0.3;
  const int multiplicity   = 1;

  // Generate build and probe tables
  cudf::test::UniformRandomGenerator<cudf::size_type> rand_gen(0, build_table_size);
  auto build_random_null_mask = [&rand_gen](int size) {
    // roughly 75% nulls
    auto validity = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [&rand_gen](auto i) { return (rand_gen.generate() & 3) == 0; });
    return cudf::test::detail::make_null_mask(validity, validity + size);
  };

  std::unique_ptr<cudf::column> build_key_column = [&]() {
    return Nullable ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                build_table_size,
                                                build_random_null_mask(build_table_size))
                    : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                build_table_size);
  }();
  std::unique_ptr<cudf::column> probe_key_column = [&]() {
    return Nullable ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                probe_table_size,
                                                build_random_null_mask(probe_table_size))
                    : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                probe_table_size);
  }();

  generate_input_tables<key_type, cudf::size_type>(
    build_key_column->mutable_view().data<key_type>(),
    build_table_size,
    probe_key_column->mutable_view().data<key_type>(),
    probe_table_size,
    selectivity,
    multiplicity);

  auto payload_data_it = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<payload_type> build_payload_column(
    payload_data_it, payload_data_it + build_table_size);

  cudf::test::fixed_width_column_wrapper<payload_type> probe_payload_column(
    payload_data_it, payload_data_it + probe_table_size);

  CHECK_CUDA(0);

  cudf::table_view build_table({build_key_column->view(), build_payload_column});
  cudf::table_view probe_table({probe_key_column->view(), probe_payload_column});

  // Setup join parameters and result table
  [[maybe_unused]] std::vector<cudf::size_type> columns_to_join = {0};

  // Benchmark the inner join operation
  if constexpr (std::is_same_v<state_type, benchmark::State> and (not is_conditional)) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true, rmm::cuda_stream_default);

      auto result = JoinFunc(
        probe_table, build_table, columns_to_join, columns_to_join, cudf::null_equality::UNEQUAL);
    }
  }
  if constexpr (std::is_same_v<state_type, nvbench::state> and (not is_conditional)) {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      rmm::cuda_stream_view stream_view{launch.get_stream()};
      auto result = JoinFunc(probe_table,
                             build_table,
                             columns_to_join,
                             columns_to_join,
                             cudf::null_equality::UNEQUAL,
                             stream_view);
    });
  }

  // Benchmark conditional join
  if constexpr (std::is_same_v<state_type, benchmark::State> and is_conditional) {
    // Common column references.
    const auto col_ref_left_0  = cudf::ast::column_reference(0);
    const auto col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
    auto left_zero_eq_right_zero =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);

    for (auto _ : state) {
      cuda_event_timer raii(state, true, rmm::cuda_stream_default);

      auto result =
        JoinFunc(probe_table, build_table, left_zero_eq_right_zero, cudf::null_equality::UNEQUAL);
    }
  }
}
