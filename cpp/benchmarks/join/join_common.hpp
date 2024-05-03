/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "generate_input_tables.cuh"

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

#include <nvbench/nvbench.cuh>

#include <vector>

using JOIN_KEY_TYPE_RANGE = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using JOIN_NULLABLE_RANGE = nvbench::enum_type_list<false, true>;

auto const JOIN_SIZE_RANGE = std::vector<nvbench::int64_t>{1000, 100'000, 10'000'000};

struct null75_generator {
  thrust::minstd_rand engine;
  thrust::uniform_int_distribution<unsigned> rand_gen;
  null75_generator() : engine(), rand_gen() {}
  __device__ bool operator()(size_t i)
  {
    engine.discard(i);
    // roughly 75% nulls
    return (rand_gen(engine) & 3) == 0;
  }
};

enum class join_t { CONDITIONAL, MIXED, HASH };

template <typename Key,
          bool Nullable,
          join_t join_type = join_t::HASH,
          typename state_type,
          typename Join>
void BM_join(state_type& state, Join JoinFunc)
{
  auto const right_size = [&]() {
    if constexpr (std::is_same_v<state_type, benchmark::State>) {
      return static_cast<cudf::size_type>(state.range(0));
    }
    if constexpr (std::is_same_v<state_type, nvbench::state>) {
      return static_cast<cudf::size_type>(state.get_int64("right_size"));
    }
  }();
  auto const left_size = [&]() {
    if constexpr (std::is_same_v<state_type, benchmark::State>) {
      return static_cast<cudf::size_type>(state.range(1));
    }
    if constexpr (std::is_same_v<state_type, nvbench::state>) {
      return static_cast<cudf::size_type>(state.get_int64("left_size"));
    }
  }();

  if constexpr (std::is_same_v<state_type, nvbench::state>) {
    if (right_size > left_size) {
      state.skip("Skip large right table");
      return;
    }
  }

  double const selectivity = 0.3;
  int const multiplicity   = 1;

  // Generate build and probe tables
  auto right_random_null_mask = [](int size) {
    // roughly 75% nulls
    auto validity =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), null75_generator{});
    return cudf::detail::valid_if(validity,
                                  validity + size,
                                  thrust::identity<bool>{},
                                  cudf::get_default_stream(),
                                  rmm::mr::get_current_device_resource());
  };

  std::unique_ptr<cudf::column> right_key_column0 = [&]() {
    auto [null_mask, null_count] = right_random_null_mask(right_size);
    return Nullable
             ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<Key>()),
                                         right_size,
                                         std::move(null_mask),
                                         null_count)
             : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<Key>()), right_size);
  }();
  std::unique_ptr<cudf::column> left_key_column0 = [&]() {
    auto [null_mask, null_count] = right_random_null_mask(left_size);
    return Nullable
             ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<Key>()),
                                         left_size,
                                         std::move(null_mask),
                                         null_count)
             : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<Key>()), left_size);
  }();

  // build table is right table, probe table is left table
  generate_input_tables<Key, cudf::size_type>(right_key_column0->mutable_view().data<Key>(),
                                              right_size,
                                              left_key_column0->mutable_view().data<Key>(),
                                              left_size,
                                              selectivity,
                                              multiplicity);

  // Copy right_key_column0 and left_key_column0 into new columns.
  // If Nullable, the new columns will be assigned new nullmasks.
  auto const right_key_column1 = [&]() {
    auto col = std::make_unique<cudf::column>(right_key_column0->view());
    if (Nullable) {
      auto [null_mask, null_count] = right_random_null_mask(right_size);
      col->set_null_mask(std::move(null_mask), null_count);
    }
    return col;
  }();
  auto const left_key_column1 = [&]() {
    auto col = std::make_unique<cudf::column>(left_key_column0->view());
    if (Nullable) {
      auto [null_mask, null_count] = right_random_null_mask(left_size);
      col->set_null_mask(std::move(null_mask), null_count);
    }
    return col;
  }();

  auto init                 = cudf::make_fixed_width_scalar<Key>(static_cast<Key>(0));
  auto right_payload_column = cudf::sequence(right_size, *init);
  auto left_payload_column  = cudf::sequence(left_size, *init);

  CUDF_CHECK_CUDA(0);

  cudf::table_view right_table(
    {right_key_column0->view(), right_key_column1->view(), *right_payload_column});
  cudf::table_view left_table(
    {left_key_column0->view(), left_key_column1->view(), *left_payload_column});

  // Setup join parameters and result table
  [[maybe_unused]] std::vector<cudf::size_type> columns_to_join = {0};

  // Benchmark the inner join operation
  if constexpr (std::is_same_v<state_type, benchmark::State> and
                (join_type != join_t::CONDITIONAL)) {
    for (auto _ : state) {
      cuda_event_timer raii(state, true, cudf::get_default_stream());

      auto result = JoinFunc(left_table.select(columns_to_join),
                             right_table.select(columns_to_join),
                             cudf::null_equality::UNEQUAL);
    }
  }
  if constexpr (std::is_same_v<state_type, nvbench::state> and (join_type != join_t::CONDITIONAL)) {
    if constexpr (join_type == join_t::MIXED) {
      auto const col_ref_left_0 = cudf::ast::column_reference(0);
      auto const col_ref_right_0 =
        cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
      auto left_zero_eq_right_zero =
        cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);
      state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        rmm::cuda_stream_view stream_view{launch.get_stream()};
        auto result = JoinFunc(left_table.select(columns_to_join),
                               right_table.select(columns_to_join),
                               left_table.select({1}),
                               right_table.select({1}),
                               left_zero_eq_right_zero,
                               cudf::null_equality::UNEQUAL,
                               stream_view);
      });
    }
    if constexpr (join_type == join_t::HASH) {
      state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        rmm::cuda_stream_view stream_view{launch.get_stream()};
        auto result = JoinFunc(left_table.select(columns_to_join),
                               right_table.select(columns_to_join),
                               cudf::null_equality::UNEQUAL,
                               stream_view);
      });
    }
  }

  // Benchmark conditional join
  if constexpr (std::is_same_v<state_type, benchmark::State> and join_type == join_t::CONDITIONAL) {
    // Common column references.
    auto const col_ref_left_0  = cudf::ast::column_reference(0);
    auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
    auto left_zero_eq_right_zero =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);

    for (auto _ : state) {
      cuda_event_timer raii(state, true, cudf::get_default_stream());

      auto result =
        JoinFunc(left_table, right_table, left_zero_eq_right_zero, cudf::null_equality::UNEQUAL);
    }
  }
}
