/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <fixture/benchmark_fixture.hpp>
#include <synchronization/synchronization.hpp>

#include <vector>

#include "generate_input_tables.cuh"

template <typename key_type, typename payload_type>
class Join : public cudf::benchmark {
};

template <typename key_type, typename payload_type, bool Nullable>
static void BM_join(benchmark::State &state)
{
  const cudf::size_type build_table_size{(cudf::size_type)state.range(0)};
  const cudf::size_type probe_table_size{(cudf::size_type)state.range(1)};
  const cudf::size_type rand_max_val{build_table_size * 2};
  const double selectivity             = 0.3;
  const bool is_build_table_key_unique = true;

  // Generate build and probe tables
  cudf::test::UniformRandomGenerator<cudf::size_type> rand_gen(0, build_table_size);
  auto build_random_null_mask = [&rand_gen](int size) {
    if (Nullable) {
      // roughly 25% nulls
      auto validity = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [&rand_gen](auto i) { return (rand_gen.generate() & 3) == 0; });
      return cudf::test::detail::make_null_mask(validity, validity + size);
    } else {
      return cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    }
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
    rand_max_val,
    is_build_table_key_unique);

  auto payload_data_it = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<payload_type> build_payload_column(
    payload_data_it, payload_data_it + build_table_size);

  cudf::test::fixed_width_column_wrapper<payload_type> probe_payload_column(
    payload_data_it, payload_data_it + probe_table_size);

  CHECK_CUDA(0);

  cudf::table_view build_table({build_key_column->view(), build_payload_column});
  cudf::table_view probe_table({probe_key_column->view(), probe_payload_column});

  // Setup join parameters and result table

  std::vector<cudf::size_type> columns_to_join = {0};

  // Benchmark the inner join operation

  for (auto _ : state) {
    cuda_event_timer raii(state, true, 0);

    auto result = cudf::inner_join(probe_table,
                                   build_table,
                                   columns_to_join,
                                   columns_to_join,
                                   {{0, 0}},
                                   cudf::null_equality::UNEQUAL);
  }
}

#define JOIN_BENCHMARK_DEFINE(name, key_type, payload_type, nullable) \
  BENCHMARK_TEMPLATE_DEFINE_F(Join, name, key_type, payload_type)     \
  (::benchmark::State & st) { BM_join<key_type, payload_type, nullable>(st); }

JOIN_BENCHMARK_DEFINE(join_32bit, int32_t, int32_t, false);
JOIN_BENCHMARK_DEFINE(join_64bit, int64_t, int64_t, false);
JOIN_BENCHMARK_DEFINE(join_32bit_nulls, int32_t, int32_t, true);
JOIN_BENCHMARK_DEFINE(join_64bit_nulls, int64_t, int64_t, true);

BENCHMARK_REGISTER_F(Join, join_32bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, join_64bit)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, join_32bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({100'000, 100'000})
  ->Args({100'000, 400'000})
  ->Args({100'000, 1'000'000})
  ->Args({10'000'000, 10'000'000})
  ->Args({10'000'000, 40'000'000})
  ->Args({10'000'000, 100'000'000})
  ->Args({100'000'000, 100'000'000})
  ->Args({80'000'000, 240'000'000})
  ->UseManualTime();

BENCHMARK_REGISTER_F(Join, join_64bit_nulls)
  ->Unit(benchmark::kMillisecond)
  ->Args({50'000'000, 50'000'000})
  ->Args({40'000'000, 120'000'000})
  ->UseManualTime();
