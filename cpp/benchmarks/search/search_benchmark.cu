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

#include <tests/utilities/column_wrapper.cuh>

#include <cudf/search.hpp>
#include <cudf/copying.hpp>

#include <benchmark/benchmark.h>

#include <random>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Search : public cudf::benchmark {};

void BM_non_null_column(benchmark::State& state){
  const gdf_size_type column_size{(gdf_size_type)state.range(0)};
  const gdf_size_type values_size = column_size;

  cudf::test::column_wrapper<float> column(column_size,
    [=](gdf_index_type row) { return static_cast<float>(row); }
  );
  cudf::test::column_wrapper<float> values(values_size,
    [=](gdf_index_type row) { return static_cast<float>(values_size - row); }
  );
  
  for(auto _ : state){
    cuda_event_timer timer(state, true);
    auto col = cudf::upper_bound({column.get()}, {values.get()}, {false});
    gdf_column_free(&col);
  }
}

BENCHMARK_DEFINE_F(Search, AllValidColumn)(::benchmark::State& state) {
  BM_non_null_column(state);
}

BENCHMARK_REGISTER_F(Search, AllValidColumn)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(100000000);

void BM_nullable_column(benchmark::State& state){
  const gdf_size_type column_size{(gdf_size_type)state.range(0)};
  const gdf_size_type values_size = column_size;

  cudf::test::column_wrapper<float> column(column_size,
    [=](gdf_index_type row) { return static_cast<float>(row); },
    [=](gdf_index_type row) { return row < (9 * column_size / 10); }
  );
  cudf::test::column_wrapper<float> values(values_size,
    [=](gdf_index_type row) { return static_cast<float>(values_size - row); },
    [=](gdf_index_type row) { return row < (9 * column_size / 10); }
  );
  
  for(auto _ : state){
    cuda_event_timer timer(state, true);
    auto col = cudf::upper_bound({column.get()}, {values.get()}, {false});
    gdf_column_free(&col);
  }
}

BENCHMARK_DEFINE_F(Search, NullableColumn)(::benchmark::State& state) {
  BM_nullable_column(state);
}

BENCHMARK_REGISTER_F(Search, NullableColumn)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Arg(100000000);

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

void sort_table(cudf::table& t, std::vector<bool>& desc_flags) {
  rmm::device_vector<int8_t> dv_desc_flags(desc_flags);
  auto d_desc_flags = dv_desc_flags.data().get();

  auto out_indices = cudf::test::column_wrapper<int32_t>(t.num_rows());

  gdf_context ctxt{};
  gdf_order_by(t.begin(), d_desc_flags, t.num_columns(), out_indices.get(), &ctxt);

  auto indices = static_cast<gdf_index_type*>(out_indices.get()->data);
  cudf::gather(&t, indices, &t);
}

void BM_table(benchmark::State& state){
  using wrapper = cudf::test::column_wrapper<float>;

  const gdf_size_type num_columns{(gdf_size_type)state.range(0)};
  const gdf_size_type column_size{(gdf_size_type)state.range(1)};
  const gdf_size_type values_size = column_size;

  std::vector<wrapper> columns;
  std::vector<wrapper> values;

  auto make_table = [&] (std::vector<wrapper>& cols,
                         gdf_size_type col_size) -> cudf::table
  {
    for (gdf_size_type i = 0; i < num_columns; i++) {
      cols.emplace_back(col_size,
        [=](gdf_index_type row) { return random_int(0, 100); },
        [=](gdf_index_type row) { return random_int(0, 100) < 90; }
      );
    }

    std::vector<gdf_column*> raw_cols(num_columns, nullptr);
    std::transform(cols.begin(), cols.end(), raw_cols.begin(),
                   [](wrapper &c) { return c.get(); });

    return cudf::table{raw_cols.data(), num_columns};
  };

  auto data_table = make_table(columns, column_size);
  auto values_table = make_table(values, values_size);

  std::vector<bool> desc_flags(num_columns, false);
  sort_table(data_table, desc_flags);
  
  for(auto _ : state){
    cuda_event_timer timer(state, true);
    auto col = cudf::lower_bound(data_table, values_table, desc_flags);
    gdf_column_free(&col);
  }
}

BENCHMARK_DEFINE_F(Search, Table)(::benchmark::State& state) {
  BM_table(state);
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int num_cols = 1; num_cols <= 10; num_cols *= 2)
    for (int col_size = 1000; col_size <= 100000000; col_size *= 10)
      b->Args({num_cols, col_size});
}

BENCHMARK_REGISTER_F(Search, Table)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);

