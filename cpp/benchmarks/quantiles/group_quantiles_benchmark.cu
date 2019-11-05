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

#include <tests/utilities/legacy/column_wrapper.cuh>

#include <cudf/quantiles.hpp>
#include <random>

#include <benchmark/benchmark.h>
#include <random>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Quantiles : public ::benchmark::Fixture {};

// TODO: put it in a struct so `uniform` can be remade with different min, max
template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

void BM_table(benchmark::State& state){
  using wrapper = cudf::test::column_wrapper<int64_t>;

  const cudf::size_type num_columns{(cudf::size_type)state.range(0)};
  const cudf::size_type column_size{(cudf::size_type)state.range(1)};

  std::vector<wrapper> key_columns;
  std::vector<wrapper> val_columns;

  auto make_table = [&] (std::vector<wrapper>& cols,
                         cudf::size_type col_size) -> cudf::table
  {
    for (cudf::size_type i = 0; i < num_columns; i++) {
      cols.emplace_back(col_size,
        [=](cudf::size_type row) { return random_int(0, 10); }
        ,[=](cudf::size_type row) { return random_int(0, 10) < 90; }
      );
    }

    std::vector<gdf_column*> raw_cols(num_columns, nullptr);
    std::transform(cols.begin(), cols.end(), raw_cols.begin(),
                   [](wrapper &c) { return c.get(); });

    return cudf::table{raw_cols.data(), num_columns};
  };

  auto key_table = make_table(key_columns, column_size);
  auto val_table = make_table(val_columns, column_size);

  for(auto _ : state){
    cuda_event_timer timer(state, true);

    cudf::table out_keys, out_vals;
    std::tie(out_keys, out_vals) = 
      cudf::group_quantiles(key_table, val_table, {0.5});
    
    out_keys.destroy();
    out_vals.destroy();
  }
}

BENCHMARK_DEFINE_F(Quantiles, Table)(::benchmark::State& state) {
  BM_table(state);
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int num_cols = 1; num_cols <= 3; num_cols++)
    for (int col_size = 1000; col_size <= 100000000; col_size *= 10)
      b->Args({num_cols, col_size});
}

BENCHMARK_REGISTER_F(Quantiles, Table)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);
