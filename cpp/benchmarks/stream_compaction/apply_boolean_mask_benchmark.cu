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

#include <cudf/stream_compaction.hpp>
#include <cudf/table.hpp>
#include <tests/utilities/column_wrapper.cuh>

#include <benchmark/benchmark.h>
#include <random>

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

template<class TypeParam, int num_columns, int column_size>
void BM_apply_boolean_mask(benchmark::State& state) {
  using wrapper = cudf::test::column_wrapper<TypeParam>;
  using mask_wrapper = cudf::test::column_wrapper<cudf::bool8>;

  const gdf_size_type percent_true{static_cast<gdf_size_type>(state.range(0))}; 

  std::vector<cudf::test::column_wrapper<TypeParam> > columns;

  for (int i = 0; i < num_columns; i++) {
    columns.emplace_back(column_size,
      [](gdf_index_type row) { return TypeParam(row); },
      [](gdf_index_type row) { return true; });
  }

  mask_wrapper mask { column_size,
    [&](gdf_index_type row) { 
      return cudf::bool8{random_int(0, 100) < percent_true}; 
    },
    [](gdf_index_type row)  { return true; }
  };

  std::vector<gdf_column*> raw_columns(num_columns, nullptr);
  std::transform(columns.begin(), columns.end(), raw_columns.begin(),
                 [](wrapper &c) { return c.get(); });  

  cudf::table source_table{raw_columns.data(), num_columns};
    
  for(auto _ : state){
    cudf::table result = cudf::apply_boolean_mask(source_table, *mask.get());
    //TODO: cudf::table::destroy(result);
  }
}

static void percent_range(benchmark::internal::Benchmark* b) {
//  for (int num_columns = 1; num_columns <= 4; ++num_columns)
//    for (int column_size = 1e5; column_size <= 1e8; column_size *= 10)
      for (int percent = 0; percent <= 100; percent += 10)
        b->Args({percent});
}

BENCHMARK_TEMPLATE(BM_apply_boolean_mask, int8_t,  1, 1000000)->Apply(percent_range);
BENCHMARK_TEMPLATE(BM_apply_boolean_mask, int32_t, 1, 1000000)->Apply(percent_range);
BENCHMARK_TEMPLATE(BM_apply_boolean_mask, float,   1, 1000000)->Apply(percent_range);
BENCHMARK_TEMPLATE(BM_apply_boolean_mask, double,  1, 1000000)->Apply(percent_range);
