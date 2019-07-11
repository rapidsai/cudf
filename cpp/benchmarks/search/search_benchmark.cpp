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

#include <benchmark/benchmark.h>

void BM_upper_bound_column(benchmark::State& state){
  const gdf_size_type column_size{(gdf_size_type)state.range(0)};
  const gdf_size_type values_size{(gdf_size_type)state.range(0)};

  cudf::test::column_wrapper<float> column(column_size,
    [=](gdf_index_type row) { return static_cast<float>(row); }
  );
  cudf::test::column_wrapper<float> values(values_size,
    [=](gdf_index_type row) { return static_cast<float>(values_size - row); }
  );
  
  for(auto _ : state){
    auto col = cudf::upper_bound(*(column.get()), *(values.get()));
    RMM_FREE(col.data, 0);
  }
}

BENCHMARK(BM_upper_bound_column)->Range(1<<10, 1<<28);

