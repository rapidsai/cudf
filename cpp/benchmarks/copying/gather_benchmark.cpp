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

#include <cudf/copying.hpp>
#include <cudf/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/types.hpp>
#include <utilities/wrapper_types.hpp>

#include <random>

template<class TypeParam>
void gather_benchmark(benchmark::State& state){
  const gdf_size_type source_size{(gdf_size_type)state.range(0)};
  const gdf_size_type destination_size{(gdf_size_type)state.range(0)};

  cudf::test::column_wrapper<TypeParam> source_column(
      source_size,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return true; });

  // Create gather_map that reverses order of source_column
  std::vector<gdf_index_type> host_gather_map(source_size);
  std::iota(host_gather_map.begin(), host_gather_map.end(), 0);
  std::reverse(host_gather_map.begin(), host_gather_map.end());
  thrust::device_vector<gdf_index_type> gather_map(host_gather_map);

  cudf::test::column_wrapper<TypeParam> destination_column(destination_size,
                                                           true);

  gdf_column* raw_source = source_column.get();
  gdf_column* raw_destination = destination_column.get();

  cudf::table source_table{&raw_source, 1};
  cudf::table destination_table{&raw_destination, 1};
  
  for(auto _ : state){
    cudf::gather(&source_table, gather_map.data().get(), &destination_table);
  }
}

BENCHMARK_TEMPLATE(gather_benchmark, double)->RangeMultiplier(2)->Range(1<<10, 1<<28);
BENCHMARK_TEMPLATE(gather_benchmark, float )->RangeMultiplier(2)->Range(1<<10, 1<<28);
BENCHMARK_TEMPLATE(gather_benchmark, int   )->RangeMultiplier(2)->Range(1<<10, 1<<28);
BENCHMARK_TEMPLATE(gather_benchmark, long  )->RangeMultiplier(2)->Range(1<<10, 1<<28);
