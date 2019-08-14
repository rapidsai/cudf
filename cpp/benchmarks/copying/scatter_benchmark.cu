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
#include <cudf/legacy/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <random>
#include <algorithm>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Scatter: public cudf::benchmark {
};

template<class TypeParam, bool coalesce>
void BM_scatter(benchmark::State& state){
  const gdf_size_type source_size{(gdf_size_type)state.range(0)};
  const gdf_size_type destination_size{(gdf_size_type)state.range(0)};

  const gdf_size_type n_cols = (gdf_size_type)state.range(1);

  std::vector<cudf::test::column_wrapper<TypeParam>> v_src(
    n_cols,
    { source_size, 
      [](gdf_index_type row){ return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return true; }
    }
  );
  std::vector<gdf_column*> vp_src(n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_src[i] = v_src[i].get();  
  }
  
  // Create scatter_map that reverses order of source_column
  std::vector<gdf_index_type> host_scatter_map(source_size);
  std::iota(host_scatter_map.begin(), host_scatter_map.end(), 0);
  if(coalesce){
    std::reverse(host_scatter_map.begin(), host_scatter_map.end());
  }else{
    std::random_shuffle(host_scatter_map.begin(), host_scatter_map.end());
  }
  thrust::device_vector<gdf_index_type> scatter_map(host_scatter_map);

  std::vector<cudf::test::column_wrapper<TypeParam>> v_dest(
    n_cols,
    { source_size, 
      [](gdf_index_type row){return static_cast<TypeParam>(row);},
      [](gdf_index_type row) { return true; }
    }
  );
  std::vector<gdf_column*> vp_dest (n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_dest[i] = v_dest[i].get();  
  }
 
  cudf::table source_table{ vp_src };
  cudf::table target_table{ vp_dest };

  for(auto _ : state){
    cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
    cudf::table destination_table = 
      cudf::scatter(source_table, scatter_map.data().get(), target_table);
  }
  
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range(0)*n_cols*2*sizeof(TypeParam));
}

using namespace cudf;

#define SBM_BENCHMARK_DEFINE(name, type, coalesce)                      \
BENCHMARK_DEFINE_F(benchmark, name)(::benchmark::State& state) {        \
  BM_scatter<type, coalesce>(state);                                    \
}                                                                       \
BENCHMARK_REGISTER_F(benchmark, name)->RangeMultiplier(2)->Ranges({{1<<10,1<<26},{1,8}})->UseManualTime();

SBM_BENCHMARK_DEFINE(double_coalesce_x,double, true);
SBM_BENCHMARK_DEFINE(double_coalesce_o,double,false);
