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

#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <random>
#include <algorithm>

#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

class Gather: public cudf::benchmark {
};

template<class TypeParam, bool coalesce>
void BM_gather(benchmark::State& state){
  const cudf::size_type source_size{(cudf::size_type)state.range(0)};
  const cudf::size_type destination_size{(cudf::size_type)state.range(0)};

  const cudf::size_type n_cols = (cudf::size_type)state.range(1);

  std::vector<cudf::test::column_wrapper<TypeParam>> v_src(
    n_cols,
    { source_size, 
      [](cudf::size_type row){ return static_cast<TypeParam>(row); },
      [](cudf::size_type row) { return true; }
    }
  );
  std::vector<gdf_column*> vp_src(n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_src[i] = v_src[i].get();  
  }
  
  // Create gather_map that reverses order of source_column
  std::vector<cudf::size_type> host_gather_map(source_size);
  std::iota(host_gather_map.begin(), host_gather_map.end(), 0);
  if(coalesce){
    std::reverse(host_gather_map.begin(), host_gather_map.end());
  }else{
    std::random_shuffle(host_gather_map.begin(), host_gather_map.end());
  }
  thrust::device_vector<cudf::size_type> gather_map(host_gather_map);

  std::vector<cudf::test::column_wrapper<TypeParam>> v_dest(
    n_cols,
    { source_size, 
      [](cudf::size_type row){return static_cast<TypeParam>(row);},
      [](cudf::size_type row) { return true; }
    }
  );
  std::vector<gdf_column*> vp_dest (n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_dest[i] = v_dest[i].get();  
  }
 
  cudf::table source_table{ vp_src };
  cudf::table destination_table{ vp_dest };

  for(auto _ : state){
    cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
    cudf::gather(&source_table, gather_map.data().get(), &destination_table);
  }
  
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range(0)*n_cols*2*sizeof(TypeParam));
}

#define GBM_BENCHMARK_DEFINE(name, type, coalesce)                      \
BENCHMARK_DEFINE_F(Gather, name)(::benchmark::State& state) {        \
  BM_gather<type, coalesce>(state);                                     \
}                                                                       \
BENCHMARK_REGISTER_F(Gather, name)->RangeMultiplier(2)->Ranges({{1<<10,1<<26},{1,8}})->UseManualTime();

GBM_BENCHMARK_DEFINE(double_coalesce_x,double, true);
GBM_BENCHMARK_DEFINE(double_coalesce_o,double,false);
