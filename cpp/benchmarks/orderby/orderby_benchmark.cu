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

class OrderBy: public cudf::benchmark {
};

template<class TypeParam, bool reverse, bool hasnulls>
void BM_orderby(benchmark::State& state){
  const gdf_size_type source_size{(gdf_size_type)state.range(0)};
  const gdf_size_type destination_size{(gdf_size_type)state.range(0)};

  const gdf_size_type n_cols = (gdf_size_type)state.range(1);
  
  std::vector<cudf::test::column_wrapper<TypeParam>> v_src(
    n_cols,
    { source_size, 
      [source_size](gdf_index_type row){ 
      if(reverse)
        return static_cast<TypeParam>(source_size-row); 
      else
        return static_cast<TypeParam>(rand()); 
      },
      [](gdf_index_type row) {
      if(hasnulls)
        return bool(rand()%2);
      else
      return true;
      }
    }
  );
  std::vector<gdf_column*> vp_src(n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_src[i] = v_src[i].get();  
  }
  
  // Allocate output ordered indices
  cudf::test::column_wrapper<gdf_index_type> ordered_indices(source_size);
  gdf_context context;

  for(auto _ : state){
    cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
    gdf_order_by(vp_src.data(),
        nullptr,
        n_cols,
        ordered_indices.get(),
        &context);

  }
  
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range(0)*n_cols*2*sizeof(TypeParam));
}

using namespace cudf;

#define OBM_BENCHMARK_DEFINE(name, type, reverse, hasnulls)            \
BENCHMARK_DEFINE_F(benchmark, name)(::benchmark::State& state) {        \
  BM_orderby<type, reverse, hasnulls>(state);                          \
}                                                                       \
BENCHMARK_REGISTER_F(benchmark, name)->Args({1<<10, 8})->Args({10<<10, 8})->Args({100<<10, 8})->Args({1<<20, 8})->Args({10<<20, 8})->Args({100<<20, 8})->UseManualTime()->Unit(benchmark::kMillisecond);
//BENCHMARK_REGISTER_F(benchmark, name)->RangeMultiplier(2)->Ranges({{1<<10,1<<26},{1,8}})->UseManualTime();

OBM_BENCHMARK_DEFINE(double_reverse_1_null_0,double, true, false);
OBM_BENCHMARK_DEFINE(double_reverse_1_null_1,double, true, true);
OBM_BENCHMARK_DEFINE(double_reverse_0_null_0,double,false, false);
OBM_BENCHMARK_DEFINE(double_reverse_0_null_1,double,false, true);
