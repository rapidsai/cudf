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
#include <algorithm>

#include "../fixture/benchmark_fixture.hpp"

template<typename T, bool opt_, bool coalesce_>
class Gather: public cudf::benchmark {
public:
  using TypeParam = T;
};

template<class TypeParam, bool opt, bool coalesce>
void BM_gather(benchmark::State& state){
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
  
  // Create gather_map that reverses order of source_column
  std::vector<gdf_index_type> host_gather_map(source_size);
  std::iota(host_gather_map.begin(), host_gather_map.end(), 0);
  if(coalesce){
    std::reverse(host_gather_map.begin(), host_gather_map.end());
  }else{
    std::random_shuffle(host_gather_map.begin(), host_gather_map.end());
  }
  thrust::device_vector<gdf_index_type> gather_map(host_gather_map);

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
  cudf::table destination_table{ vp_dest };
  
//  if(opt){
//    cudf::opt::gather(&source_table, gather_map.data().get(), &destination_table, 128);
//  }else{
//    cudf::gather(&source_table, gather_map.data().get(), &destination_table);
//  }

  for(auto _ : state){
    if(opt){
      cudf::opt::gather(&source_table, gather_map.data().get(), &destination_table, state.range(2), state.range(3));
    }else{
      cudf::gather(&source_table, gather_map.data().get(), &destination_table);
    }
  }
  
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range(0)*n_cols*2*sizeof(TypeParam));
}

#define GBM_BENCHMARK_DEFINE(name, type, opt, coalesce)                   \
BENCHMARK_TEMPLATE_DEFINE_F(Gather, name, type, opt, coalesce)            \
(::benchmark::State& st) {                                                \
  BM_gather<TypeParam, opt, coalesce>(st);                        \
}

GBM_BENCHMARK_DEFINE(double_opt_x_coa_x,double, true, true);
GBM_BENCHMARK_DEFINE(double_opt_o_coa_x,double,false, true);
GBM_BENCHMARK_DEFINE(double_opt_x_coa_o,double, true,false);
GBM_BENCHMARK_DEFINE(double_opt_o_coa_o,double,false,false);

// BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,128},{640,640}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{320,320}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{400,400}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{480,480}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{560,560}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{640,640}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{720,720}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{800,800}});
BENCHMARK_REGISTER_F(Gather, double_opt_x_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{128,256},{1280,1280}});
// BENCHMARK_REGISTER_F(Gather, double_opt_o_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{4,4},{256,256},{640,640}});
// BENCHMARK_REGISTER_F(Gather, double_opt_o_coa_o)->RangeMultiplier(2)->Ranges({{1<<26,1<<26},{1,4},{256},{}});

