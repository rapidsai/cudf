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

#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
 
#include <cudf/merge.hpp>
 
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <random>
 
// to enable, run cmake with -DBUILD_BENCHMARKS=ON
 
class Merge: public cudf::benchmark {};
 
using IntColWrap = cudf::test::fixed_width_column_wrapper<int32_t>;

void BM_merge(benchmark::State& state) {   
   const cudf::size_type avg_rows = state.range(0);
   const int num_tables = state.range(1);
    
   auto sequence0 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      return row; });
        
   // Content is irrelevant for the benchmark
   auto sequence1 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      return 0; });
  
   std::mt19937 rand_gen(0);
   std::normal_distribution<> dist(avg_rows, avg_rows/2);

   std::vector<std::pair<IntColWrap, IntColWrap>> facts{};
   size_t total_rows = 0;
   std::vector<cudf::table_view> tables{};
   for (int i = 0; i < num_tables; ++i){
      const cudf::size_type rows = std::round(dist(rand_gen));
      const auto clamped_rows = std::max(std::min(rows , avg_rows*2), 0);

      facts.emplace_back(std::pair<IntColWrap, IntColWrap>{
         IntColWrap{sequence0, sequence0 + clamped_rows}, 
         IntColWrap{sequence1, sequence1 + clamped_rows}
      });
      tables.push_back(cudf::table_view{{facts.back().first, facts.back().second}});
      total_rows += clamped_rows;
   }
   std::vector<cudf::size_type> key_cols{0};
   std::vector<cudf::order> column_order {cudf::order::ASCENDING};
   std::vector<cudf::null_order> null_precedence{};
  
   for(auto _ : state){
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      auto result = cudf::experimental::merge(tables,
                                              key_cols,
                                              column_order,
                                              null_precedence);    
    }   
 
   state.SetBytesProcessed(state.iterations()*2*sizeof(int32_t)*total_rows);
}
 
 
 #define MBM_BENCHMARK_DEFINE(name)                 \
 BENCHMARK_DEFINE_F(Merge, name)(::benchmark::State& state) {                       \
    BM_merge(state);                                                               \
 }                                                                                            \
 BENCHMARK_REGISTER_F(Merge, name)->Unit(benchmark::kNanosecond)->UseManualTime() \
                                      ->RangeMultiplier(2)->Ranges({{1<<19, 1<<19},{2, 128}});
                                                                       
 MBM_BENCHMARK_DEFINE(pow2tables);