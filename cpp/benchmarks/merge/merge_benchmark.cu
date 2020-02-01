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
 
 // to enable, run cmake with -DBUILD_BENCHMARKS=ON
 
 class Merge: public cudf::benchmark {};
 
 using IntColWrap = cudf::test::fixed_width_column_wrapper<int32_t>;

 void BM_merge(benchmark::State& state)
 {   
    const cudf::size_type inputRows = state.range(0);
    const int num_tables = state.range(1);
    
    auto sequence0 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        return row; });
        
    auto sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        return inputRows - row; });
  
    std::vector<std::pair<IntColWrap, IntColWrap>> facts{};
    std::vector<cudf::table_view> tables{};
    for (int i = 0; i < num_tables; ++i){
      facts.emplace_back(std::pair<IntColWrap, IntColWrap>{IntColWrap{sequence0, sequence0 + inputRows}, IntColWrap{sequence1, sequence1 + inputRows}});
      tables.push_back(cudf::table_view{{facts.back().first, facts.back().second}});
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
 
    state.SetBytesProcessed(state.iterations()*2*sizeof(int32_t)*inputRows*num_tables);
 }
 
 
 #define MBM_BENCHMARK_DEFINE(name)                 \
 BENCHMARK_DEFINE_F(Merge, name)(::benchmark::State& state) {                       \
    BM_merge(state);                                                               \
 }                                                                                            \
 BENCHMARK_REGISTER_F(Merge, name)->Unit(benchmark::kNanosecond)->UseManualTime() \
                                      ->RangeMultiplier(2)->Ranges({{1<<19, 1<<19},{2, 1<<7}});
                                                                       
 MBM_BENCHMARK_DEFINE(pow2tables);