/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/io/functions.hpp>

// to enable, run cmake with -DBUILD_BENCHMARKS=ON

namespace cudf_io = cudf::experimental::io;

class ParquetWrite: public cudf::benchmark {};

template<typename T>
std::unique_ptr<cudf::experimental::table> create_random_fixed_table(cudf::size_type num_columns, cudf::size_type num_rows, bool include_validity)
{       
   auto valids = cudf::test::make_counting_transform_iterator(0, 
      [](auto i) { 
        return i % 2 == 0 ? true : false; 
      }
    );
   std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
   for(int idx=0; idx<num_columns; idx++){
      auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](T i){return rand();});
      if(include_validity){
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
      } else {
         src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(rand_elements, rand_elements + num_rows);
      }
   }      
   std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
   std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](cudf::test::fixed_width_column_wrapper<T> &in){   
      auto ret = in.release();
      ret->has_nulls();
      return ret;
   });
   return std::make_unique<cudf::experimental::table>(std::move(columns));   
}

void PQ_write(benchmark::State& state)
{
   int64_t total_desired_bytes = state.range(0);
   cudf::size_type num_cols = state.range(1);

   cudf::size_type el_size = 4;     // ints and floats
   int64_t num_rows = total_desired_bytes / (num_cols * el_size);

   auto tbl = create_random_fixed_table<int>(num_cols, num_rows, true);
   cudf::table_view view = tbl->view();

   // this isn't really -super- safe, since validity will add some size,
   // we're using random data and it's not impossible for the compression
   // to cause the data to get larger in general.
   // std::vector<char> out_buffer;
   // out_buffer.resize(total_desired_bytes);

   for(auto _ : state){
      cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
      
      // out_buffer.clear();
      cudf_io::write_parquet_args args{cudf_io::sink_info("parquet_bm.parquet"), view};
      cudf_io::write_parquet(args);      
   }

   state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations())*state.range(0));
}


#define PWBM_BENCHMARK_DEFINE(name, size, num_columns)                                        \
BENCHMARK_DEFINE_F(ParquetWrite, name)(::benchmark::State& state) {                           \
   PQ_write(state);                                                                           \
}                                                                                             \
BENCHMARK_REGISTER_F(ParquetWrite, name)->Args({size, num_columns})                           \
                                           ->Unit(benchmark::kMillisecond)->UseManualTime()

PWBM_BENCHMARK_DEFINE(2Gb8Cols, (int64_t)512 * 1024 * 1024, 8);