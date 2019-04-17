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

#include <chrono>
#include <random>
#include <cuda_profiler_api.h>
#include <stream_compaction.hpp>
#include <utilities/error_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

template <typename T>
T random_int(T min, T max)
{
  static std::mt19937 engine{std::random_device{}};
  static std::uniform_int_distribution<T> uniform;

  return uniform(engine,
                 typename std::uniform_int_distribution<T>::param_type{min, max});
}

// default implementation
template <typename T>
struct TypeName
{
    static const char* Get()
    {
        return typeid(T).name();
    }
};

gdf_dtype type_from_name(const std::string &name)
{
  if      (name == "a") return GDF_INT8;
  else if (name == "s") return GDF_INT16;
  else if (name == "i") return GDF_INT32;
  else if (name == "l") return GDF_INT64;
  else if (name == "f") return GDF_FLOAT32;
  else if (name == "d") return GDF_FLOAT64;
  else return N_GDF_TYPES;
}

struct benchmark
{
  template <typename T, typename Init, typename Bench>
  void operator()(Init init, Bench bench, int iters = 100)
  {
    auto columns = init(T{0});
    cudaProfilerStart();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      bench(columns.first, columns.second);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    cudaProfilerStop();
    std::cout << TypeName<T>::Get() << ": " << diff.count() / iters << " s\n";
  }
};

template <typename T, typename Bench>
void benchmark_fraction_shmoo(Bench bench, gdf_size_type column_size,
                              int pct_step = 5, int iters = 100)
{
  gdf_size_type fraction = 0;
  auto init_fraction_true = [column_size](auto a, int fraction) {
    using TypeParam = decltype(a);
    cudf::test::column_wrapper<TypeParam> source{
      column_size,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return true; }};
    cudf::test::column_wrapper<gdf_bool> mask{
      column_size,
      [&](gdf_index_type row) { return gdf_bool{random_int(0, 100) < fraction}; },
      [](gdf_index_type row) { return true; }};
    
    return std::make_tuple(source, mask);
  };

  for (fraction = 0; fraction <= 100; fraction += pct_step) {
    std::cout << fraction << "% output ";
    benchmark().template operator()<T>(init_fraction_true, bench, iters);
  }
}


template <typename Init, typename Bench>
void benchmark_types(gdf_dtype type, Init init, Bench bench, int iters = 100)
{
  if (type == N_GDF_TYPES) { // run all
    std::vector<gdf_dtype> types{GDF_INT8, GDF_INT16, GDF_INT32, GDF_INT64, 
                                 GDF_FLOAT32, GDF_FLOAT64};
    for (gdf_dtype t : types)
      cudf::type_dispatcher(t, benchmark(), init, bench, iters);
  }
  else
    cudf::type_dispatcher(type, benchmark(), init, bench, iters);
}

int main(int argc, char **argv)
{
  gdf_size_type column_size{42000000};
  int iters{100};
  int index = -1; // all benches
  gdf_dtype type = N_GDF_TYPES;

  if (argc > 1) column_size = std::stoi(argv[1]);
  if (argc > 2) iters = std::stoi(argv[2]);
  if (argc > 3) index = std::stoi(argv[3]);
  if (argc > 4) type = type_from_name(argv[4]);

  rmmOptions_t options{PoolAllocation, 0, false};
  rmmInitialize(&options); 

  auto bench = [](gdf_column const* source, gdf_column const* mask) {
    gdf_column result = cudf::apply_boolean_mask(source, mask);
    gdf_column_free(&result);
  };

  if (index == -1 || index == 0) {
    auto init = [column_size](auto a) {
      using TypeParam = decltype(a);
      cudf::test::column_wrapper<TypeParam> source{
        column_size,
        [](gdf_index_type row) { return TypeParam(row); },
        [](gdf_index_type row) { return row % 2 == 0; }};
      cudf::test::column_wrapper<cudf::bool8> mask{
        column_size,
        [](gdf_index_type row) { return cudf::bool8{true}; },
        [](gdf_index_type row) { return row % 2 == 1; }};
      
      return std::make_pair(source, mask);
    };
    
    std::cout << "With null masks: Avg time to apply_boolean_mask for "
              << column_size << " elements:\n";
    benchmark_types(type, init, bench, iters);
  }

  if (index == -1 || index == 1) {
    auto init_no_null = [column_size](auto a) {
      using TypeParam = decltype(a);
      cudf::test::column_wrapper<TypeParam> source{column_size, false};
      cudf::test::column_wrapper<cudf::bool8> mask{
        column_size,
        [](gdf_index_type row) { return cudf::bool8{true}; },
        [](gdf_index_type row) { return row % 2 == 1; }};
      cudf::test::column_wrapper<TypeParam> output{column_size, false};

      return std::make_pair(source, mask);  
    };

    std::cout << "Without null masks: Avg time to apply_boolean_mask for "
              << column_size << " elements:\n";
    benchmark_types(type, init_no_null, bench, iters);
  }

  if (index == -1 || index == 2) {
    auto init_all_false_mask = [column_size](auto a) {
      using TypeParam = decltype(a);
      cudf::test::column_wrapper<TypeParam> source{
        column_size,
        [](gdf_index_type row) { return TypeParam(row); },
        [](gdf_index_type row) { return row % 2 == 0; }};
      cudf::test::column_wrapper<cudf::bool8> mask{
        column_size,
        [](gdf_index_type row) { return cudf::bool8{false}; },
        [](gdf_index_type row) { return row % 2 == 1; }};

      return std::make_pair(source, mask);  
    };

    std::cout << "All false mask: Avg time to apply_boolean_mask for "
              << column_size << " elements:\n";
    benchmark_types(type, init_all_false_mask, bench, iters);
  }

  rmmFinalize();

  return 0;
}
