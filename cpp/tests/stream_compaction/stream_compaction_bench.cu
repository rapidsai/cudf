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
#include <cudf/stream_compaction.hpp>
#include <utilities/error_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
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

const char* name_from_type(gdf_dtype type) 
{
  switch (type) {
    case GDF_INT8:    return "GDF_INT8";
    case GDF_INT16:   return "GDF_INT16";
    case GDF_INT32:   return "GDF_INT32";
    case GDF_INT64:   return "GDF_INT64";
    case GDF_FLOAT32: return "GDF_FLOAT32";
    case GDF_FLOAT64: return "GDF_FLOAT64";
    default:          return "GDF_INVALID";
  }
}
 
struct warmup
{
  template <typename T, typename Init, typename Bench>
  void operator()(Init init, Bench bench, int fraction=100)
  {
    auto columns = init(T{0}, fraction);
    bench(columns.first, columns.second);
  }
};

struct benchmark
{
  template <typename T, typename Init, typename Bench>
  void operator()(Init init, Bench bench, int iters = 100, int fraction=100, 
                  bool shmoo=false)
  {
    auto columns = init(T{0}, fraction);
    bench(columns.first, columns.second); // warm up
    cudaProfilerStart();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      bench(columns.first, columns.second);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    cudaProfilerStop();
    std::cout << diff.count() / iters << (shmoo ? "," : " s") << std::flush;
  }
};

template <typename Init, typename Bench>
void benchmark_types(gdf_dtype type, Init init, Bench bench, 
                     int iters = 100, int pct_step=100, bool shmoo = false)
{
  std::vector<gdf_dtype> types{};
  if (type == N_GDF_TYPES)
    types = {GDF_INT8, GDF_INT16, GDF_INT32, GDF_INT64, GDF_FLOAT32, GDF_FLOAT64};
  else 
    types = {type};

  for (gdf_dtype t : types) {
    cudf::type_dispatcher(t, warmup(), init, bench);
    std::cout << name_from_type(t) << ",";
    if (shmoo) {
      for (int fraction = 0; fraction <= 100; fraction += pct_step)
        cudf::type_dispatcher(t, benchmark(), init, bench, iters, fraction, shmoo);
    }
    else {
      cudf::type_dispatcher(t, benchmark(), init, bench, iters, 50);
    }
    std::cout << "\n";
  }
}

// Shmoo the fraction of true elements that are masked from 0 to 100
template <typename Bench>
void benchmark_fraction_shmoo(gdf_dtype type, Bench bench, 
                              gdf_size_type column_size,
                              int pct_step = 5, int iters = 100)
{
  gdf_size_type fraction = 0;
  auto init_fraction_true = [column_size](auto a, int fraction) {
    using TypeParam = decltype(a);
    cudf::test::column_wrapper<TypeParam> source(
      column_size,
      [](gdf_index_type row) { return TypeParam(row); },
      [](gdf_index_type row) { return true; });
    cudf::test::column_wrapper<cudf::bool8> mask(
      column_size,
      [&](gdf_index_type row) { return cudf::bool8{random_int(0, 100) < fraction}; },
      [](gdf_index_type row)  { return true; });
    
    return std::make_pair(source, mask);
  };

  std::cout << "type,";
  for (fraction = 0; fraction <= 100; fraction += pct_step)
    std::cout << fraction << ",";
  std::cout << "\n";

  benchmark_types(type, init_fraction_true, bench, iters, pct_step, true);
}

int main(int argc, char **argv)
{
  gdf_size_type column_size{42000000};
  int iters{100};
  int index = -1; // all benches
  gdf_dtype type = N_GDF_TYPES;
  bool shmoo = false;

  if (argc > 1) column_size = std::stoi(argv[1]);
  if (argc > 2) iters = std::stoi(argv[2]);
  if (argc > 3) {
    shmoo = (std::string(argv[3]) == "shmoo");
    if (!shmoo)
      index = std::stoi(argv[3]);
  }
  if (argc > 4) type = type_from_name(argv[4]);

  rmmOptions_t options{PoolAllocation, 0, false};
  rmmInitialize(&options); 

  auto bench = [](gdf_column const& source, gdf_column const& mask) {
    gdf_column result = cudf::apply_boolean_mask(source, mask);
    gdf_column_free(&result);
  };

  if (!shmoo) {
    if (index == -1 || index == 0) {
      auto init = [column_size](auto a, int) {
        using TypeParam = decltype(a);
        cudf::test::column_wrapper<TypeParam> source(
          column_size,
          [](gdf_index_type row) { return TypeParam(row); },
          [](gdf_index_type row) { return row % 2 == 0; });
        cudf::test::column_wrapper<cudf::bool8> mask(
          column_size,
          [](gdf_index_type row) { return cudf::bool8{true}; },
          [](gdf_index_type row) { return row % 2 == 1; });
        
        return std::make_pair(source, mask);
      };
      
      std::cout << "With null masks: Avg time to apply_boolean_mask for "
                << column_size << " elements:\n";
      benchmark_types(type, init, bench, iters);
    }

    if (index == -1 || index == 1) {
      auto init_no_null = [column_size](auto a, int) {
        using TypeParam = decltype(a);
        cudf::test::column_wrapper<TypeParam> source(column_size, false);
        cudf::test::column_wrapper<cudf::bool8> mask(
          column_size,
          [](gdf_index_type row) { return cudf::bool8{true}; },
          [](gdf_index_type row) { return row % 2 == 1; });
        cudf::test::column_wrapper<TypeParam> output(column_size, false);

        return std::make_pair(source, mask);  
      };

      std::cout << "Without null masks: Avg time to apply_boolean_mask for "
                << column_size << " elements:\n";
      benchmark_types(type, init_no_null, bench, iters);
    }

    if (index == -1 || index == 2) {
      auto init_all_false_mask = [column_size](auto a, int) {
        using TypeParam = decltype(a);
        cudf::test::column_wrapper<TypeParam> source(
          column_size,
          [](gdf_index_type row) { return TypeParam(row); },
          [](gdf_index_type row) { return row % 2 == 0; });
        cudf::test::column_wrapper<cudf::bool8> mask(
          column_size,
          [](gdf_index_type row) { return cudf::bool8{false}; },
          [](gdf_index_type row) { return row % 2 == 1; });

        return std::make_pair(source, mask);  
      };

      std::cout << "All false mask: Avg time to apply_boolean_mask for "
                << column_size << " elements:\n";
      benchmark_types(type, init_all_false_mask, bench, iters);
    }
  }
  else { // shmoo
    benchmark_fraction_shmoo(type, bench, column_size, 5, 100);
  }

  rmmFinalize();

  return 0;
}
