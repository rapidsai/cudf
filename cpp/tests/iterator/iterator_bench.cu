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

/** --------------------------------------------------------------------------*
 * This is benchmarks for iterator and raw pointer for cub and thrust
 * This code won't be compiled by default.
 * To compile this code, add `CMAKE_ENABLE_BENCHMARKS=ON` at cmake option like this:
 *   cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON -DCMAKE_ENABLE_BENCHMARKS=ON
 * The benchmark app binary will be placed under `bench`,
 * and the app name is `ITERATOR_BENCH`.
 *
 * the benchmark command:
 * ITERATOR_BENCH [column size] [num iteration] [dtypes (a|s|i|l|f|d|o)] [do full tests]
 *
 * examples:
 * 1. default parameter
 *   ./ITERATOR_BENCH
 * 2. heavy bench parameter
 *   ./ITERATOR_BENCH 42000000 1000 others
 * 3. full bench parameter
 *   ./ITERATOR_BENCH 3000000 1000 others full
 * -------------------------------------------------------------------------**/


#include <chrono>
#include <random>
#include <tuple>

#include <cuda_profiler_api.h>
#include <utilities/error_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

#include <iterator/iterator.cuh>    // include iterator header
#include <utilities/device_operators.cuh>
#include <reductions/reduction_operators.cuh>

// for reduction tests
#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

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

// -----------------------------------------------------------------------------

class BenchMarkTimer
{
public:
  BenchMarkTimer(int iters_) : iters(iters_)
  {
    start();
  };
  ~BenchMarkTimer(){end();};

protected:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
  int iters;

  void start()
  {
    cudaProfilerStart();
    start_point = std::chrono::high_resolution_clock::now();
  };

  void end()
  {
    cudaDeviceSynchronize();
    auto end_point = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_point-start_point;
    cudaProfilerStop();
    std::cout << diff.count() / iters  << std::endl << std::flush;
  };

};

// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T>
inline void reduce_by_cub_storage(
  void *d_temp_storage, size_t& temp_storage_bytes,
  InputIterator d_in, OutputIterator result, int num_items, T init)
{
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, result, num_items,
      cudf::DeviceSum{}, init);
}

template <typename InputIterator, typename OutputIterator, typename T>
inline auto reduce_by_cub(OutputIterator result, InputIterator d_in, int num_items, T init)
{
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, d_in, result, num_items, init);

  // Allocate temporary storage
  RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

  // Run reduction
  reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, d_in, result, num_items, init);

  // Free temporary storage
  RMM_TRY(RMM_FREE(d_temp_storage, 0));

  return temp_storage_bytes;
}
// ------------------------

template <typename T>
void raw_stream_bench_cub(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters, bool no_new_allocate=false)
{
  std::cout << "raw stream cub: " << "\t\t";

  T init{0};
  auto begin = static_cast<T*>(col.get()->data);
  int num_items = col.size();

  if( no_new_allocate ){
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    auto bench = [&](){ reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, begin, result.begin(), num_items, init);};

    bench();
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));
    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, 0));
  }else{
    auto bench = [&](){ reduce_by_cub(result.begin(), begin, num_items, init);};

    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);
  }
};

template <typename T, bool has_null>
void iterator_bench_cub(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters, bool no_new_allocate=false)
{

  std::cout << "iterator cub " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto begin = cudf::make_iterator<has_null, T>(col, init);
  int num_items = col.size();

  if( no_new_allocate ){
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    auto bench = [&](){ reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, begin, result.begin(), num_items, init);};

    bench();
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));
    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, 0));
  }else{
    auto bench = [&](){ reduce_by_cub(result.begin(), begin, num_items, init);};

    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);
  }
};


template <typename T>
void raw_stream_bench_thrust(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters)
{
  std::cout << "raw stream thust: " << "\t\t";

  T init{0};
  auto d_in = static_cast<T*>(col.get()->data);
  auto d_end = d_in + col.size();

  auto bench = [&](){ thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{}); };

  bench(); // warm up

  do{
    BenchMarkTimer timer(iters);
    for (int i = 0; i < iters; ++i) {
      bench();
    }
  }while(0);
}


template <typename T, bool has_null>
void iterator_bench_thrust(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters)
{
  std::cout << "iterator thust " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto d_in = cudf::make_iterator<has_null, T>(col, init);
  auto d_end = d_in + col.size();

  auto bench = [&](){ thrust::reduce(thrust::device, d_in, d_end, init, cudf::DeviceSum{}); };

  bench(); // warm up

  do{
    BenchMarkTimer timer(iters);
    for (int i = 0; i < iters; ++i) {
      bench();
    }
  }while(0);
}


// --------------------------------------------------------------------------
static constexpr int reduction_block_size = 128;

/*
Generic reduction implementation with support for validity mask
*/
template<typename T_in, typename T_out, typename F, typename Ld>
__global__
void gpu_reduction_op(const T_in *data, const gdf_valid_type *mask,
                      gdf_size_type size, T_out *result,
                      F functor, T_out identity, Ld loader)
{
    typedef cub::BlockReduce<T_out, reduction_block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int step = blksz * gridsz;

    T_out agg = identity;
    for (int base=blkid * blksz; base<size; base+=step) {
        // Threadblock synchronous loop
        int i = base + tid;
        // load
        T_out loaded = identity;
        if (i < size && gdf_is_valid(mask, i))
            loaded = static_cast<T_out>(loader(data, i));

        // Block reduce
        T_out temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }

    // First thread of each block stores the result.
    if (tid == 0){
        cudf::genericAtomicOperation(result, agg, functor);
    }
}

template<typename T_in, typename T_out, typename Op>
void ReduceOp(const gdf_column *input, rmm::device_vector<T_out>& dev_result, int iters)
{
    T_out identity = Op::Op::template identity<T_out>();

    // allocate temporary memory for the result
    T_out* result = dev_result.data().get();

    int blocksize = reduction_block_size;
    int gridsize = (input->size + reduction_block_size -1 )
        /reduction_block_size;

    auto bench = [&](){
      // kernel call
      gpu_reduction_op<<<gridsize, blocksize>>>(
          static_cast<const T_in*>(input->data), input->valid, input->size,
          result,
          typename Op::Op{}, identity, typename Op::Loader{});
    };

    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);
};

// --------------------------------------------------------------------------
template <typename T, bool has_null>
void raw_stream_bench_cub_block(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters)
{
  std::cout << "raw stream cub::BlockReduce" << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";
  ReduceOp<T, T, cudf::reductions::ReductionSum>(col, result, iters);
}

// -----------------------------------------------------------------------------

struct benchmark
{
  template <typename T>
  void operator()(gdf_size_type column_size, int iters, bool do_full_test)
  {
    cudf::test::column_wrapper<T> hasnull_F(
      column_size,
      [](gdf_index_type row) { return T(row); });

    cudf::test::column_wrapper<T> hasnull_T(
      column_size,
      [](gdf_index_type row) { return T(row); },
      [](gdf_index_type row) { return row % 2 == 0; });

    rmm::device_vector<T> dev_result(1, T{0});

    // if no_new_allocate = false, *_bench_cub will allocate temporary buffer every calls
    bool no_new_allocate = false;

    do{
      std::cout << "new allocation: " << no_new_allocate << std::endl;

      raw_stream_bench_cub<T>(hasnull_F, dev_result, iters, no_new_allocate);      // driven by raw pointer
      iterator_bench_cub<T, false>(hasnull_F, dev_result, iters, no_new_allocate); // driven by riterator without nulls
      iterator_bench_cub<T, true >(hasnull_T, dev_result, iters, no_new_allocate); // driven by riterator with nulls

      no_new_allocate = !no_new_allocate;
    }while (no_new_allocate);

    raw_stream_bench_thrust<T>(hasnull_F, dev_result, iters);      // driven by raw pointer
    iterator_bench_thrust<T, false>(hasnull_F, dev_result, iters); // driven by riterator without nulls
    iterator_bench_thrust<T, true >(hasnull_T, dev_result, iters); // driven by riterator with nulls

    if( do_full_test ){
      // these uses same logic cudf::reduction used at branch-0.7.
      // thise uses `cub::BlockReduce` + `atomicAdd`
      // do_full_test = false by default, since the exec time of `raw_stream_bench_cub_block` for
      // GDF_INT8, GDF_INT16 is extremely slow, 1500x slower than others.

      raw_stream_bench_cub_block<T, false>(hasnull_F, dev_result, iters);
      raw_stream_bench_cub_block<T, true >(hasnull_T, dev_result, iters);

    }
  };
};

void benchmark_types(gdf_size_type column_size, int iters, gdf_dtype type=N_GDF_TYPES, bool do_full_test=false)
{
  std::vector<gdf_dtype> types{};
  if (type == N_GDF_TYPES)
    types = {GDF_INT8, GDF_INT16, GDF_INT32, GDF_INT64, GDF_FLOAT32, GDF_FLOAT64};
  else
    types = {type};

  std::cout <<  "Iterator performance test:" << std::endl;
  std::cout <<  "  column_size = " << column_size << std::endl;
  std::cout <<  "  num iterates = " << iters << std::endl << std::endl;

  for (gdf_dtype t : types) {
    std::cout << name_from_type(t) << std::endl;
    cudf::type_dispatcher(t, benchmark(), column_size, iters, do_full_test);
    std::cout << std::endl << std::endl;
  }
}

int main(int argc, char **argv)
{
  gdf_size_type column_size{10000000};
  int iters{1000};
  gdf_dtype type = N_GDF_TYPES;
  bool do_full_test = false;

  if (argc > 1) column_size = std::stoi(argv[1]);
  if (argc > 2) iters = std::stoi(argv[2]);
  if (argc > 3) type = type_from_name(argv[3]);
  if (argc > 4) do_full_test = (argv[4][0] == 'f')? true : false;

  rmmOptions_t options{PoolAllocation, 0, false};
  rmmInitialize(&options);

  // -----------------------------------
  // type = GDF_FLOAT64;
  benchmark_types(column_size, iters, type, do_full_test);
  // -----------------------------------

  rmmFinalize();
  return 0;
}
