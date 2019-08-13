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


#include <cudf/copying.hpp>
#include <cudf/legacy/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <table/legacy/device_table.cuh>

#include <random>
#include <utilities/cuda_utils.hpp>
#include "../synchronization/synchronization.hpp"
#include "../fixture/benchmark_fixture.hpp"

enum DispatchingType {
  HOST_DISPATCHING,
  DEVICE_DISPATCHING,
  NO_DISPATCHING
};

enum FunctorType {
  BANDWIDTH_BOUND,
  COMPUTE_BOUND
};

template<class T, FunctorType ft>
struct Functor{
  static __device__ T f(T x){
    if(ft == BANDWIDTH_BOUND){
      return x + static_cast<T>(1) - static_cast<T>(1);
    }else{
      for(int i = 0; i < 1000; i++){
        x = (x*x + static_cast<T>(1)) - x*x - static_cast<T>(1);
      }
      return x;
    }
  }
};

constexpr int block_size = 256;

// This is for NO_DISPATCHING
template<FunctorType functor_type, class T>
__global__ void no_dispatching_kernel(T** A, gdf_size_type n_rows, gdf_size_type n_cols){
  using F = Functor<T, functor_type>;
  gdf_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
  while(index < n_rows){
    for(int c = 0; c < n_cols; c++){
      A[c][index] = F::f(A[c][index]);
    }
    index += blockDim.x * gridDim.x;
  }
}

// This is for HOST_DISPATCHING
template<FunctorType functor_type, class T>
__global__ void host_dispatching_kernel(T* A, gdf_size_type n_rows){
  using F = Functor<T, functor_type>;
  gdf_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
  while(index < n_rows){
    A[index] = F::f(A[index]);
    index += blockDim.x * gridDim.x;
  }
}

template<FunctorType functor_type>
struct ColumnHandle {
  template <typename ColumnType>
  void operator()(gdf_column* source_column, int work_per_thread, cudaStream_t stream = 0) {
    ColumnType* source_data = static_cast<ColumnType*>(source_column->data);
    gdf_size_type const n_rows = source_column->size;
    int grid_size = cudf::util::cuda::grid_config_1d(n_rows, block_size).num_blocks;
    // Launch the kernel.
    host_dispatching_kernel<functor_type><<<grid_size, block_size, 0, stream>>>(source_data, n_rows);
  }
};


// The following is for DEVICE_DISPATCHING:
// The dispatching is done on device. The loop loops over
// each row (across different coluns). Type is dispatched each time 
// a column is visited so the total number of dispatching is 
// n_rows * n_cols. 
template<FunctorType functor_type>
struct RowHandle {
  template<typename T>
  __device__ void operator()(const gdf_column& source, gdf_index_type index){
    using F = Functor<T, functor_type>;
    static_cast<T*>(source.data)[index] = 
      F::f(static_cast<T*>(source.data)[index]);
  }
};

// This is for DEVICE_DISPATCHING
template<FunctorType functor_type>
__global__ void device_dispatching_kernel(device_table source){

  const gdf_index_type n_rows = source.num_rows();
  gdf_index_type index = threadIdx.x + blockIdx.x * blockDim.x;
  
  while(index < n_rows){
    for(gdf_size_type i = 0; i < source.num_columns(); i++){
      cudf::type_dispatcher(source.get_column(i)->dtype,
                          RowHandle<functor_type>{}, *source.get_column(i), index);
    }
    index += blockDim.x * gridDim.x;
  } // while
}

template<FunctorType functor_type, DispatchingType dispatching_type, class T>
void launch_kernel(cudf::table& input, T** d_ptr, int work_per_thread){
  
  const gdf_size_type n_rows = input.num_rows();
  const gdf_size_type n_cols = input.num_columns();
    
  int grid_size = cudf::util::cuda::grid_config_1d(n_rows, block_size).num_blocks;
 
  if(dispatching_type == HOST_DISPATCHING){
    // std::vector<cudf::util::cuda::scoped_stream> v_stream(n_cols);
    for(int c = 0; c < n_cols; c++){
      cudf::type_dispatcher(input.get_column(c)->dtype, ColumnHandle<functor_type>{}, input.get_column(c), work_per_thread);
    }
  }else if(dispatching_type == DEVICE_DISPATCHING){
    auto d_source_table = device_table::create(input);
    auto f = device_dispatching_kernel<functor_type>;
    // Launch the kernel
    f<<<grid_size, block_size>>>(*d_source_table);
  }else if(dispatching_type == NO_DISPATCHING){
    auto f = no_dispatching_kernel<functor_type, T>;
    // Launch the kernel
    f<<<grid_size, block_size>>>(d_ptr, n_rows, n_cols);
  }
}

template<class TypeParam, FunctorType functor_type, DispatchingType dispatching_type>
void type_dispatcher_benchmark(benchmark::State& state){
  const gdf_size_type source_size = static_cast<gdf_size_type>(state.range(1));
  
  const gdf_size_type n_cols = static_cast<gdf_size_type>(state.range(0));
  
  const gdf_size_type work_per_thread = static_cast<gdf_size_type>(state.range(2));

  
  std::vector<cudf::test::column_wrapper<TypeParam>> v_src(
      n_cols,
      {
        source_size,
        [](gdf_index_type row){ return static_cast<TypeParam>(row); },
        [](gdf_index_type row) { return true; }
      }
  );
  
  std::vector<gdf_column*> vp_src(n_cols);
  for(size_t i = 0; i < v_src.size(); i++){
    vp_src[i] = v_src[i].get();  
  }

  cudf::table source_table{ vp_src };
  
  // For no dispatching
  std::vector<rmm::device_vector<TypeParam>> h_vec(n_cols, rmm::device_vector<TypeParam>(source_size, 0));
  std::vector<TypeParam*> h_vec_p(n_cols);
  for(int c = 0; c < n_cols; c++){
    h_vec_p[c] = h_vec[c].data().get();
  }
  rmm::device_vector<TypeParam*> d_vec(n_cols);
  
  if(dispatching_type == NO_DISPATCHING){
    CUDA_TRY(cudaMemcpy(d_vec.data().get(), h_vec_p.data(), sizeof(TypeParam*)*n_cols, cudaMemcpyHostToDevice));
  }
  
  // Warm up  
  launch_kernel<functor_type, dispatching_type>(source_table, d_vec.data().get(), work_per_thread);
  cudaDeviceSynchronize();
  
  for(auto _ : state){
    cuda_event_timer raii(state, true); // flush_l2_cache = true, stream = 0
    launch_kernel<functor_type, dispatching_type>(source_table, d_vec.data().get(), work_per_thread);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())*source_size*n_cols*2*sizeof(TypeParam));

}

using namespace cudf;

#define TBM_BENCHMARK_DEFINE(name, TypeParam, functor_type, dispatching_type)                  \
BENCHMARK_DEFINE_F(benchmark, name)(::benchmark::State& state) {                               \
  type_dispatcher_benchmark<TypeParam, functor_type, dispatching_type>(state);                 \
}                                                                                              \
BENCHMARK_REGISTER_F(benchmark, name)->RangeMultiplier(2)->Ranges({{1, 8},{1<<10, 1<<26},{1, 1}})->UseManualTime();

TBM_BENCHMARK_DEFINE(fp64_bandwidth_host, double, BANDWIDTH_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_bandwidth_device, double, BANDWIDTH_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_bandwidth_no, double, BANDWIDTH_BOUND, NO_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_host, double, COMPUTE_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_device, double, COMPUTE_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_no, double, COMPUTE_BOUND, NO_DISPATCHING);

