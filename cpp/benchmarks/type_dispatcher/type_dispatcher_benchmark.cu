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
#include <cudf/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf/types.hpp>
#include <utilities/wrapper_types.hpp>

#include <table/device_table.cuh>

#include <random>

#include "../synchronization/synchronization.h"
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
      return x + static_cast<T>(1);
    }else{
      #pragma unroll
      for(int i = 0; i < 1000; i++){
        x = (x*x + static_cast<T>(1)) - x*x;
      }
      return x;
    }
  }
};

constexpr int block_size = 256;

int launch_configuration(int total_work, int work_per_thread){
  return (total_work + work_per_thread*block_size - 1) / (work_per_thread*block_size);
}

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
  void operator()(gdf_column* source_column, int work_per_thread) {
    ColumnType* source_data = static_cast<ColumnType*>(source_column->data);
    gdf_size_type const n_rows = source_column->size;
    int grid_size = launch_configuration(n_rows, work_per_thread);
    
    // Launch the kernel.
    host_dispatching_kernel<functor_type><<<grid_size, block_size>>>(source_data, n_rows);
  }
};


// >>>>>>
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
// <<<<<<

template<FunctorType functor_type, DispatchingType dispatching_type, class T>
void launch_kernel(cudf::table& input, T** d_ptr, int work_per_thread){
  
  const gdf_size_type n_rows = input.num_rows();
  const gdf_size_type n_cols = input.num_columns();
    
  int grid_size = launch_configuration(n_rows, work_per_thread);
 
  if(dispatching_type == HOST_DISPATCHING){
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

class TypeDispatching : public cudf::benchmark{ };

template<class TypeParam, FunctorType functor_type, DispatchingType dispatching_type>
void type_dispatcher_benchmark(benchmark::State& state){
  const gdf_size_type source_size = static_cast<gdf_size_type>(state.range(0));
  
  const gdf_size_type n_cols = static_cast<gdf_size_type>(state.range(1));
  
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
  
  // For no dispatching:
  std::vector<TypeParam*> h_vec(n_cols);
  
  TypeParam** d_ptr = nullptr;
  
  if(dispatching_type == NO_DISPATCHING){
    RMM_TRY(RMM_ALLOC(&d_ptr, sizeof(TypeParam*)*n_cols, 0));
    for(int c = 0; c < n_cols; c++){
      RMM_TRY(RMM_ALLOC(&h_vec[c], sizeof(TypeParam)*source_size, 0));
    }
    CUDA_TRY(cudaMemcpy(d_ptr, h_vec.data(), sizeof(TypeParam*)*n_cols, cudaMemcpyHostToDevice));
  }
  
  for(auto _ : state){
    cuda_event_timer raii(state);
    launch_kernel<functor_type, dispatching_type>(source_table, d_ptr, work_per_thread);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())*source_size*n_cols*2*sizeof(TypeParam));
  
  if(dispatching_type == NO_DISPATCHING){
    for(int c = 0; c < n_cols; c++){
      RMM_TRY(RMM_FREE(h_vec[c], 0));
    }
    RMM_TRY(RMM_FREE(d_ptr, 0));
  }
}

#define TBM_BENCHMARK_DEFINE(name, TypeParam, functor_type, dispatching_type)                  \
BENCHMARK_DEFINE_F(TypeDispatching, name)(::benchmark::State& state) {                                      \
  type_dispatcher_benchmark<TypeParam, functor_type, dispatching_type>(state);                 \
}                                                                                                           \
BENCHMARK_REGISTER_F(TypeDispatching, name)->RangeMultiplier(2)->Ranges({{1<<10, 1<<26},{1, 1},{1, 1}})->UseManualTime();

TBM_BENCHMARK_DEFINE(fp64_bandwidth_host, double, BANDWIDTH_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_bandwidth_device, double, BANDWIDTH_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_mono_bandwidth_no, double, BANDWIDTH_BOUND, NO_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_host, double, COMPUTE_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_device, double, COMPUTE_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_no, double, COMPUTE_BOUND, NO_DISPATCHING);

