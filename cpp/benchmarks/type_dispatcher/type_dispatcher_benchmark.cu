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

#include <table/device_table.cuh>

#include <random>

enum DispatchingType {
  HOST_DISPATCHING,
  DEVICE_DISPATCHING,
  NO_DISPATCHING
};

enum FunctorType {
  BANDWIDTH_BOUND,
  COMPUTE_BOUND
};

enum KernelType {
  MONOLITHIC_KERNEL,
  GRID_STRIDE_LOOP_KERNEL
};

template<class T, FunctorType ft>
struct Functor{
  static __device__ T f(T x){
    if(ft == BANDWIDTH_BOUND){
      return x + static_cast<T>(1);
    }else if(ft == COMPUTE_BOUND){
      #pragma unroll
      for(int i = 0; i < 100; i++){
        x += static_cast<T>(1);
      }
      return x;
    }
    return static_cast<T>(0);
  }
};

template<class F, class T>
__global__ void host_dispatching_kernel(T* A, gdf_size_type n_rows){
  gdf_index_type index = blockIdx.x * blockDim.x + threadIdx.x;
  while(index < n_rows){
    A[index] = F::f(A[index]);
    index += blockDim.x * gridDim.x;
  }
}

// This is for HOST_DISPATCHING
template<KernelType kernel_type, FunctorType functor_type>
struct ColumnHandle {
  template <typename ColumnType>
  void operator()(gdf_column* source_column) {
    
    using F = Functor<ColumnType, functor_type>;

    ColumnType* source_data = static_cast<ColumnType*>(source_column->data);

    gdf_size_type const n_rows = source_column->size;
   
    int block_size, grid_size;
    if(kernel_type == MONOLITHIC_KERNEL){
      block_size = 256;
      grid_size = (n_rows + block_size - 1) / block_size;
    }else{
      CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, host_dispatching_kernel<F, ColumnType>));
    }
    
    // Launch the kernel.
    host_dispatching_kernel<F><<<grid_size, block_size>>>(source_data, n_rows);
  }
};

template<FunctorType functor_type>
struct RowHandle {
  template<typename T>
  void operator()(const gdf_column& source, gdf_index_type index){
    using F = Functor<T, functor_type>;
    static_cast<T*>(source.data)[index] = 
      F::f(static_cast<T*>(source.data)[index]);
  }
};

// This is for DEVICE_DISPATCHING
template<FunctorType functor_type>
__global__ void device_dispatching_monolithic(device_table source){

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

template<KernelType kernel_type, FunctorType functor_type, DispatchingType dispatching_type>
void launch_kernel(cudf::table& input){
  
  const gdf_size_type n_rows = input.num_rows();
  const gdf_size_type n_cols = input.num_columns();
 
  if(dispatching_type == HOST_DISPATCHING){
    
    for(int c = 0; c < n_cols; c++){
      cudf::type_dispatcher(input.get_column(c)->dtype, ColumnHandle<kernel_type, functor_type>{}, input.get_column(c));
    }

  }else if(dispatching_type == DEVICE_DISPATCHING){
    
    if(kernel_type == MONOLITHIC_KERNEL){
      constexpr int block_size = 256;
      const int grid_size = (n_rows + block_size - 1) / block_size;
      auto d_source_table = device_table::create(input);
      // Launcht the kernel
      device_dispatching_monolithic<functor_type><<<grid_size, block_size>>>(*d_source_table);
    }else if(kernel_type == GRID_STRIDE_LOOP_KERNEL){
      // In this case we don't need to do anything?
    }

  }else if(dispatching_type == NO_DISPATCHING){
  
  }
}

template<class TypeParam, KernelType kernel_type, FunctorType functor_type, DispatchingType dispatching_type>
void type_dispatcher_benchmark(benchmark::State& state){
  const gdf_size_type source_size{(gdf_size_type)state.range(0)};
  const gdf_size_type destination_size{(gdf_size_type)state.range(0)};

  cudf::test::column_wrapper<TypeParam> source_column(
      source_size,
      [](gdf_index_type row) { return static_cast<TypeParam>(row); },
      [](gdf_index_type row) { return true; });

  gdf_column* raw_source = source_column.get();

  cudf::table source_table{&raw_source, 1};
  
  for(auto _ : state){
    launch_kernel<kernel_type, functor_type, dispatching_type>(source_table);
  }
}

BENCHMARK_TEMPLATE(type_dispatcher_benchmark, double, MONOLITHIC_KERNEL, BANDWIDTH_BOUND, HOST_DISPATCHING)->RangeMultiplier(2)->Range(1<<10, 1<<26);
BENCHMARK_TEMPLATE(type_dispatcher_benchmark, double, GRID_STRIDE_LOOP_KERNEL, BANDWIDTH_BOUND, HOST_DISPATCHING)->RangeMultiplier(2)->Range(1<<10, 1<<26);
