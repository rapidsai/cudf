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

enum KernelType {
  MONOLITHIC_KERNEL,
  GRID_STRIDE_LOOP_KERNEL
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

template<FunctorType functor_type, class T>
__global__ void host_dispatching_kernel(T* A, gdf_size_type n_rows){
  
  using F = Functor<T, functor_type>;
  
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
    

    ColumnType* source_data = static_cast<ColumnType*>(source_column->data);

    gdf_size_type const n_rows = source_column->size;
   
    int block_size, grid_size;
    if(kernel_type == MONOLITHIC_KERNEL){
      block_size = 256;
      grid_size = (n_rows + block_size - 1) / block_size;
    }else{
      CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, host_dispatching_kernel<functor_type, ColumnType>));
    }
    
    // Launch the kernel.
    host_dispatching_kernel<functor_type><<<grid_size, block_size>>>(source_data, n_rows);
  }
};


// >>>>>>
// The following is for DEVICE_DISPATCHING with MONOLITHIC_KERNEL:
// The dispatching is done on device. The monolithic loop loops over
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
// <<<<<<

// >>>>>>
// The following is for DEVICE_DISPATCHING with GRID_STRIDE_LOOP:
// The dispatching is done on device. The grid stride loop loops over
// each column after type of that column is dispatched.
// In principle this version should be the same as HOST_DISPATCHING with 
// GRID_STRIDE_LOOP, except for the dispatching location.
template<FunctorType functor_type>
struct ColumnHandleDevice {
  template <class T>
  __device__ void operator()(const gdf_column& source){
    const gdf_index_type n_rows = source.size;
    gdf_index_type index = threadIdx.x + blockIdx.x * blockDim.x;
    using F = Functor<T, functor_type>;
    while(index < n_rows){
      static_cast<T*>(source.data)[index] = F::f(static_cast<T*>(source.data)[index]);
      index += blockDim.x * gridDim.x;
    } // while
  }
};

template<FunctorType functor_type>
__global__ void device_dispatching_grid_loop(device_table source){
  
  for(gdf_size_type i = 0; i < source.num_columns(); i++){
      cudf::type_dispatcher(source.get_column(i)->dtype,
                          ColumnHandleDevice<functor_type>{}, *source.get_column(i));

  }
}
// <<<<<<

template<KernelType kernel_type, FunctorType functor_type, DispatchingType dispatching_type, class T>
void launch_kernel(cudf::table& input, T* d_ptr){
  
  const gdf_size_type n_rows = input.num_rows();
  const gdf_size_type n_cols = input.num_columns();
 
  if(dispatching_type == HOST_DISPATCHING){
    
    for(int c = 0; c < n_cols; c++){
      cudf::type_dispatcher(input.get_column(c)->dtype, ColumnHandle<kernel_type, functor_type>{}, input.get_column(c));
    }

  }else if(dispatching_type == DEVICE_DISPATCHING){
    
    auto d_source_table = device_table::create(input);
    
    if(kernel_type == MONOLITHIC_KERNEL){
      constexpr int block_size = 256;
      const int grid_size = (n_rows + block_size - 1) / block_size;
      // Launch the kernel
      device_dispatching_monolithic<functor_type><<<grid_size, block_size>>>(*d_source_table);
    }else if(kernel_type == GRID_STRIDE_LOOP_KERNEL){
      int block_size, grid_size;
      auto f = device_dispatching_monolithic<functor_type>;
      CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, f));
      // Launch the kernel
      f<<<grid_size, block_size>>>(*d_source_table);
    }

  }else if(dispatching_type == NO_DISPATCHING){
    int block_size, grid_size;
    auto f = host_dispatching_kernel<functor_type, T>;
    if(kernel_type == MONOLITHIC_KERNEL){
      block_size = 256;
      grid_size = (n_rows*n_cols + block_size - 1) / block_size;
    }else if(kernel_type == GRID_STRIDE_LOOP_KERNEL){
      CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, f));
    }
    // Launch the kernel
    f<<<grid_size, block_size>>>(d_ptr, n_rows*n_cols);
  }
}

class TypeDispatching : public benchmark::Fixture {
public: 
  virtual void SetUp(const ::benchmark::State& state) {
    rmmOptions_t options{PoolAllocation, 0, false};
    rmmInitialize(&options);
  }

  virtual void TearDown(const ::benchmark::State& state) {
    rmmFinalize();
  }

  // eliminate partial override warnings (see benchmark/benchmark.h)
  virtual void SetUp(::benchmark::State& st) {
    SetUp(const_cast<const ::benchmark::State&>(st));
  }
  virtual void TearDown(::benchmark::State& st) {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }
};

template<class TypeParam, KernelType kernel_type, FunctorType functor_type, DispatchingType dispatching_type>
void type_dispatcher_benchmark(benchmark::State& state){
  const gdf_size_type source_size = static_cast<gdf_size_type>(state.range(0));
  
  const gdf_size_type n_cols = static_cast<gdf_size_type>(state.range(1));

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
  TypeParam* d_ptr;
  RMM_TRY(RMM_ALLOC(&d_ptr, sizeof(TypeParam)*source_size*n_cols, 0));

  for(auto _ : state){
    launch_kernel<kernel_type, functor_type, dispatching_type>(source_table, d_ptr);
    CUDA_TRY(cudaDeviceSynchronize());
  }


  for(int c = 0; c < n_cols; c++){
    std::vector<TypeParam> result_data;
    std::vector<gdf_valid_type> result_bitmask;
    std::tie(result_data, result_bitmask) = v_src[c].to_host();
    for (gdf_index_type i = 0; i < source_size; i++) {
      // assert(static_cast<TypeParam>(12)+static_cast<TypeParam>(i) == result_data[i]);
      // if(i==100) printf("%.8f vs %.8f\n", static_cast<TypeParam>(12)+static_cast<TypeParam>(i), result_data[i]);
    }
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())*state.range(0)*n_cols*2*sizeof(TypeParam));
 
  RMM_TRY(RMM_FREE(d_ptr, 0));
}

#define TBM_BENCHMARK_DEFINE(name, TypeParam, kernel_type, functor_type, dispatching_type)                  \
BENCHMARK_DEFINE_F(TypeDispatching, name)(::benchmark::State& state) {                                      \
  type_dispatcher_benchmark<TypeParam, kernel_type, functor_type, dispatching_type>(state);                 \
}

TBM_BENCHMARK_DEFINE(fp64_mono_bndwdth_host, double, MONOLITHIC_KERNEL, BANDWIDTH_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_bndwdth_host, double, GRID_STRIDE_LOOP_KERNEL, BANDWIDTH_BOUND, HOST_DISPATCHING);

TBM_BENCHMARK_DEFINE(fp64_mono_bndwdth_devc, double, MONOLITHIC_KERNEL, BANDWIDTH_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_bndwdth_devc, double, GRID_STRIDE_LOOP_KERNEL, BANDWIDTH_BOUND, DEVICE_DISPATCHING);

TBM_BENCHMARK_DEFINE(fp64_mono_bndwdth___no, double, MONOLITHIC_KERNEL, BANDWIDTH_BOUND, NO_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_bndwdth___no, double, GRID_STRIDE_LOOP_KERNEL, BANDWIDTH_BOUND, NO_DISPATCHING);

TBM_BENCHMARK_DEFINE(fp64_mono_compute_host, double, MONOLITHIC_KERNEL, COMPUTE_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_compute_host, double, GRID_STRIDE_LOOP_KERNEL, COMPUTE_BOUND, HOST_DISPATCHING);

TBM_BENCHMARK_DEFINE(fp64_mono_compute_devc, double, MONOLITHIC_KERNEL, COMPUTE_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_compute_devc, double, GRID_STRIDE_LOOP_KERNEL, COMPUTE_BOUND, DEVICE_DISPATCHING);

TBM_BENCHMARK_DEFINE(fp64_mono_compute___no, double, MONOLITHIC_KERNEL, COMPUTE_BOUND, NO_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_loop_compute___no, double, GRID_STRIDE_LOOP_KERNEL, COMPUTE_BOUND, NO_DISPATCHING);


BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_bndwdth_host)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_bndwdth_host)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_bndwdth_devc)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_bndwdth_devc)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_bndwdth___no)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_bndwdth___no)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_compute_host)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_compute_host)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_compute_devc)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_compute_devc)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

BENCHMARK_REGISTER_F(TypeDispatching, fp64_mono_compute___no)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});
BENCHMARK_REGISTER_F(TypeDispatching, fp64_loop_compute___no)->RangeMultiplier(2)->Ranges({{1<<16, 1<<26},{1,4}});

