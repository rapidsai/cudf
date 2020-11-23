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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/detail/utilities/cuda.cuh>

#include <random>
#include <type_traits>
#include "../fixture/benchmark_fixture.hpp"
#include "../synchronization/synchronization.hpp"

using namespace cudf;

enum DispatchingType { HOST_DISPATCHING, DEVICE_DISPATCHING, NO_DISPATCHING };

enum FunctorType { BANDWIDTH_BOUND, COMPUTE_BOUND };

template <class NotFloat, FunctorType ft, class DisableNotFloat = void>
struct Functor {
  static __device__ NotFloat f(NotFloat x) { return x; }
};

template <class Float, FunctorType ft>
struct Functor<Float, ft, typename std::enable_if_t<std::is_floating_point<Float>::value>> {
  static __device__ Float f(Float x)
  {
    if (ft == BANDWIDTH_BOUND) {
      return x + static_cast<Float>(1) - static_cast<Float>(1);
    } else {
      for (int i = 0; i < 1000; i++) {
        x = (x * x + static_cast<Float>(1)) - x * x - static_cast<Float>(1);
      }
      return x;
    }
  }
};

constexpr int block_size = 256;

// This is for NO_DISPATCHING
template <FunctorType functor_type, class T>
__global__ void no_dispatching_kernel(T** A, cudf::size_type n_rows, cudf::size_type n_cols)
{
  using F               = Functor<T, functor_type>;
  cudf::size_type index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < n_rows) {
    for (int c = 0; c < n_cols; c++) { A[c][index] = F::f(A[c][index]); }
    index += blockDim.x * gridDim.x;
  }
}

// This is for HOST_DISPATCHING
template <FunctorType functor_type, class T>
__global__ void host_dispatching_kernel(mutable_column_device_view source_column)
{
  using F               = Functor<T, functor_type>;
  T* A                  = source_column.data<T>();
  cudf::size_type index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < source_column.size()) {
    A[index] = F::f(A[index]);
    index += blockDim.x * gridDim.x;
  }
}

template <FunctorType functor_type>
struct ColumnHandle {
  template <typename ColumnType>
  void operator()(mutable_column_device_view source_column,
                  int work_per_thread,
                  rmm::cuda_stream_view stream = rmm::cuda_stream_default)
  {
    cudf::detail::grid_1d grid_config{source_column.size(), block_size};
    int grid_size = grid_config.num_blocks;
    // Launch the kernel.
    host_dispatching_kernel<functor_type, ColumnType>
      <<<grid_size, block_size, 0, stream.value()>>>(source_column);
  }
};

// The following is for DEVICE_DISPATCHING:
// The dispatching is done on device. The loop loops over
// each row (across different columns). Type is dispatched each time
// a column is visited so the total number of dispatching is
// n_rows * n_cols.
template <FunctorType functor_type>
struct RowHandle {
  template <typename T>
  __device__ void operator()(mutable_column_device_view source, cudf::size_type index)
  {
    using F                 = Functor<T, functor_type>;
    source.data<T>()[index] = F::f(source.data<T>()[index]);
  }
};

// This is for DEVICE_DISPATCHING
template <FunctorType functor_type>
__global__ void device_dispatching_kernel(mutable_table_device_view source)
{
  const cudf::size_type n_rows = source.num_rows();
  cudf::size_type index        = threadIdx.x + blockIdx.x * blockDim.x;

  while (index < n_rows) {
    for (cudf::size_type i = 0; i < source.num_columns(); i++) {
      cudf::type_dispatcher(
        source.column(i).type(), RowHandle<functor_type>{}, source.column(i), index);
    }
    index += blockDim.x * gridDim.x;
  }  // while
}

template <FunctorType functor_type, DispatchingType dispatching_type, class T>
void launch_kernel(mutable_table_view input, T** d_ptr, int work_per_thread)
{
  const cudf::size_type n_rows = input.num_rows();
  const cudf::size_type n_cols = input.num_columns();

  cudf::detail::grid_1d grid_config{n_rows, block_size};
  int grid_size = grid_config.num_blocks;

  if (dispatching_type == HOST_DISPATCHING) {
    // std::vector<cudf::util::cuda::scoped_stream> v_stream(n_cols);
    for (int c = 0; c < n_cols; c++) {
      auto d_column = mutable_column_device_view::create(input.column(c));
      cudf::type_dispatcher(
        d_column->type(), ColumnHandle<functor_type>{}, *d_column, work_per_thread);
    }
  } else if (dispatching_type == DEVICE_DISPATCHING) {
    auto d_table_view = mutable_table_device_view::create(input);
    auto f            = device_dispatching_kernel<functor_type>;
    // Launch the kernel
    f<<<grid_size, block_size>>>(*d_table_view);
  } else if (dispatching_type == NO_DISPATCHING) {
    auto f = no_dispatching_kernel<functor_type, T>;
    // Launch the kernel
    f<<<grid_size, block_size>>>(d_ptr, n_rows, n_cols);
  }
}

template <class TypeParam, FunctorType functor_type, DispatchingType dispatching_type>
void type_dispatcher_benchmark(::benchmark::State& state)
{
  const cudf::size_type source_size = static_cast<cudf::size_type>(state.range(1));

  const cudf::size_type n_cols = static_cast<cudf::size_type>(state.range(0));

  const cudf::size_type work_per_thread = static_cast<cudf::size_type>(state.range(2));

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::mutable_column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }
  cudf::mutable_table_view source_table{source_columns};

  // For no dispatching
  std::vector<rmm::device_vector<TypeParam>> h_vec(n_cols,
                                                   rmm::device_vector<TypeParam>(source_size, 0));
  std::vector<TypeParam*> h_vec_p(n_cols);
  for (int c = 0; c < n_cols; c++) { h_vec_p[c] = h_vec[c].data().get(); }
  rmm::device_vector<TypeParam*> d_vec(n_cols);

  if (dispatching_type == NO_DISPATCHING) {
    CUDA_TRY(cudaMemcpy(
      d_vec.data().get(), h_vec_p.data(), sizeof(TypeParam*) * n_cols, cudaMemcpyHostToDevice));
  }

  // Warm up
  launch_kernel<functor_type, dispatching_type>(source_table, d_vec.data().get(), work_per_thread);
  CUDA_TRY(cudaDeviceSynchronize());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    launch_kernel<functor_type, dispatching_type>(
      source_table, d_vec.data().get(), work_per_thread);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * source_size * n_cols * 2 *
                          sizeof(TypeParam));
}

class TypeDispatcher : public cudf::benchmark {
};

#define TBM_BENCHMARK_DEFINE(name, TypeParam, functor_type, dispatching_type)    \
  BENCHMARK_DEFINE_F(TypeDispatcher, name)(::benchmark::State & state)           \
  {                                                                              \
    type_dispatcher_benchmark<TypeParam, functor_type, dispatching_type>(state); \
  }                                                                              \
  BENCHMARK_REGISTER_F(TypeDispatcher, name)                                     \
    ->RangeMultiplier(2)                                                         \
    ->Ranges({{1, 8}, {1 << 10, 1 << 26}, {1, 1}})                               \
    ->UseManualTime();

TBM_BENCHMARK_DEFINE(fp64_bandwidth_host, double, BANDWIDTH_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_bandwidth_device, double, BANDWIDTH_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_bandwidth_no, double, BANDWIDTH_BOUND, NO_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_host, double, COMPUTE_BOUND, HOST_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_device, double, COMPUTE_BOUND, DEVICE_DISPATCHING);
TBM_BENCHMARK_DEFINE(fp64_compute_no, double, COMPUTE_BOUND, NO_DISPATCHING);
