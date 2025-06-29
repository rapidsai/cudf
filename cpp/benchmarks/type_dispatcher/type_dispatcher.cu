/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <type_traits>

enum DispatchingType { HOST_DISPATCHING, DEVICE_DISPATCHING, NO_DISPATCHING };

enum FunctorType { BANDWIDTH_BOUND, COMPUTE_BOUND };

template <class NotFloat, FunctorType ft, class DisableNotFloat = void>
struct Functor {
  static __device__ NotFloat f(NotFloat x) { return x; }
};

template <class Float, FunctorType ft>
struct Functor<Float, ft, std::enable_if_t<std::is_floating_point_v<Float>>> {
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
CUDF_KERNEL void no_dispatching_kernel(T** A, cudf::size_type n_rows, cudf::size_type n_cols)
{
  using F           = Functor<T, functor_type>;
  auto tidx         = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  while (tidx < n_rows) {
    auto const index = static_cast<cudf::size_type>(tidx);
    for (int c = 0; c < n_cols; c++) {
      A[c][index] = F::f(A[c][index]);
    }
    tidx += stride;
  }
}

// This is for HOST_DISPATCHING
template <FunctorType functor_type, class T>
CUDF_KERNEL void host_dispatching_kernel(cudf::mutable_column_device_view source_column)
{
  using F           = Functor<T, functor_type>;
  T* A              = source_column.data<T>();
  auto tidx         = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  while (tidx < source_column.size()) {
    auto const index = static_cast<cudf::size_type>(tidx);
    A[index]         = F::f(A[index]);
    tidx += stride;
  }
}

template <FunctorType functor_type>
struct ColumnHandle {
  template <typename ColumnType, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<ColumnType>())>
  void operator()(cudf::mutable_column_device_view source_column, int work_per_thread)
  {
    cudf::detail::grid_1d grid_config{source_column.size(), block_size};
    int grid_size = grid_config.num_blocks;
    // Launch the kernel.
    host_dispatching_kernel<functor_type, ColumnType><<<grid_size, block_size>>>(source_column);
  }

  template <typename ColumnType, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<ColumnType>())>
  void operator()(cudf::mutable_column_device_view source_column, int work_per_thread)
  {
    CUDF_FAIL("Invalid type to benchmark.");
  }
};

// The following is for DEVICE_DISPATCHING:
// The dispatching is done on device. The loop loops over
// each row (across different columns). Type is dispatched each time
// a column is visited so the total number of dispatching is
// n_rows * n_cols.
template <FunctorType functor_type>
struct RowHandle {
  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  __device__ void operator()(cudf::mutable_column_device_view source, cudf::size_type index)
  {
    using F                 = Functor<T, functor_type>;
    source.data<T>()[index] = F::f(source.data<T>()[index]);
  }

  template <typename T, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
  __device__ void operator()(cudf::mutable_column_device_view source, cudf::size_type index)
  {
    CUDF_UNREACHABLE("Unsupported type.");
  }
};

// This is for DEVICE_DISPATCHING
template <FunctorType functor_type>
CUDF_KERNEL void device_dispatching_kernel(cudf::mutable_table_device_view source)
{
  cudf::size_type const n_rows = source.num_rows();
  auto tidx                    = cudf::detail::grid_1d::global_thread_id();
  auto const stride            = cudf::detail::grid_1d::grid_stride();
  while (tidx < n_rows) {
    auto const index = static_cast<cudf::size_type>(tidx);
    for (cudf::size_type i = 0; i < source.num_columns(); i++) {
      cudf::type_dispatcher(
        source.column(i).type(), RowHandle<functor_type>{}, source.column(i), index);
    }
    tidx += stride;
  }  // while
}

template <FunctorType functor_type, DispatchingType dispatching_type, class T>
void launch_kernel(cudf::mutable_table_view input, T** d_ptr, int work_per_thread)
{
  cudf::size_type const n_rows = input.num_rows();
  cudf::size_type const n_cols = input.num_columns();

  cudf::detail::grid_1d grid_config{n_rows, block_size};
  int grid_size = grid_config.num_blocks;

  if (dispatching_type == HOST_DISPATCHING) {
    // std::vector<cudf::util::cuda::scoped_stream> v_stream(n_cols);
    for (int c = 0; c < n_cols; c++) {
      auto d_column = cudf::mutable_column_device_view::create(input.column(c));
      cudf::type_dispatcher(
        d_column->type(), ColumnHandle<functor_type>{}, *d_column, work_per_thread);
    }
  } else if (dispatching_type == DEVICE_DISPATCHING) {
    auto d_table_view = cudf::mutable_table_device_view::create(input);
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
  auto const n_cols          = static_cast<cudf::size_type>(state.range(0));
  auto const source_size     = static_cast<cudf::size_type>(state.range(1));
  auto const work_per_thread = static_cast<cudf::size_type>(state.range(2));

  auto init = cudf::make_fixed_width_scalar<TypeParam>(static_cast<TypeParam>(0));

  std::vector<std::unique_ptr<cudf::column>> source_column_wrappers;
  std::vector<cudf::mutable_column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(cudf::sequence(source_size, *init));
    source_columns.push_back(*source_column_wrappers[i]);
  }
  cudf::mutable_table_view source_table{source_columns};

  // For no dispatching
  std::vector<rmm::device_buffer> h_vec(n_cols);
  std::vector<TypeParam*> h_vec_p(n_cols);
  std::transform(h_vec.begin(), h_vec.end(), h_vec_p.begin(), [source_size](auto& col) {
    col.resize(source_size * sizeof(TypeParam), cudf::get_default_stream());
    return static_cast<TypeParam*>(col.data());
  });
  rmm::device_uvector<TypeParam*> d_vec(n_cols, cudf::get_default_stream());

  if (dispatching_type == NO_DISPATCHING) {
    CUDF_CUDA_TRY(
      cudaMemcpy(d_vec.data(), h_vec_p.data(), sizeof(TypeParam*) * n_cols, cudaMemcpyDefault));
  }

  // Warm up
  launch_kernel<functor_type, dispatching_type>(source_table, d_vec.data(), work_per_thread);
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  for (auto _ : state) {
    cuda_event_timer raii(state, true);  // flush_l2_cache = true, stream = 0
    launch_kernel<functor_type, dispatching_type>(source_table, d_vec.data(), work_per_thread);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * source_size * n_cols * 2 *
                          sizeof(TypeParam));
}

class TypeDispatcher : public cudf::benchmark {};

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
