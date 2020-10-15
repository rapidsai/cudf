#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <cstdint>
#include "cudf/utilities/error.hpp"

using cudf::detail::device_span;

template <typename T>
struct doop {
  T value;
  constexpr T operator()(T) { return value; }
};

template <typename T_,
          typename ScanOp_,
          typename PredicateOp_,
          int BLOCK_DIM_X_,
          int ITEMS_PER_THREAD_>
struct InclusiveCopyIfPolicy {
  static constexpr int BLOCK_DIM_X      = BLOCK_DIM_X_;
  static constexpr int ITEMS_PER_THREAD = ITEMS_PER_THREAD_;
  using T                               = T_;
  using ScanOp                          = ScanOp_;
  using PredicateOp                     = PredicateOp_;

  template <typename TData>
  using BlockLoad = cub::BlockLoad<  //
    TData,
    BLOCK_DIM_X,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

  template <typename TData>
  using BlockStore = cub::BlockStore<  //
    TData,
    BLOCK_DIM_X,
    ITEMS_PER_THREAD,
    cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE>;

  template <typename TData>
  using BlockScan = cub::BlockScan<  //
    TData,
    BLOCK_DIM_X>;

  template <typename TData>
  using BlockReduce = cub::BlockReduce<  //
    TData,
    BLOCK_DIM_X>;
};

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_1(  //
  device_span<T> input,
  device_span<T> block_temp_values,
  ScanOp scan_op)
{
  // block-wise aggregates
  __shared__ union {
    typename Policy::template BlockLoad<T>::TempStorage load_value;
    typename Policy::template BlockScan<T>::TempStorage scan_value;
  } temp_storage;

  auto block_offset = ITEMS_PER_THREAD * blockIdx.x * blockDim.x;
  auto valid_values = input.size() - block_offset;

  T thread_values[ITEMS_PER_THREAD];

  Policy::template BlockLoad<T>(temp_storage.load_value)  //
    .Load(input.data() + block_offset,                    //
          thread_values);

  T block_value_aggregate;

  // TODO: use a sequential reduce here instead.
  Policy::template BlockScan<T>(temp_storage.scan_value)  //
    .InclusiveScan(thread_values,                         //
                   thread_values,
                   scan_op,
                   block_value_aggregate);

  if (threadIdx.x == 0) {  //
    // printf("bid(%i) tid(%i) assigning block_value_aggregate\n", blockIdx.x);
    block_temp_values[blockIdx.x + 1] = block_value_aggregate;
  }
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_2(  //
  device_span<T> block_temp_values,
  ScanOp scan_op)
{
  // device-wise aggregates
  __shared__ union {
    typename Policy::template BlockLoad<T>::TempStorage load_value;
    typename Policy::template BlockScan<T>::TempStorage scan_value;
    typename Policy::template BlockStore<T>::TempStorage store_value;
  } temp_storage;

  // auto block_offset = ITEMS_PER_THREAD * blockIdx.x * blockDim.x;
  // auto thread_offset = ITEMS_PER_THREAD * threadIdx.x;
  // auto valid_values = block_temp_values.size() - block_offset;
  auto valid_values = block_temp_values.size();

  T thread_values[ITEMS_PER_THREAD];

  // load
  Policy::template BlockLoad<T>(temp_storage.load_value)  //
    .Load(block_temp_values.data(),                       //
          thread_values,
          valid_values);

  // scan
  Policy::template BlockScan<T>(temp_storage.scan_value)  //
    .InclusiveScan(thread_values,                         //
                   thread_values,
                   scan_op);

  // store
  Policy::template BlockStore<T>(temp_storage.store_value)  //
    .Store(block_temp_values.data(),                        //
           thread_values,
           valid_values);
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_3(  //
  device_span<T> input,
  device_span<T> block_temp_values,
  device_span<uint32_t> block_temp_counts,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
  // block-wise count
  __shared__ union {
    typename Policy::template BlockLoad<T>::TempStorage load_value;
    typename Policy::template BlockScan<T>::TempStorage scan_value;
    typename Policy::template BlockReduce<uint32_t>::TempStorage reduce_count;
  } temp_storage;

  auto block_offset  = ITEMS_PER_THREAD * blockIdx.x * blockDim.x;
  auto thread_offset = ITEMS_PER_THREAD * threadIdx.x;
  auto valid_values  = input.size() - block_offset;

  T thread_values[ITEMS_PER_THREAD];

  Policy::template BlockLoad<T>(temp_storage.load_value)  //
    .Load(input.data() + block_offset,                    //
          thread_values);

  T block_seed = block_temp_values[blockIdx.x];

  auto prefix_op = doop<T>{block_seed};

  Policy::template BlockScan<T>(temp_storage.scan_value)  //
    .InclusiveScan(thread_values,                         //
                   thread_values,
                   scan_op,
                   prefix_op);

  uint32_t count = 0;

  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    count += thread_offset + i < valid_values and predicate_op(thread_values[i]);
  }

  count = Policy::template BlockReduce<uint32_t>(temp_storage.reduce_count)  //
            .Sum(count);

  if (threadIdx.x == 0) {  //
    // printf("bid(%i) tid(%i) block-wide count: %u\n", blockIdx.x, threadIdx.x, count);
    block_temp_counts[blockIdx.x + 1] = count;
  }

  __syncthreads();
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_4(  //
  device_span<uint32_t> block_temp_counts,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
  // device-wise count
  __shared__ union {
    typename Policy::template BlockScan<uint32_t>::TempStorage scan_count;
    typename Policy::template BlockStore<uint32_t>::TempStorage store_count;
    typename Policy::template BlockLoad<uint32_t>::TempStorage load_count;
  } temp_storage;

  auto valid_values = block_temp_counts.size();

  uint32_t thread_counts[ITEMS_PER_THREAD];

  // load
  Policy::template BlockLoad<uint32_t>(temp_storage.load_count)  //
    .Load(block_temp_counts.data(),                              //
          thread_counts,
          valid_values);

  // for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
  //   printf("bid(%i) tid(%i) %i: %u\n", blockIdx.x, threadIdx.x, valid_values, thread_counts[i]);
  // }

  // scan
  Policy::template BlockScan<uint32_t>(temp_storage.scan_count)  //
    .InclusiveSum(thread_counts,                                 //
                  thread_counts);

  // for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
  //   printf("bid(%i) tid(%i) %i: %u\n", blockIdx.x, threadIdx.x, valid_values, thread_counts[i]);
  // }

  // store
  Policy::template BlockStore<uint32_t>(temp_storage.store_count)  //
    .Store(block_temp_counts.data(),                               //
           thread_counts,
           valid_values);
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_5(  //
  device_span<T> input,
  device_span<uint32_t> output,
  device_span<T> block_temp_values,
  device_span<uint32_t> block_temp_counts,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
  // device-wise gather
  __shared__ union {
    typename Policy::template BlockLoad<T>::TempStorage load_value;
    typename Policy::template BlockScan<T>::TempStorage scan_value;
    typename Policy::template BlockLoad<uint32_t>::TempStorage load_count;
    typename Policy::template BlockScan<uint32_t>::TempStorage scan_count;
  } temp_storage;

  auto block_offset  = ITEMS_PER_THREAD * blockIdx.x * blockDim.x;
  auto thread_offset = ITEMS_PER_THREAD * threadIdx.x;
  auto valid_values  = input.size() - block_offset;

  T thread_values[ITEMS_PER_THREAD];

  // load
  Policy::template BlockLoad<T>(temp_storage.load_value)  //
    .Load(input.data() + block_offset,                    //
          thread_values);

  // scan
  T block_values_seed = block_temp_values[blockIdx.x];
  auto prefix_op      = doop<T>{block_values_seed};
  Policy::template BlockScan<T>(temp_storage.scan_value)  //
    .InclusiveScan(thread_values,                         //
                   thread_values,
                   scan_op,
                   prefix_op);

  // scatter
  // TODO: scatter to temp storage, then use BlockStore.
  uint32_t block_counts_seed = block_temp_counts[blockIdx.x];

  uint32_t count = 0;

  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    count += thread_offset + i < valid_values and predicate_op(thread_values[i]);
  }

  Policy::template BlockScan<uint32_t>(temp_storage.scan_count)  //
    .ExclusiveSum(count, count);

  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    if (thread_offset + i < valid_values and predicate_op(thread_values[i])) {
      // printf("bid(%i) tid(%i) [%i] = %i + %i + %i\n",
      //        blockIdx.x,
      //        threadIdx.x,
      //        block_counts_seed,
      //        block_offset,
      //        thread_offset,
      //        i);
      output[block_counts_seed++ + count] = block_offset + thread_offset + i;
    }
  }
}

// void sync(std::string message)
// {
//   cudaDeviceSynchronize();
//   printf("synced\n===== %s =====\n", message.c_str());
//   cudaDeviceSynchronize();
// }

/**
 * @brief inclusive_scan + copy_if
 *
 * f(a)    -> b // upgrade input to state
 * f(b, a) -> b // integrate input to state
 * f(b, b) -> b // merge state with state
 * f(b)    -> c // downgrade state to output
 *
 * @tparam T
 * @tparam ScanOp
 * @tparam PredicateOp
 * @param input
 * @param scan_op
 * @param predicate_op
 * @param stream
 * @return rmm::device_vector<uint32_t>
 */
template <typename T,
          typename ScanOp,
          typename PredicateOp>
rmm::device_vector<uint32_t>  //
inclusive_copy_if(device_span<T> input,
                  ScanOp scan_op,
                  PredicateOp predicate_op,
                  cudaStream_t stream = 0)
{
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 8 };  // 1b * 1t * 8i
  // enum { BLOCK_DIM_X = 8, ITEMS_PER_THREAD = 1 };  // 1b * 8t * 1i
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 4 };  // 2b * 1t * 4i
  enum {  //
    BLOCK_DIM_X      = 128,
    ITEMS_PER_THREAD = 32,
    ITEMS_PER_BLOCK  = BLOCK_DIM_X * ITEMS_PER_THREAD
  };

  using Policy = InclusiveCopyIfPolicy<T, ScanOp, PredicateOp, BLOCK_DIM_X, ITEMS_PER_THREAD>;

  cudf::detail::grid_1d grid(input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);
  auto num_temp_values = grid.num_blocks + 1;

  // TODO: use single-pass chained-scan prefix scan and support any number of inputs.
  CUDF_EXPECTS(num_temp_values < ITEMS_PER_BLOCK, "too many inputs.");

  auto d_block_temp_values = rmm::device_vector<T>(num_temp_values);
  auto d_block_temp_counts = rmm::device_vector<uint32_t>(num_temp_values);
  auto kernel_phase_1      = kernel_pass_1<Policy>;
  auto kernel_phase_2      = kernel_pass_2<Policy>;
  auto kernel_phase_3      = kernel_pass_3<Policy>;
  auto kernel_phase_4      = kernel_pass_4<Policy>;
  auto kernel_phase_5      = kernel_pass_5<Policy>;

  // local aggregates
  kernel_phase_1<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    d_block_temp_values,
    scan_op);

  // global aggregates
  kernel_phase_2<<<1, grid.num_threads_per_block, 0, stream>>>(  //
    d_block_temp_values,
    scan_op);

  // local counts
  kernel_phase_3<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  // global counts
  kernel_phase_4<<<1, grid.num_threads_per_block, 0, stream>>>(  //
    d_block_temp_counts,
    scan_op,
    predicate_op);

  // global indices gather
  auto num_results = static_cast<uint32_t>(d_block_temp_counts.back());
  auto output      = rmm::device_vector<uint32_t>(num_results);

  kernel_phase_5<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    output,
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  return output;
}
