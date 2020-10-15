#include "csv_test_new.cuh"
#include "rmm/thrust_rmm_allocator.h"

// f(a)    -> b // upgrade input to state
// f(b, a) -> b // integrate input to state
// f(b, b) -> b // merge state with state
// f(b)    -> c // downgrade state to output

template <typename T,
          typename ScanOp,
          typename PredicateOp,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD>
__global__ void kernel_pass_scan_1(  //
  device_span<T> input,
  device_span<T> block_temp_values,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
  // using BlockLoadValue = typename cub::  //
  //   BlockLoad<T, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

  // using BlockScanValue = typename cub::  //
  //   BlockScan<T, BLOCK_DIM_X>;

  // using BlockScanCount = typename cub::  //
  //   BlockScan<uint32_t, BLOCK_DIM_X>;

  // __shared__ union {
  //   typename BlockLoadValue::TempStorage load_value;
  //   typename BlockScanValue::TempStorage scan_value;
  //   typename BlockScanCount::TempStorage scan_count;
  // } temp_storage;

  // uint32_t const block_offset  = (blockIdx.x * blockDim.x) * ITEMS_PER_THREAD;
  // uint32_t const thread_offset = threadIdx.x * ITEMS_PER_THREAD;
  // uint32_t const valid_items   = input.size() - block_offset;

  // T thread_values[ITEMS_PER_THREAD];

  // T block_aggregate_value[ITEMS_PER_THREAD];
  // uint32_t block_aggregate_count = 0;

  // // ===== LOAD =====
  // // load the block-wise aggregates
  // BlockLoadValue(temp_storage.load_value)  //
  //   .Load(                                 //
  //     block_temp_values.data() + block_offset,
  //     thread_values,
  //     valid_items);

  // // ===== SCAN =====
  // // scan the block-wise aggregates to create true aggregates
  // BlockScanValue(temp_storage.scan_value)  //
  //   .InclusiveScan(                        //
  //     thread_values,
  //     thread_values,
  //     scan_op,
  //     block_aggregate_value);

  // // scan the block-wise aggregates to create true aggregates
  // uint32_t thread_counts;

  // for (auto i = 0; i < ITEMS_PER_THREAD; i++) {  //
  //   thread_counts += thread_offset + i < valid_items and predicate_op(thread_values[i]);
  // }

  // // TODO: change to reduce.
  // BlockScanCount(temp_storage.scan_count)  //
  //   .InclusiveSum(                         //
  //     thread_counts,
  //     thread_counts,
  //     block_aggregate_count);

  // if (threadIdx.x == 0) {
  //   // store block-wise aggregates
  //   block_temp_values[blockIdx.x] = block_aggregate_value;
  //   block_temp_counts[blockIdx.x] = block_aggregate_count;
  // }
}

template <typename T,
          typename ScanOp,
          typename PredicateOp,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD>
__global__ void kernel_pass_scan_2(  //
  device_span<T> block_temp_values,
  ScanOp scan_op)
{
  // using BlockLoadValue = typename cub::  //
  //   BlockLoad<T, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  // using BlockLoadCount = typename cub::  //
  //   BlockLoad<uint32_t,
  //             BLOCK_DIM_X,
  //             ITEMS_PER_THREAD,
  //             cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

  // using BlockScanValue = typename cub::  //
  //   BlockScan<T, BLOCK_DIM_X>;
  // using BlockScanCount = typename cub::  //
  //   BlockScan<uint32_t, BLOCK_DIM_X>;

  // using BlockStoreValue = typename cub::  //
  //   BlockStore<T, BLOCK_DIM_X, ITEMS_PER_THREAD,
  //   cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE>;
  // using BlockStoreValue = typename cub::  //
  //   BlockStore<uint32_t,
  //              BLOCK_DIM_X,
  //              ITEMS_PER_THREAD,
  //              cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE>;

  // using BlockScanCount = typename cub::  //
  //   BlockScan<uint32_t, BLOCK_DIM_X>;

  // __shared__ union {
  //   typename BlockLoadValue::TempStorage load_value;
  //   typename BlockScanValue::TempStorage scan_value;
  //   typename BlockStoreValue::TempStorage store_value;
  //   typename BlockLoadCount::TempStorage load_count;
  //   typename BlockScanCount::TempStorage scan_count;
  //   typename BlockStoreCount::TempStorage store_count;
  // } temp_storage;

  // uint32_t const block_offset = (blockIdx.x * blockDim.x) * ITEMS_PER_THREAD;
  // uint32_t const valid_items  = block_temp_values.size() - block_offset;

  // T thread_values[ITEMS_PER_THREAD];
  // uint32_t thread_counts[ITEMS_PER_THREAD];

  // // ===== LOAD =====
  // // load the block-wise aggregates
  // BlockLoadValue(temp_storage.load_value)  //
  //   .Load(                                 //
  //     block_temp_values.data() + block_offset,
  //     thread_values,
  //     valid_items);

  // // load the block-wise aggregates
  // BlockLoadCount(temp_storage.load_count)  //
  //   .Load(                                 //
  //     block_temp_counts.data() + block_offset,
  //     thread_counts,
  //     valid_items);

  // // ===== SCAN =====
  // // scan the block-wise aggregates to create true aggregates
  // BlockScanValue(temp_storage.scan_value)  //
  //   .InclusiveScan(                        //
  //     thread_values,
  //     thread_values,
  //     scan_op);

  // BlockStoreCount(temp_storage.scan_count)  //
  //   .InclusiveScan(                         //
  //     thread_counts,
  //     thread_counts,
  //     scan_op);

  // // ===== STORE =====
  // // store device-wide aggregates
  // BlockStoreValue(temp_storage.store_value)          //
  //   .Store(block_temp_values.data() + block_offset,  //
  //          thread_values,
  //          valid_items);

  // BlockStoreCount(temp_storage.store_count)          //
  //   .Store(block_temp_counts.data() + block_offset,  //
  //          thread_counts,
  //          valid_items);
}
template <typename T,
          typename ScanOp,
          typename PredicateOp,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD>
__global__ void kernel_pass_gather(  //
  device_span<T> input,
  device_span<uint32_t> output,
  device_span<T> block_temp_value,
  device_span<uint32_t> block_temp_count,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
  // using BlockLoadValue = typename cub::  //
  //   BlockLoad<T, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

  // using BlockScanValue = typename cub::  //
  //   BlockScan<T, BLOCK_DIM_X>;

  // using BlockScanCount = typename cub::  //
  //   BlockScan<uint32_t, BLOCK_DIM_X>;

  // __shared__ union {
  //   typename BlockLoadValue::TempStorage load_value;
  //   typename BlockLoadValue::TempStorage load_count;
  //   typename BlockScanValue::TempStorage scan_value;
  //   typename BlockScanCount::TempStorage scan_count;
  // } temp_storage;

  // uint32_t const block_offset  = (blockIdx.x * blockDim.x) * ITEMS_PER_THREAD;
  // uint32_t const thread_offset = threadIdx.x * ITEMS_PER_THREAD;
  // uint32_t const valid_items   = input.size() - block_offset;

  // T thread_values[ITEMS_PER_THREAD];

  // // load a sequential block of inputs
  // BlockLoadValue(temp_storage.load_value)  //
  //   .Load(                                 //
  //     input.data() + block_offset,
  //     thread_values,
  //     valid_items);

  // T block_aggregate_value;

  // // scan the inputs to create block-wise aggregates
  // BlockScanValue(temp_storage.scan_value)  //
  //   .InclusiveScan(                        //
  //     thread_values,
  //     thread_values,
  //     scan_op,
  //     block_aggregate_value);

  // // uint32_t count[ITEMS_PER_THREAD];

  // // for (auto i = 0; i < ITEMS_PER_THREAD; i++) {
  // //   count[i] = predicate_op(block_offset + thread_offset + thread_values[i]);
  // // }

  // // uint32_t block_aggregate_count = 0;

  // // // requesting size
  // // BlockScanCount(temp_storage.scan_count)  //
  // //   .InclusiveScan(                        //
  // //     count,
  // //     count,
  // //     block_aggregate_count);
}

template <typename T, typename ScanOp, typename PredicateOp>
rmm::device_vector<uint32_t>  //
inclusive_scan_copy_if(device_span<T> input,
                       ScanOp scan_op,
                       PredicateOp predicate_op,
                       cudaStream_t stream = 0)
{
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 8 };  // 1b * 1t * 8i : [pass]
  // enum { BLOCK_DIM_X = 8, ITEMS_PER_THREAD = 1 };  // 1b * 8t * 1i : [pass]
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 4 };  // 2b * 1t * 4i : [fail]
  enum { BLOCK_DIM_X = 2, ITEMS_PER_THREAD = 2 };  // 2b * 2t * 2i : [fail]

  cudf::detail::grid_1d grid(input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);

  auto d_block_temp_values = rmm::device_vector<T>(grid.num_blocks + 1);
  auto d_block_temp_counts = rmm::device_vector<uint32_t>(grid.num_blocks + 1);

  // block-wise aggregates
  auto kernel_scan_1 = kernel_pass_scan_1<T, ScanOp, PredicateOp, BLOCK_DIM_X, ITEMS_PER_THREAD>;

  kernel_scan_1<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    d_block_temp_values,
    scan_op,
    predicate_op);

  // device-wise aggregates
  auto kernel_scan_2 = kernel_pass_scan_2<T, ScanOp, PredicateOp, BLOCK_DIM_X, ITEMS_PER_THREAD>;

  kernel_scan_2<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    d_block_temp_values,
    scan_op,
    predicate_op);

  // device-wise count, allocate, gather
  auto kernel_gather = kernel_pass_gather<T, ScanOp, PredicateOp, BLOCK_DIM_X, ITEMS_PER_THREAD>;

  kernel_gather<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    device_span<uint32_t>(nullptr, 0),
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  auto output = rmm::device_vector<uint32_t>(d_block_temp_counts.back());

  kernel_gather<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    output,
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  return output;
}
