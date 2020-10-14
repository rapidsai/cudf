#include "csv_test_new.cuh"

// f(a)    -> b // upgrade input to state
// f(b, a) -> b // integrate input to state
// f(b, b) -> b // merge state with state
// f(b)    -> c // downgrade state to output

template <typename T,
          typename ReduceOp,
          typename UnaryPredicate,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD>
__global__ void inclusive_scan_copy_if_kernel(device_span<T> input,
                                              device_span<uint32_t> block_temp_count,
                                              device_span<T> block_temp_value,
                                              device_span<uint32_t> output,
                                              ReduceOp reduce_op,
                                              UnaryPredicate predicate_op)
{
  using BlockLoad      = typename cub::BlockLoad<  //
    T,
    BLOCK_DIM_X,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using BlockReduce    = typename cub::BlockReduce<uint32_t, BLOCK_DIM_X>;
  using BlockScan      = typename cub::BlockScan<uint32_t, BLOCK_DIM_X>;
  using BlockScanValue = typename cub::BlockScan<T, BLOCK_DIM_X>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    typename BlockScan::TempStorage scan;
    typename BlockScanValue::TempStorage value;
  } temp_storage;

  T thread_data[ITEMS_PER_THREAD];

  uint32_t const block_offset  = (blockIdx.x * blockDim.x) * ITEMS_PER_THREAD;
  uint32_t const thread_offset = threadIdx.x * ITEMS_PER_THREAD;
  uint32_t valid_items         = input.size() - block_offset;

  BlockLoad(temp_storage.load)  //
    .Load(input.data() + block_offset, thread_data, valid_items);

  auto seed = block_temp_value[blockIdx.x];

  auto get = [seed] __device__(T) { return seed; };

  BlockScanValue(temp_storage.value).InclusiveScan(thread_data, thread_data, reduce_op, get);

  // BlockScanValue(temp_storage.value)  //
  //   .InclusiveScan(thread_data, thread_data, reduce_op);

  uint32_t count_thread = 0;

  for (auto i = 0; i < ITEMS_PER_THREAD; i++) {
    if (thread_offset + i >= valid_items) { break; }
    if (predicate_op(thread_data[i])) {  //
      count_thread++;
    }
  }
  uint32_t count_block;
  uint32_t thread_output_offset;

  BlockScan(temp_storage.scan).InclusiveSum(count_thread, thread_output_offset, count_block);

  // we used inclusive scan to get the sum for count_block, but we need the exclusive scan for
  // thread output offset. We can get that by subtracting count_thread.
  thread_output_offset = thread_output_offset - count_thread;

  if (output.data() == nullptr) {
    // This is the first pass, so just return the block sums.
    // TODO: incorperate cross-thread and cross-block reduction and predicate
    if (threadIdx.x == 0) { block_temp_count[blockIdx.x + 1] = count_block; }

    return;
  }

  // scan + copy_if

  uint32_t const block_output_offset = block_temp_count[blockIdx.x];

  for (auto i = 0; i < ITEMS_PER_THREAD; i++) {
    if (thread_offset + i >= valid_items) { break; }
    if (predicate_op(thread_data[i])) {
      output[block_output_offset + thread_output_offset++] = block_offset + thread_offset + i;
    }
  }
}

template <typename T, typename ReduceOp, typename UnaryPredicate>
rmm::device_vector<uint32_t>  //
inclusive_scan_copy_if(device_span<T> d_input,
                       ReduceOp reduce,
                       UnaryPredicate predicate,
                       cudaStream_t stream = 0)
{
  {
    // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 8 };  // 1b * 1t * 8i : [pass]
    // enum { BLOCK_DIM_X = 8, ITEMS_PER_THREAD = 1 };  // 1b * 8t * 1i : [pass]
    // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 4 };  // 2b * 1t * 4i : [fail]
    enum { BLOCK_DIM_X = 2, ITEMS_PER_THREAD = 2 };  // 2b * 2t * 2i [fail]

    cudf::detail::grid_1d grid(d_input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);

    auto d_block_temp_count = rmm::device_vector<uint32_t>(grid.num_blocks + 1);
    auto d_block_temp_value = rmm::device_vector<T>(grid.num_blocks + 1);

    auto kernel =
      inclusive_scan_copy_if_kernel<T, ReduceOp, UnaryPredicate, BLOCK_DIM_X, ITEMS_PER_THREAD>;

    kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
      d_input,
      d_block_temp_count,
      d_block_temp_value,
      device_span<uint32_t>(),
      reduce,
      predicate);

    // convert block result sizes to block result offsets.
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_block_temp_count.begin(),
                           d_block_temp_count.end(),
                           d_block_temp_count.begin());

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_block_temp_value.begin(),
                           d_block_temp_value.end(),
                           d_block_temp_value.begin(),
                           reduce);

    auto d_output = rmm::device_vector<uint32_t>(d_block_temp_count.back());

    kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
      d_input,
      d_block_temp_count,
      d_block_temp_value,
      d_output,
      reduce,
      predicate);

    cudaStreamSynchronize(stream);

    return d_output;

    return {};
  }
}
