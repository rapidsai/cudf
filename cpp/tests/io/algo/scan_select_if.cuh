#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <iterator>
#include "cudf/types.hpp"
#include "dependencies/cub/cub/block/block_exchange.cuh"
#include "rmm/device_buffer.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/thrust_rmm_allocator.h"

#include <type_traits>

enum class select_scan_if_mode { count_only, count_and_gather };

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

template <typename T, int N>
inline __device__ void print_array_2(char const* message,
                                     uint32_t num_items_remaining,
                                     T (&items)[N])
{
  printf("b(%i) t(%i) remaining(%i) %s: %i %i\n",
         blockIdx.x,
         threadIdx.x,
         num_items_remaining,
         message,
         items[0],
         items[1]);
}

template <typename InputIterator,
          typename OutputIterator,
          typename OutputNumSelectedIterator,
          typename ScanOperator,
          typename SelectOperator,
          int THREADS_PER_BLOCK,
          int ITEMS_PER_THREAD>
struct agent {
  enum { ITEMS_PER_TILE = ITEMS_PER_THREAD * THREADS_PER_BLOCK };

  using T = typename std::iterator_traits<InputIterator>::value_type;

  using BlockLoadItem = cub::BlockLoad<  //
    T,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  using BlockScanItem = cub::BlockScan<  //
    T,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  using BlockScanCount = cub::BlockScan<  //
    uint32_t,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  using PrefixOpItem  = cub::TilePrefixCallbackOp<T, ScanOperator, cub::ScanTileState<T>>;
  using PrefixOpCount = cub::TilePrefixCallbackOp<uint32_t, cub::Sum, cub::ScanTileState<uint32_t>>;

  InputIterator d_input;
  OutputIterator d_output;
  OutputNumSelectedIterator d_num_selected_output;
  uint32_t num_items;
  ScanOperator scan_op;
  SelectOperator select_op;
  cub::ScanTileState<uint32_t>& count_state;
  cub::ScanTileState<T>& item_state;
  select_scan_if_mode count_num_selected_only;

  __device__ agent(InputIterator d_input,
                   OutputIterator d_output,
                   OutputNumSelectedIterator d_num_selected_output,
                   uint32_t num_items,
                   ScanOperator scan_op,
                   SelectOperator select_op,
                   cub::ScanTileState<uint32_t>& count_state,
                   cub::ScanTileState<T>& item_state,
                   select_scan_if_mode count_num_selected_only)
    : d_input(d_input),
      d_output(d_output),
      d_num_selected_output(d_num_selected_output),
      num_items(num_items),
      scan_op(scan_op),
      select_op(select_op),
      count_state(count_state),
      item_state(item_state),
      count_num_selected_only(count_num_selected_only)
  {
  }

  inline __device__ void consume_range(int num_tiles, int start_tile)
  {
    int tile_idx            = start_tile + blockIdx.x;
    int tile_offset         = ITEMS_PER_TILE * tile_idx;
    int num_items_remaining = num_items - tile_offset;

    if (num_items_remaining <= 0) { return; }

    typename std::iterator_traits<OutputNumSelectedIterator>::value_type num_selected;

    if (tile_idx < num_tiles - 1) {
      num_selected = consume_tile<false>(tile_idx, tile_offset, num_items_remaining);
    } else {
      num_selected = consume_tile<true>(tile_idx, tile_offset, num_items_remaining);

      if (threadIdx.x == 0) {  //
        *d_num_selected_output = num_selected;

        printf("b(%i) t(%i) num selected: %i\n", blockIdx.x, threadIdx.x, num_selected);
      }
    }
  }

  template <bool is_last_tile>
  inline __device__ uint32_t consume_tile(uint32_t tile_idx,
                                          uint32_t tile_offset,
                                          uint32_t num_items_remaining)
  {
    T items[ITEMS_PER_THREAD];

    {
      __shared__ typename BlockLoadItem::TempStorage item_load;

      if (is_last_tile) {
        BlockLoadItem(item_load).Load(d_input + tile_offset, items, num_items_remaining);
      } else {
        BlockLoadItem(item_load).Load(d_input + tile_offset, items);
      }
    }

    __syncthreads();

    {
      __shared__ typename BlockScanItem::TempStorage item_scan;
      __shared__ typename PrefixOpItem::TempStorage item_prefix;

      // Scan Inputs
      if (tile_idx == 0) {
        T block_aggregate;
        BlockScanItem(item_scan).InclusiveScan(items, items, scan_op, block_aggregate);
        if (threadIdx.x == 0 and not is_last_tile) { item_state.SetInclusive(0, block_aggregate); }
      } else {
        auto prefix_op = PrefixOpItem(item_state, item_prefix, scan_op, tile_idx);
        BlockScanItem(item_scan).InclusiveScan(items, items, scan_op, prefix_op);
      }
    }

    __syncthreads();

    uint32_t selection_flags[ITEMS_PER_THREAD];

    // Initialize Selection Flags
    for (uint64_t i = 0; i < ITEMS_PER_THREAD; i++) {
      selection_flags[i] = 0;
      if (threadIdx.x * ITEMS_PER_THREAD + i < num_items_remaining) {
        selection_flags[i] = select_op(items[i]);
      }
    }

    // Scan Selection
    uint32_t selection_indices[ITEMS_PER_THREAD];
    uint32_t num_selections;

    __syncthreads();
    {
      __shared__ typename BlockScanCount::TempStorage count_scan;
      __shared__ typename PrefixOpCount::TempStorage count_prefix;

      if (tile_idx == 0) {
        BlockScanCount(count_scan).ExclusiveSum(selection_flags, selection_indices, num_selections);
        if (threadIdx.x == 0 and not is_last_tile) { count_state.SetInclusive(0, num_selections); }
      } else {
        auto prefix_op = PrefixOpCount(count_state, count_prefix, cub::Sum(), tile_idx);
        BlockScanCount(count_scan).ExclusiveSum(selection_flags, selection_indices, prefix_op);
        num_selections = prefix_op.GetInclusivePrefix();
      }
    }

    printf("b(%i) t(%i) num selections: %i\n", blockIdx.x, threadIdx.x, num_selections);

    __syncthreads();

    if (count_num_selected_only == select_scan_if_mode::count_only) {  //
      return num_selections;
    }

    // Scatter

    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      if (selection_flags[i]) {
        if (selection_indices[i] < num_selections) {  //
          d_output[selection_indices[i]] = i;
        }
      }
    }

    __syncthreads();

    return num_selections;
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_buffer buffer;
  cub::ScanTileState<T> state;

  scan_tile_state() = default;
  scan_tile_state(uint32_t num_tiles)
  {
    uint64_t temp_storage_bytes;
    CUDA_TRY(cub::ScanTileState<T>::AllocationSize(num_tiles, temp_storage_bytes));
    buffer = rmm::device_buffer(temp_storage_bytes);
    state  = cub::ScanTileState<T>();
    state.Init(num_tiles, buffer.data(), temp_storage_bytes);
  }
};

/**
 * @brief
 *
 * @tparam Policy
 * @param tile_state
 */
template <typename T>
__global__ void initialization_pass_kernel(  //
  cub::ScanTileState<uint32_t> count_state,
  cub::ScanTileState<T> item_state,
  uint32_t num_tiles)
{
  count_state.InitializeStatus(num_tiles);
  item_state.InitializeStatus(num_tiles);
}

template <typename T,
          typename InputIterator,
          typename OutputIterator,
          typename OutputNumSelectedIterator,
          typename ScanOperator,
          typename SelectOperator,
          int THREADS_PER_BLOCK,
          int ITEMS_PER_THREAD>
__global__ void execution_pass_kernel(  //
  InputIterator d_input,
  OutputIterator d_output,
  OutputNumSelectedIterator d_num_selected_output,
  ScanOperator scan_op,
  SelectOperator select_op,
  uint32_t num_items,
  uint32_t num_tiles,
  uint32_t start_tile,
  cub::ScanTileState<uint32_t> count_state,
  cub::ScanTileState<T> item_state,
  select_scan_if_mode count_num_selected_only)
{
  using Agent = agent<InputIterator,
                      OutputIterator,
                      OutputNumSelectedIterator,
                      ScanOperator,
                      SelectOperator,
                      THREADS_PER_BLOCK,
                      ITEMS_PER_THREAD>;

  Agent(  //
    d_input,
    d_output,
    d_num_selected_output,
    num_items,
    scan_op,
    select_op,
    count_state,
    item_state,
    count_num_selected_only)
    .consume_range(num_tiles, start_tile);
}

/**
 * @brief
 *
 * @tparam Policy
 */
template <typename InputIterator,  //
          typename OutputIterator,
          typename OutputNumResultsIterator,
          typename ScanOperator,
          typename SelectOperator>
struct scan_select_if_dispatch {
  using T = typename std::iterator_traits<InputIterator>::value_type;

  enum {
    THREADS_PER_BLOCK = 128,
    ITEMS_PER_THREAD  = 16,
    ITEMS_PER_TILE    = ITEMS_PER_THREAD * THREADS_PER_BLOCK,
  };

  InputIterator d_input_begin;
  uint32_t num_items;
  uint32_t num_tiles;
  ScanOperator scan_op;
  SelectOperator select_op;
  scan_tile_state<uint32_t> count_state;
  scan_tile_state<T> item_state;

  scan_select_if_dispatch(InputIterator d_input_begin,  //
                          InputIterator d_input_end,
                          ScanOperator scan_op,
                          SelectOperator select_op,
                          cudaStream_t stream)
    : d_input_begin(d_input_begin),  //
      scan_op(scan_op),
      select_op(select_op)
  {
    num_items   = d_input_end - d_input_begin;
    num_tiles   = ceil_div(num_items, ITEMS_PER_TILE);
    count_state = scan_tile_state<uint32_t>(num_tiles);
    item_state  = scan_tile_state<T>(num_tiles);
  }

  void initialize(cudaStream_t stream)
  {
    auto num_init_blocks       = ceil_div(num_tiles, THREADS_PER_BLOCK);
    auto initialization_kernel = initialization_pass_kernel<T>;

    initialization_kernel<<<num_init_blocks, THREADS_PER_BLOCK, 0, stream>>>(  //
      count_state.state,
      item_state.state,
      num_tiles);

    CHECK_CUDA(stream);
  }

  void execute(OutputIterator d_output,
               OutputNumResultsIterator d_num_selected,
               select_scan_if_mode count_num_selected_only,
               cudaStream_t stream)
  {
    uint32_t max_blocks     = 1 << 15;
    auto num_tiles_per_pass = num_tiles < max_blocks ? num_tiles : max_blocks;
    auto execution_kernel   = execution_pass_kernel<  //
      T,
      InputIterator,
      OutputIterator,
      decltype(d_num_selected),
      ScanOperator,
      SelectOperator,
      THREADS_PER_BLOCK,
      ITEMS_PER_THREAD>;

    CHECK_CUDA(stream);

    for (uint32_t tile = 0; tile < num_tiles; tile += num_tiles_per_pass) {
      num_tiles_per_pass = std::min(num_tiles_per_pass, num_tiles - tile);
      execution_kernel<<<num_tiles_per_pass, THREADS_PER_BLOCK, 0, stream>>>(  //
        d_input_begin,
        d_output,
        d_num_selected,
        scan_op,
        select_op,
        num_items,
        num_tiles,
        tile,
        count_state.state,
        item_state.state,
        count_num_selected_only);

      CHECK_CUDA(stream);
    }
  }
};

/**
 * @brief
 *
 * @tparam InputIterator
 * @tparam ScanOperator
 * @tparam SelectOperator
 * @param d_input_begin
 * @param d_input_end
 * @param scan_op
 * @param select_op
 * @param stream
 * @param mr
 * @return rmm::device_vector<uint32_t>
 */
template <typename InputIterator, typename ScanOperator, typename SelectOperator>
auto scan_select_if(  //
  InputIterator d_input_begin,
  InputIterator d_input_end,
  ScanOperator scan_op,
  SelectOperator select_op,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto num_selected   = rmm::device_scalar<uint32_t>(stream);
  auto d_num_selected = num_selected.data();

  using Input          = typename std::iterator_traits<InputIterator>::value_type;
  using Output         = uint32_t;
  using OutputIterator = Output*;
  using Dispatch       = scan_select_if_dispatch<InputIterator,
                                           OutputIterator,
                                           decltype(d_num_selected),
                                           ScanOperator,
                                           SelectOperator>;

  auto dispatcher = Dispatch(d_input_begin, d_input_end, scan_op, select_op, stream);

  // initialize tile state and perform upsweep phase
  dispatcher.initialize(stream);
  dispatcher.execute(  //
    nullptr,
    d_num_selected,
    select_scan_if_mode::count_only,
    stream);

  CHECK_CUDA(stream);

  // // allocate result and perform downsweep
  auto d_output = rmm::device_vector<Output>(num_selected.value(stream));

  // perform downsweep phase
  dispatcher.execute(  //
    d_output.data().get(),
    d_num_selected,
    select_scan_if_mode::count_and_gather,
    stream);

  CHECK_CUDA(stream);

  return d_output;
}
