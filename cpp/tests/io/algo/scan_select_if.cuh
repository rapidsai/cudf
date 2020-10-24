#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/transform_output_iterator.h>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>

#include <iterator>
#include <type_traits>

enum class select_scan_if_mode { count_only, count_and_gather, gather_only };

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

template <typename T, int N>
inline __device__ void print_array_16(  //
  char const* message,
  uint32_t num_items_remaining,
  T (&values)[N])
{
  printf("b(%i) t(%i) %s (%i): %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n",
         blockIdx.x,
         threadIdx.x,
         message,
         num_items_remaining,
         values[0],
         values[1],
         values[2],
         values[3],
         values[4],
         values[5],
         values[6],
         values[7],
         values[8],
         values[9],
         values[10],
         values[11],
         values[12],
         values[13],
         values[14],
         values[15]);
}

template <typename Policy>
struct agent {
  typename Policy::InputIterator d_input;
  typename Policy::OutputCountIterator d_output_count;
  typename Policy::OutputIterator d_output;
  uint32_t num_items;
  typename Policy::ScanOperator scan_op;
  typename Policy::PredOperator pred_op;
  typename Policy::ItemsTileState items_tile_state;
  typename Policy::IndexTileState index_tile_state;

  inline __device__ void consume_range(uint32_t num_tiles, uint32_t start_tile)
  {
    uint32_t tile_idx            = start_tile + blockIdx.x;
    uint32_t tile_offset         = Policy::ITEMS_PER_TILE * tile_idx;
    uint32_t num_items_remaining = num_items - tile_offset;

    typename Policy::OutputCount num_selected;

    // if (num_items_remaining <= 0) { return; }

    if (threadIdx.x == 0) {
      printf(
        "b(%i) t(%i) entry with tile_idx(%i), tile_offset(%i), and %i of %i items remaining.\n",
        blockIdx.x,
        threadIdx.x,
        tile_idx,
        tile_offset,
        num_items_remaining,
        num_items);
    }

    if (tile_idx < num_tiles - 1) {
      num_selected = consume_tile<false>(tile_idx, tile_offset, num_items_remaining);
    } else {
      num_selected = consume_tile<true>(tile_idx, tile_offset, num_items_remaining);
      if (threadIdx.x == 0) {  //
        // is there a race condition somewhere?
        printf("b(%i) t(%i) num selected: %i\n", blockIdx.x, threadIdx.x, num_selected);
        *d_output_count = num_selected;
      }
    }
  }

  template <bool IS_LAST_TILE>
  inline __device__ uint32_t consume_tile(uint32_t tile_idx,
                                          uint32_t tile_offset,
                                          uint32_t num_items_remaining)
  {
    typename Policy::Input items[Policy::ITEMS_PER_THREAD];

    __shared__ typename Policy::ItemsBlockLoad::TempStorage item_load;
    __shared__ typename Policy::IndexBlockScan::TempStorage index_scan;
    __shared__ typename Policy::IndexPrefixCallback::TempStorage index_prefix;
    __shared__ typename Policy::ItemsBlockScan::TempStorage items_scan;
    __shared__ typename Policy::ItemsPrefixCallback::TempStorage items_prefix;

    __syncthreads();

    if (IS_LAST_TILE) {
      Policy::ItemsBlockLoad(item_load).Load(d_input + tile_offset, items, num_items_remaining);
    } else {
      Policy::ItemsBlockLoad(item_load).Load(d_input + tile_offset, items);
    }

    __syncthreads();

    // Scan Inputs

    if (tile_idx == 0) {
      typename Policy::Input block_aggregate;
      Policy::ItemsBlockScan(items_scan).InclusiveScan(items, items, scan_op, block_aggregate);
      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        items_tile_state.SetInclusive(0, block_aggregate);
      }
    } else {
      auto prefix_op =
        Policy::ItemsPrefixCallback(items_tile_state, items_prefix, scan_op, tile_idx);
      Policy::ItemsBlockScan(items_scan).InclusiveScan(items, items, scan_op, prefix_op);
    }

    __syncthreads();

    // print_array_16("items", num_items_remaining, items);

    // __syncthreads();

    uint32_t selection_flags[Policy::ITEMS_PER_THREAD];

    // Initialize Selection Flags

    for (uint64_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      selection_flags[i] = 0;
      // if (threadIdx.x * Policy::ITEMS_PER_THREAD + i < num_items_remaining) {
      //   if (pred_op(items[i])) { selection_flags[i] = 1; }
      // }
    }

    // Scan Selection

    __syncthreads();

    // print_array_16("flags", num_items_remaining, selection_flags);

    // __syncthreads();

    uint32_t selection_indices[Policy::ITEMS_PER_THREAD] = {};
    uint32_t num_selections;

    // for (uint64_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
    //   selection_indices[i] = 0;
    //   // if (threadIdx.x * Policy::ITEMS_PER_THREAD + i < num_items_remaining) {
    //   //   if (pred_op(items[i])) { selection_flags[i] = 1; }
    //   // }
    // }

    print_array_16("indices", num_items_remaining, selection_indices);

    if (tile_idx == 0) {
      Policy::IndexBlockScan(index_scan)
        .ExclusiveScan(selection_flags, selection_indices, cub::Sum(), num_selections);
      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        index_tile_state.SetInclusive(0, num_selections);
      }
    } else {
      auto prefix_op =
        Policy::IndexPrefixCallback(index_tile_state, index_prefix, cub::Sum(), tile_idx);
      Policy::IndexBlockScan(index_scan)
        .ExclusiveScan(selection_flags, selection_indices, cub::Sum(), prefix_op);
      num_selections = prefix_op.GetInclusivePrefix();
    }

    __syncthreads();

    // __syncthreads();

    if (threadIdx.x == 0 and num_selections != 0) {
      printf("b(%i) t(%i) selected: %i\n", blockIdx.x, threadIdx.x, num_selections);
    }

    // return if unexpected number of selections (consumer doesn't know how to store values).

    if (*d_output_count == 0) {  // best way to communicate this?
      return num_selections;
    }

    // Scatter

    for (int i = 0; i < Policy::ITEMS_PER_THREAD; ++i) {
      if (selection_flags[i]) {
        if (selection_indices[i] < num_selections) {  //
          d_output[selection_indices[i]] =
            thrust::make_pair(tile_offset + threadIdx.x * Policy::ITEMS_PER_THREAD + i, items[i]);
        } else {
          printf("b(%i) t(%i) %i = %i\n", blockIdx.x, threadIdx.x, selection_indices[i], items[i]);
        }
      }
    }
    __syncthreads();

    return num_selections;
  }
};

// ===== KERNELS ===================================================================================

template <typename Policy>
__global__ void initialization_pass_kernel(  //
  typename Policy::ItemsTileState items_state,
  typename Policy::IndexTileState index_state,
  uint32_t num_tiles)
{
  items_state.InitializeStatus(num_tiles);
  index_state.InitializeStatus(num_tiles);
}

template <typename Policy>
__global__ void execution_pass_kernel(  //
  typename Policy::InputIterator d_input,
  typename Policy::OutputCountIterator d_output_count,
  typename Policy::OutputIterator d_output,
  uint32_t num_items,
  typename Policy::ScanOperator scan_op,
  typename Policy::PredOperator pred_op,
  typename Policy::ItemsTileState items_tile_state,
  typename Policy::IndexTileState index_tile_state,
  uint32_t num_tiles,
  uint32_t start_tile)
{
  auto agent_instance = agent<Policy>{
    d_input,
    d_output_count,
    d_output,
    num_items,
    scan_op,
    pred_op,
    items_tile_state,
    index_tile_state  //
  };

  agent_instance.consume_range(num_tiles, start_tile);
}

// ===== POLICY ====================================================================================

template <typename InputIterator_,
          typename OutputCountIterator_,
          typename OutputIterator_,
          typename ScanOperator_,
          typename PredicateOperator_>
struct policy {
  enum {
    THREADS_PER_INIT_BLOCK = 128,
    THREADS_PER_BLOCK      = 128,
    ITEMS_PER_THREAD       = 16,
    ITEMS_PER_TILE         = ITEMS_PER_THREAD * THREADS_PER_BLOCK,
  };

  using InputIterator       = InputIterator_;
  using OutputIterator      = OutputIterator_;
  using OutputCountIterator = OutputCountIterator_;

  using ScanOperator = ScanOperator_;
  using PredOperator = PredicateOperator_;

  using Offset      = typename std::iterator_traits<InputIterator>::difference_type;
  using Input       = typename std::iterator_traits<InputIterator>::value_type;
  using OutputCount = typename std::iterator_traits<OutputCountIterator>::value_type;
  using OutputValue = typename std::iterator_traits<OutputIterator>::value_type;

  // Item Load and Scan

  using ItemsTileState      = cub::ScanTileState<Input>;
  using ItemsPrefixCallback = cub::TilePrefixCallbackOp<  //
    Input,
    ScanOperator,
    cub::ScanTileState<Input>>;

  using ItemsBlockLoad = cub::BlockLoad<  //
    Input,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  using ItemsBlockScan = cub::BlockScan<  //
    Input,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;

  // Index Scan

  using IndexTileState      = cub::ScanTileState<uint32_t>;
  using IndexPrefixCallback = cub::TilePrefixCallbackOp<  //
    uint32_t,
    cub::Sum,
    cub::ScanTileState<uint32_t>>;

  using IndexBlockScan = cub::BlockScan<  //
    uint32_t,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>;
};

// ===== ENTRY =====================================================================================

template <typename InputIterator,
          typename OutputCountIterator,
          typename OutputIterator,
          typename ScanOperator,
          typename PredOperator>
void scan_select_if(  //
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIterator d_in,
  OutputCountIterator d_count_out,
  OutputIterator d_out,
  uint32_t num_items,
  ScanOperator scan_op,
  PredOperator pred_op,
  cudaStream_t stream = 0)
{
  using Policy =
    policy<InputIterator, OutputCountIterator, OutputIterator, ScanOperator, PredOperator>;

  uint32_t num_tiles = ceil_div(num_items, Policy::ITEMS_PER_TILE);

  // calculate temp storage requirements

  void* allocations[2];
  size_t allocation_sizes[2];

  CUDA_TRY(Policy::ItemsTileState::AllocationSize(num_tiles, allocation_sizes[0]));
  CUDA_TRY(Policy::IndexTileState::AllocationSize(num_tiles, allocation_sizes[1]));

  CUDA_TRY(
    cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));

  if (d_temp_storage == nullptr) { return; }

  // initialize

  typename Policy::ItemsTileState items_tile_state;
  typename Policy::IndexTileState index_tile_state;

  CUDA_TRY(items_tile_state.Init(num_tiles, d_temp_storage, allocation_sizes[0]));
  CUDA_TRY(index_tile_state.Init(num_tiles, d_temp_storage, allocation_sizes[1]));

  uint32_t num_init_blocks = ceil_div(num_tiles, Policy::THREADS_PER_INIT_BLOCK);

  auto init_kernel = initialization_pass_kernel<Policy>;
  init_kernel<<<num_init_blocks, Policy::THREADS_PER_INIT_BLOCK, 0, stream>>>(  //
    items_tile_state,
    index_tile_state,
    num_tiles);

  CHECK_CUDA(stream);

  // execute

  auto exec_kernel = execution_pass_kernel<Policy>;

  uint32_t tiles_per_pass = 1 << 10;

  for (uint32_t start_tile = 0; start_tile < num_tiles; start_tile += tiles_per_pass) {
    tiles_per_pass = std::min(tiles_per_pass, num_tiles - start_tile);

    exec_kernel<<<tiles_per_pass, Policy::THREADS_PER_BLOCK, 0, stream>>>(  //
      d_in,
      d_count_out,
      d_out,
      num_items,
      scan_op,
      pred_op,
      items_tile_state,
      index_tile_state,
      num_tiles,
      start_tile);

    CHECK_CUDA(stream);
  }
}

template <typename InputIterator,
          typename ScanOperator,
          typename PredOperator>
rmm::device_uvector<typename InputIterator::value_type>  //
scan_select_if(                                          //
  InputIterator d_in_begin,
  InputIterator d_in_end,
  ScanOperator scan_op,
  PredOperator pred_op,
  cudaStream_t stream = 0)
{
  using Input = typename InputIterator::value_type;

  auto output_projection = [] __device__(thrust::pair<uint32_t, Input> output) -> Input {
    printf("b(%i) t(%i) (%i) (%i)\n", blockIdx.x, threadIdx.x, output.first, output.second);
    return thrust::get<1>(output);
  };

  using OutputIterator = thrust::transform_output_iterator<decltype(output_projection), Input*>;

  auto d_num_selections = rmm::device_scalar<uint32_t>(0, stream);

  uint64_t temp_storage_bytes;

  // query required temp storage (does not launch kernel)

  scan_select_if(nullptr,
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 OutputIterator(nullptr, output_projection),
                 d_in_end - d_in_begin,
                 scan_op,
                 pred_op,
                 stream);

  auto d_temp_storage = rmm::device_buffer(temp_storage_bytes, stream);

  // phase 1 - determine number of resultsd_temp_storage

  scan_select_if(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 OutputIterator(nullptr, output_projection),
                 d_in_end - d_in_begin,
                 scan_op,
                 pred_op,
                 stream);

  auto d_output = rmm::device_uvector<Input>(d_num_selections.value(stream), stream);

  // phase 2 - gather results

  scan_select_if(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 OutputIterator(d_output.data(), output_projection),
                 d_in_end - d_in_begin,
                 scan_op,
                 pred_op,
                 stream);

  cudaStreamSynchronize(stream);

  return d_output;
}
