#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/transform_output_iterator.h>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#include <iterator>
#include <type_traits>

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

// ===== Agent =====================================================================================

template <typename Policy>
struct agent {
  typename Policy::InputIterator d_input;
  typename Policy::OutputCountIterator d_output_count;
  typename Policy::OutputIterator d_output;
  uint32_t num_items;
  typename Policy::ScanOperator scan_op;
  typename Policy::PredOperator pred_op;
  typename Policy::ItemsTileState& items_tile_state;
  typename Policy::IndexTileState& index_tile_state;

  inline __device__ void consume_range(bool const do_scatter,
                                       uint32_t const num_tiles,
                                       uint32_t const start_tile)
  {
    uint32_t const tile_idx            = start_tile + blockIdx.x;
    uint32_t const tile_offset         = Policy::ITEMS_PER_TILE * tile_idx;
    uint32_t const num_items_remaining = num_items - tile_offset;

    typename Policy::OutputCount num_selected;

    if (tile_idx < num_tiles - 1) {
      num_selected = consume_tile<false>(do_scatter, tile_idx, tile_offset, Policy::ITEMS_PER_TILE);
    } else {
      num_selected = consume_tile<true>(do_scatter, tile_idx, tile_offset, num_items_remaining);
      if (threadIdx.x == 0) { *d_output_count = num_selected; }
    }
  }

  template <bool IS_LAST_TILE>
  inline __device__ uint32_t consume_tile(bool const do_scatter,
                                          uint32_t const tile_idx,
                                          uint32_t const tile_offset,
                                          uint32_t const num_items_remaining)
  {
    typename Policy::Input items[Policy::ITEMS_PER_THREAD];

    __shared__ union {
      typename Policy::ItemsBlockLoad::TempStorage item_load;
      struct {
        typename Policy::IndexBlockScan::TempStorage index_scan;
        typename Policy::IndexPrefixCallback::TempStorage index_prefix;
      };
      struct {
        typename Policy::ItemsBlockScan::TempStorage items_scan;
        typename Policy::ItemsPrefixCallback::TempStorage items_prefix;
      };
    } temp_storage;

    // Load Inputs

    if (IS_LAST_TILE) {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items, num_items_remaining);
    } else {
      Policy::ItemsBlockLoad(temp_storage.item_load)  //
        .Load(d_input + tile_offset, items);
    }

    __syncthreads();

    // Scan Inputs Locally

    auto thread_aggregate = items[0];

    for (uint32_t i = 1; i < Policy::ITEMS_PER_THREAD; i++) {
      if (threadIdx.x * Policy::ITEMS_PER_THREAD + i < num_items_remaining) {
        thread_aggregate = scan_op(thread_aggregate, items[i]);
      };
    };

    // Scan Inputs

    if (tile_idx == 0) {
      typename Policy::Input block_aggregate;
      Policy::ItemsBlockScan(temp_storage.items_scan)  //
        .InclusiveScan(                                //
          items,
          items,
          scan_op,
          block_aggregate);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        items_tile_state.SetInclusive(0, block_aggregate);
      }
    } else {
      auto prefix_op = Policy::ItemsPrefixCallback(  //
        items_tile_state,
        temp_storage.items_prefix,
        scan_op,
        tile_idx);

      Policy::ItemsBlockScan(temp_storage.items_scan)  //
        .InclusiveScan(                                //
          items,
          items,
          scan_op,
          prefix_op);
    }

    __syncthreads();

    uint32_t selection_flags[Policy::ITEMS_PER_THREAD];

    // Initialize Selection Flags

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; i++) {
      selection_flags[i] = 0;
      if (threadIdx.x * Policy::ITEMS_PER_THREAD + i < num_items_remaining) {
        if (pred_op(items[i])) { selection_flags[i] = 1; }
      }
    }

    // Scan Selection

    uint32_t selection_indices[Policy::ITEMS_PER_THREAD];
    uint32_t num_selections;

    if (tile_idx == 0) {
      Policy::IndexBlockScan(temp_storage.index_scan)  //
        .ExclusiveScan(                                //
          selection_flags,
          selection_indices,
          cub::Sum(),
          num_selections);

      if (threadIdx.x == 0 and not IS_LAST_TILE) {
        index_tile_state.SetInclusive(0, num_selections);
      }
    } else {
      auto prefix_op = Policy::IndexPrefixCallback(  //
        index_tile_state,
        temp_storage.index_prefix,
        cub::Sum(),
        tile_idx);

      Policy::IndexBlockScan(temp_storage.index_scan)  //
        .ExclusiveScan(                                //
          selection_flags,
          selection_indices,
          cub::Sum(),
          prefix_op);

      num_selections = prefix_op.GetInclusivePrefix();
    }

    if (not do_scatter) { return num_selections; }

    // Scatter

    for (uint32_t i = 0; i < Policy::ITEMS_PER_THREAD; ++i) {
      if (selection_flags[i]) {
        if (selection_indices[i] < num_selections) {  //
          d_output[selection_indices[i]] =
            thrust::make_pair(tile_offset + threadIdx.x * Policy::ITEMS_PER_THREAD + i, items[i]);
        }
      }
    }

    return num_selections;
  }
};

// ===== Kernels ===================================================================================

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
  bool do_scatter,
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

  agent_instance.consume_range(do_scatter, num_tiles, start_tile);
}

// ===== Policy ====================================================================================

template <typename InputIterator_,
          typename OutputCountIterator_,
          typename OutputIterator_,
          typename ScanOperator_,
          typename PredicateOperator_>
struct policy {
  enum : uint32_t {
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

  // Items Load

  using ItemsBlockLoad = cub::BlockLoad<  //
    Input,
    THREADS_PER_BLOCK,
    ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT>;

  // Items Scan

  using ItemsTileState      = cub::ScanTileState<Input>;
  using ItemsPrefixCallback = cub::TilePrefixCallbackOp<Input, ScanOperator, ItemsTileState>;

  using ItemsBlockScan = cub::BlockScan<  //
    Input,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;

  // Index Scan

  using IndexTileState      = cub::ScanTileState<uint32_t>;
  using IndexPrefixCallback = cub::TilePrefixCallbackOp<uint32_t, cub::Sum, IndexTileState>;
  using IndexBlockScan      = cub::BlockScan<  //
    uint32_t,
    THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;
};

// ===== Entry =====================================================================================

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
  bool do_initialize,
  bool do_scatter,
  cudaStream_t stream = 0)
{
  CUDF_FUNC_RANGE();

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

  CUDA_TRY(items_tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
  CUDA_TRY(index_tile_state.Init(num_tiles, allocations[1], allocation_sizes[1]));

  if (do_initialize) {
    uint32_t num_init_blocks = ceil_div(num_tiles, Policy::THREADS_PER_INIT_BLOCK);

    auto init_kernel = initialization_pass_kernel<Policy>;
    init_kernel<<<num_init_blocks, Policy::THREADS_PER_INIT_BLOCK, 0, stream>>>(  //
      items_tile_state,
      index_tile_state,
      num_tiles);

    CHECK_CUDA(stream);
  }

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
      do_scatter,
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
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_FUNC_RANGE();
  using Input = typename InputIterator::value_type;

  auto output_projection = [] __device__(thrust::pair<uint32_t, Input> output) -> Input {
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
                 false,  // do_initialize
                 false,  // do_scatter
                 stream);

  auto d_temp_storage = rmm::device_buffer(temp_storage_bytes, stream);

  // phase 1 - determine number of results

  scan_select_if(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 OutputIterator(nullptr, output_projection),
                 d_in_end - d_in_begin,
                 scan_op,
                 pred_op,
                 true,   // do_initialize
                 false,  // do_scatter
                 stream);

  auto d_output = rmm::device_uvector<Input>(d_num_selections.value(stream), stream, mr);

  // phase 2 - gather results

  scan_select_if(d_temp_storage.data(),
                 temp_storage_bytes,
                 d_in_begin,
                 d_num_selections.data(),
                 OutputIterator(d_output.data(), output_projection),
                 d_in_end - d_in_begin,
                 scan_op,
                 pred_op,
                 false,  // do_initialize
                 true,   // do_scatter
                 stream);

  return d_output;
}
