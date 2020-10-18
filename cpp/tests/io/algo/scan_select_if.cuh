
#include <cudf/utilities/error.hpp>

#include <__clang_cuda_device_functions.h>
#include <rmm/thrust_rmm_allocator.h>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <iterator>
#include "cudf/types.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/thrust_rmm_allocator.h"

#include <type_traits>

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

/**
 * @brief
 *
 */
**@tparam Policy* @tparam InIter* @tparam OutIter* @tparam ScanOp* @tparam PredOp* /
  template <typename Policy>
  struct scan_select_if_agent {
  using TileStateView = typename Policy::TileStateView;
  using InIter        = typename Policy::InIter;
  using OutIter       = typename Policy::OutIter;
  using ScanOp        = typename Policy::ScanOp;
  using PredOp        = typename Policy::PredOp;
  using Output        = typename Policy::Output;

  using BlockLoad = cub::BlockLoad<  //
    Output,
    Policy::THREADS_PER_BLOCK,
    Policy::ITEMS_PER_THREAD,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;

  using BlockScan = cub::BlockScan<  //
    Output,
    Policy::THREADS_PER_BLOCK,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS>;

  union TempStorage {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
  };

  TempStorage temp_storage;
  InIter d_input;
  OutIter d_output;
  ScanOp scan_op;
  PredOp pred_op;

  inline __device__ scan_select_if_agent(  //
    TempStorage temp_storage,
    InIter d_input,
    OutIter d_output,
    ScanOp scan_op,
    PredOp pred_op)
    : temp_storage(temp_storage),
      d_input(d_input),
      d_output(d_output),
      scan_op(scan_op),
      pred_op(pred_op)
  {
  }

  template <bool is_last_tile>
  inline __device__ void consume_tile(TileStateView& tile_state,
                                      uint32_t tile_idx,
                                      uint32_t tile_offset,
                                      uint32_t num_remaining)
  {
    Output items[Policy::ITEMS_PER_THREAD];
    uint32_t selection_flags[Policy::ITEMS_PER_THREAD];
    uint32_t selection_indices[Policy::ITEMS_PER_THREAD];

    if (is_last_tile) {
      BlockLoad(temp_storage.load).Load(d_input + tile_offset, items);
    } else {
      BlockLoad(temp_storage.load).Load(d_input + tile_offset, items, num_remaining);
    }

    __syncthreads();

    if (blockIdx.x == 0) {
      typename Policy::Output block_aggregate;
      BlockScan(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
      tile_state.set_inclusive();
    } else {
      TilePrefixCallbackOp < Policy::typename Policy::PrefixCallbackOp prefix_op(
                               tile_state, temp_storage.prefix, scan_op, tile_idx);
      BlockScan(temp_storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
    }

    __syncthreads();

    // must scan values first, then scan for selection

    scatter<is_last_tile>(items, selection_flags, selection_flags);
  }

  inline __device__ void consume_range(TileStateView& tile_state, int num_items, int start_tile)
  {
    auto tile_idx          = start_tile + blockIdx.x;
    uint32_t tile_offset   = Policy::ITEMS_PER_TILE * tile_idx;
    uint32_t num_remaining = num_items - tile_offset;

    const bool is_last_tile = num_remaining < Policy::ITEMS_PER_TILE;

    if (not is_last_tile) {
      consume_tile<false>(tile_state, tile_idx, tile_offset, num_remaining);
    } else {
      consume_tile<true>(tile_state, tile_idx, tile_offset, num_remaining);
    }
  }

  template <bool is_last_tile>
  inline __device__ void scatter(  //
    Output (&items)[Policy::ITEMS_PER_THREAD],
    uint32_t (&select_flags)[Policy::ITEMS_PER_THREAD],
    uint32_t (&selection_indices)[Policy::ITEMS_PER_THREAD],
    uint32_t num_selections)
  {
    for (uint32_t i; i < Policy::ITEMS_PER_THREAD; i++) {
      if (select_flags[i]) {
        if (not is_last_tile || selection_indices[i] < num_selections) {
          d_output[selection_indices[i]] = items[i];
        }
      }
    }
  }
}

/**
 * @brief
 *
 */
enum class tile_status : uint8_t {
  INVALID,
  PARTIAL,
  INCLUSIVE
};

/**
 * @brief
 *
 * @tparam T
 */
template <typename T>
struct tile_state_view {
  uint32_t num_tiles;
  tile_status* d_tile_status;
  T* d_tile_partial;
  T* d_tile_inclusive;

  tile_state_view(uint32_t num_tiles,
                  tile_status* d_tile_status,
                  T* d_tile_partial,
                  T* d_tile_inclusive)
    : num_tiles(num_tiles),
      d_tile_status(d_tile_status),
      d_tile_partial(d_tile_partial),
      d_tile_inclusive(d_tile_inclusive)
  {
  }

  inline __device__ void initialize_status()
  {
    uint32_t tile_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tile_idx < num_tiles) { d_tile_status[tile_idx] = tile_status::INVALID; }
  }

  inline __device__ void set_partial(uint32_t tile, T prefix)
  {
    d_tile_partial[tile] = prefix;
    __threadfence();
    d_tile_status[tile] = tile_status::PARTIAL;
  }

  inline __device__ void set_inclusive(uint32_t tile, T prefix)
  {
    d_tile_inclusive[tile] = prefix;
    __threadfence();
    d_tile_status[tile] = tile_status::INCLUSIVE;
  }

  inline __device__ void wait_for_valid(uint32_t tile, tile_status& status, T& value)
  {
    do {
      status = d_tile_status[tile];
      __threadfence();
    } while (status == tile_status::INVALID);

    if (status == tile_status::PARTIAL) {
      value = d_tile_partial[tile];
    } else {
      value = d_tile_inclusive[tile];
    }
  }
};

/**
 * @brief
 *
 * @tparam T
 */
template <typename T>
struct tile_state {
  uint32_t num_tiles;
  rmm::device_uvector<tile_status> d_tile_status;
  rmm::device_uvector<T> d_tile_partial;
  rmm::device_uvector<T> d_tile_inclusive;

  tile_state(uint32_t num_tiles, cudaStream_t stream)
    : num_tiles(num_tiles),
      d_tile_status(rmm::device_uvector<tile_status>(num_tiles, stream)),
      d_tile_inclusive(rmm::device_uvector<T>(num_tiles, stream)),
      d_tile_partial(rmm::device_uvector<T>(num_tiles, stream))
  {
  }

  tile_state_view<T> view()
  {
    return {num_tiles,  //
            d_tile_status.data(),
            d_tile_partial.data(),
            d_tile_inclusive.data()};
  }
};

/**
 * @brief
 *
 * @tparam Policy
 * @param tile_state
 */
template <typename Policy>
__global__ void initialization_pass_kernel(  //
  tile_state_view<typename Policy::Output> tile_state)
{
  tile_state.initialize_status();
}

/**
 * @brief
 *
 * @tparam Policy
 * @param tile_state
 * @param d_input
 * @param d_output
 * @param scan_op
 * @param pred_op
 * @param num_items
 * @param start_tile
 */
template <typename Policy>
__global__ void execution_pass_kernel(  //
  typename Policy::TileStateView d_tile_state_view,
  typename Policy::InIter d_input,
  typename Policy::OutIter d_output,
  typename Policy::ScanOp scan_op,
  typename Policy::PredOp pred_op,
  uint32_t num_items,
  uint32_t start_tile)
{
  using Agent = scan_select_if_agent<Policy>;

  __shared__ typename Agent::TempStorage temp_storage;

  Agent(temp_storage, d_input, d_output, scan_op, pred_op)
    .consume_range(  //
      d_tile_state_view,
      num_items,
      start_tile);
}

/**
 * @brief
 *
 * @tparam Policy
 */
template <typename Policy>
struct scan_select_if_dispatch {
  using InIter    = typename Policy::InIter;
  using OutIter   = typename Policy::OutIter;
  using Output    = typename Policy::Output;
  using ScanOp    = typename Policy::ScanOp;
  using PredOp    = typename Policy::PredOp;
  using TileState = typename Policy::TileState;

  InIter d_input_begin;
  TileState d_tile_state;
  uint32_t num_items;
  uint32_t num_tiles;
  uint32_t temp_storage_bytes;
  ScanOp scan_op;
  PredOp pred_op;

  using Input = typename std::iterator_traits<InIter>::value_type;

  scan_select_if_dispatch(InIter d_input_begin,  //
                          InIter d_input_end,
                          ScanOp scan_op,
                          PredOp pred_op,
                          cudaStream_t stream)
    : d_input_begin(d_input_begin),  //
      scan_op(scan_op),
      pred_op(pred_op),
      num_items(d_input_end - d_input_begin),
      num_tiles(ceil_div(num_items, Policy::ITEMS_PER_TILE)),
      d_tile_state(TileState(d_input_end - d_input_begin, stream))
  {
  }

  void initialize(cudaStream_t stream)
  {
    auto num_init_blocks       = ceil_div(num_tiles, Policy::THREADS_PER_BLOCK);
    auto initialization_kernel = initialization_pass_kernel<Policy>;

    initialization_kernel<<<num_init_blocks, Policy::THREADS_PER_BLOCK, 0, stream>>>(  //
      d_tile_state.view());

    CHECK_CUDA(stream);
  }

  void execute(OutIter d_output, uint32_t& num_results, cudaStream_t stream)
  {
    uint32_t max_blocks     = 1 << 15;
    auto num_tiles_per_pass = num_tiles < max_blocks ? num_tiles : max_blocks;
    auto execution_kernel   = execution_pass_kernel<Policy>;

    for (uint32_t tile = 0; tile < num_tiles; tile += num_tiles_per_pass) {
      execution_kernel<<<num_tiles_per_pass, Policy::THREADS_PER_BLOCK, 0, stream>>>(  //
        d_tile_state.view(),
        d_input_begin,
        d_output,
        scan_op,
        pred_op,
        num_items,
        tile);

      CHECK_CUDA(stream);
    }
  }
};

/**
 * @brief
 *
 * @tparam InIter_
 * @tparam OutIter_
 * @tparam ScanOp_
 * @tparam PredOp_
 */
template <typename InIter_, typename OutIter_, typename ScanOp_, typename PredOp_>
struct scan_select_if_policy {
  using InIter        = InIter_;
  using OutIter       = OutIter_;
  using ScanOp        = ScanOp_;
  using PredOp        = PredOp_;
  using Input         = typename std::iterator_traits<InIter>::value_type;
  using Output        = typename std::iterator_traits<OutIter>::value_type;
  using TileState     = tile_state<Output>;
  using TileStateView = tile_state_view<Output>;

  enum : uint32_t {
    THREADS_PER_BLOCK = 128,
    ITEMS_PER_THREAD  = 16,
    ITEMS_PER_TILE    = THREADS_PER_BLOCK * ITEMS_PER_THREAD,
  };
};

/**
 * @brief
 *
 * @tparam InIter
 * @tparam ScanOp
 * @tparam PredOp
 * @param d_input_begin
 * @param d_input_end
 * @param scan_op
 * @param pred_op
 * @param stream
 * @param mr
 * @return rmm::device_vector<uint32_t>
 */
template <typename InIter, typename ScanOp, typename PredOp>
auto scan_select_if(  //
  InIter d_input_begin,
  InIter d_input_end,
  ScanOp scan_op,
  PredOp pred_op,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  using Output   = uint32_t;
  using OutIter  = Output*;
  using Policy   = scan_select_if_policy<InIter, OutIter, ScanOp, PredOp>;
  using Dispatch = scan_select_if_dispatch<Policy>;

  auto dispatcher = Dispatch(d_input_begin, d_input_end, scan_op, pred_op, stream);

  // initialize tile state and perform upsweep
  uint32_t num_results = 0;
  dispatcher.initialize(stream);
  dispatcher.execute(nullptr, num_results, stream);

  // allocate result and perform downsweep
  auto d_output = rmm::device_uvector<Output>(num_results, stream, mr);  // TODO: use mr
  dispatcher.execute(d_output.data(), num_results, stream);

  return d_output;
}
