#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/text/input_stream.hpp>
#include <cudf/io/text/superstate.hpp>
#include <cudf/io/text/trie.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#include <bitset>
#include <iostream>
#include <memory>

namespace {

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

using superstate = cudf::io::text::superstate<16>;

template <typename T>
struct scan_tile_state_view {
  bool* tile_status;
  T* tile_state;

  __device__ void initialize(cudf::size_type num_tiles)
  {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < num_tiles) { tile_status[thread_idx] = false; }
  }

  __device__ void set_state(cudf::size_type tile_idx, T value)
  {
    cub::ThreadStore<cub::STORE_CG>(tile_state + tile_idx, value);
    __threadfence();
    cub::ThreadStore<cub::STORE_CG>(tile_status + tile_idx, true);
  }

  __device__ T get_state_sync(cudf::size_type tile_idx)
  {
    while (cub::ThreadLoad<cub::LOAD_CG>(tile_status + tile_idx) == false) { __threadfence(); }
    return cub::ThreadLoad<cub::LOAD_CG>(tile_state + tile_idx);
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_uvector<bool> tile_status;
  rmm::device_uvector<T> tile_state;

  scan_tile_state(cudf::size_type num_tiles,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : tile_status(rmm::device_uvector<bool>(num_tiles + 1, stream, mr)),
      tile_state(rmm::device_uvector<T>(num_tiles + 1, stream, mr))

  {
  }

  operator scan_tile_state_view<T>()
  {
    return scan_tile_state_view<T>{tile_status.data(), tile_state.data()};
  }

  T back_element(rmm::cuda_stream_view s) const { return tile_state.back_element(s); }
};

// keep ITEMS_PER_TILE below input size to force multi-tile execution.
auto constexpr ITEMS_PER_THREAD = 32;
auto constexpr THREADS_PER_TILE = 512;
auto constexpr ITEMS_PER_TILE   = ITEMS_PER_THREAD * THREADS_PER_TILE;
auto constexpr TILES_PER_CHUNK  = 1024;
auto constexpr BYTES_PER_CHUNK  = ITEMS_PER_TILE * TILES_PER_CHUNK;
// multibyte_split works by splitting up inputs in to 32 inputs (bytes) per thread, and transforming
// them in to data structures called "superstates". these superstates are created by searching a
// trie, but instead of a tradition trie where the search begins at a single node at the beginning,
// we allow our search to begin anywhere within the trie tree. The position within the trie tree is
// stored as a "partial match path", which indicates "we can get from here to there by a set of
// specific transitions". By scanning together superstates, we effectively know "we can get here
// from the beginning by following the inputs". By doing this, each thread knows exactly what state
// it begins in. From there, each thread can then take deterministic action. In this case, the
// deterministic action is counting and outputting delimiter offsets when a delimiter is found.

struct PatternScan {
  typedef cub::BlockScan<superstate, THREADS_PER_TILE> BlockScan;

  struct _TempStorage {
    typename BlockScan::TempStorage scan;
    superstate block_aggregate;
    superstate exclusive_prefix;
    superstate inclusive_prefix;
  };

  _TempStorage& _temp_storage;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline PatternScan(TempStorage& temp_storage) : _temp_storage(temp_storage.Alias()) {}

  __device__ inline void Scan(scan_tile_state_view<superstate> tile_state,
                              cudf::io::text::trie_device_view trie,
                              char (&thread_data)[ITEMS_PER_THREAD],
                              uint32_t (&thread_state)[ITEMS_PER_THREAD])
  {
    // create a state that represents all possible starting states.
    auto thread_superstate = superstate();

    // transition all possible states
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      thread_superstate = thread_superstate.apply([&](uint8_t state) {  //
        return trie.transition(state, thread_data[i]);
      });
    }

    auto prefix_callback = [&] __device__(superstate const& block_aggregate) -> superstate {
      if (threadIdx.x == 0) {
        _temp_storage.block_aggregate  = block_aggregate;
        _temp_storage.exclusive_prefix = tile_state.get_state_sync(blockIdx.x);
        _temp_storage.inclusive_prefix = _temp_storage.exclusive_prefix + block_aggregate;
        tile_state.set_state(blockIdx.x + 1, _temp_storage.inclusive_prefix);
      }
      return _temp_storage.exclusive_prefix;
    };

    BlockScan(_temp_storage.scan)
      .ExclusiveSum(thread_superstate, thread_superstate, prefix_callback);

    // transition from known state to known state
    thread_state[0] = trie.transition(thread_superstate.get(0), thread_data[0]);

    for (uint32_t i = 1; i < ITEMS_PER_THREAD; i++) {
      thread_state[i] = trie.transition(thread_state[i - 1], thread_data[i]);
    }
  }
};

__global__ void multibyte_split_init_kernel(cudf::size_type num_tiles,
                                            scan_tile_state_view<superstate> tile_superstates,
                                            scan_tile_state_view<uint32_t> tile_output_offsets)
{
  tile_superstates.initialize(num_tiles);
  tile_superstates.set_state(0, superstate());
  tile_output_offsets.initialize(num_tiles);
  tile_output_offsets.set_state(0, 0);
}

__global__ void multibyte_split_kernel(cudf::size_type num_tiles,
                                       scan_tile_state_view<superstate> tile_superstates,
                                       scan_tile_state_view<uint32_t> tile_output_offsets,
                                       cudf::io::text::trie_device_view trie,
                                       cudf::device_span<char const> data,
                                       cudf::device_span<int32_t> string_offsets)
{
  typedef cub::BlockScan<uint32_t, THREADS_PER_TILE> OffsetScan;

  __shared__ union {
    typename PatternScan::TempStorage pattern_scan;
    struct {
      typename OffsetScan::TempStorage offset_scan;
      uint32_t offset_scan_exclusive_prefix;
    };
  } temp_storage;

  int32_t const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t const data_begin = thread_idx * ITEMS_PER_THREAD;
  int32_t const num_valid  = data.size() - data_begin;

  // STEP 1: Load inputs

  char thread_data[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {  //
    thread_data[i] = data[data_begin + i];
  }

  // STEP 2: Scan inputs to determine absolute thread states

  uint32_t thread_states[ITEMS_PER_THREAD];

  PatternScan(temp_storage.pattern_scan)  //
    .Scan(tile_superstates, trie, thread_data, thread_states);

  // STEP 3: Flag matches

  uint32_t thread_offsets[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_offsets[i] = i < num_valid and trie.is_match(thread_states[i]);
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  __syncthreads();  // required before temp_memory re-use

  auto prefix_callback = [&] __device__(uint32_t const& block_aggregate) -> uint32_t {
    if (threadIdx.x == 0) {
      temp_storage.offset_scan_exclusive_prefix = tile_output_offsets.get_state_sync(blockIdx.x);
      auto inclusive_prefix = temp_storage.offset_scan_exclusive_prefix + block_aggregate;
      tile_output_offsets.set_state(blockIdx.x + 1, inclusive_prefix);
    }
    return temp_storage.offset_scan_exclusive_prefix;
  };

  OffsetScan(temp_storage.offset_scan)
    .ExclusiveSum(thread_offsets, thread_offsets, prefix_callback);

  // Step 5: Assign string_offsets from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {
    auto const match_length = trie.get_match_length(thread_states[i]);

    if (match_length == 0) { continue; }

    auto const match_end   = data_begin + i + 1;
    auto const match_begin = match_end - match_length;

    printf("bid(%2u) tid(%2u) byte(%2u): %c %2u - [%3u, %3u)\n",  //
           blockIdx.x,
           threadIdx.x,
           i,
           thread_data[i],
           thread_offsets[i],
           match_begin,
           match_end);

    if (string_offsets.size() > thread_offsets[i]) {  //
      string_offsets[thread_offsets[i]] = match_end;
    }
  }
}

}  // namespace

namespace cudf {
namespace io {
namespace text {
namespace detail {

std::unique_ptr<cudf::column> multibyte_split(cudf::string_scalar const& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto const trie = cudf::io::text::trie::create(delimeters, stream);

  auto num_tiles = ceil_div(input.size(), ITEMS_PER_TILE);

  // pattern-match and count delimiters

  auto tile_superstates = scan_tile_state<superstate<16>>(num_tiles, stream);
  auto tile_offsets     = scan_tile_state<uint32_t>(num_tiles, stream);
  auto num_init_blocks  = ceil_div(num_tiles, THREADS_PER_TILE);

  multibyte_split_init_kernel<<<num_init_blocks, THREADS_PER_TILE, 0, stream.value()>>>(  //
    num_tiles,
    tile_superstates,
    tile_offsets);

  multibyte_split_kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    num_tiles,
    tile_superstates,
    tile_offsets,
    trie.view(),
    cudf::device_span<char const>(input.data(), input.size()),
    cudf::device_span<cudf::size_type>(static_cast<size_type*>(nullptr), 0));

  // allocate string offsets

  auto num_results    = tile_offsets.back_element(stream);
  auto string_offsets = rmm::device_uvector<cudf::size_type>(num_results + 2, stream);
  auto const x        = string_offsets.size() - 1;
  auto const y        = input.size();

  // first and last element are set manually to zero and size of input, respectively.
  // kernel is only responsible for determining delimiter offsets
  string_offsets.set_element_to_zero_async(0, stream);
  string_offsets.set_element_async(x, y, stream);

  // pattern-match and materialize string offsets

  multibyte_split_kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    num_tiles,
    tile_superstates,
    tile_offsets,
    trie.view(),
    cudf::device_span<char const>(input.data(), input.size()),
    cudf::device_span<cudf::size_type>(string_offsets).subspan(1, num_results));

  return cudf::make_strings_column(  //
    cudf::device_span<char const>(input.data(), input.size()),
    string_offsets,
    {},
    0,
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::string_scalar const& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeters, rmm::cuda_stream_default, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
