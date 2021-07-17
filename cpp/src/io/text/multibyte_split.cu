#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/text/device_istream.hpp>
#include <cudf/io/text/superstate.hpp>
#include <cudf/io/text/trie.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>

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

enum class scan_tile_status : uint8_t {
  oob,
  invalid,
  partial,
  inclusive,
};

template <typename T>
struct scan_tile_state_view {
  uint64_t num_tiles;
  scan_tile_status* tile_status;
  T* tile_partial;
  T* tile_inclusive;

  __device__ inline void initialize_status(cudf::size_type base_tile_idx,
                                           cudf::size_type count,
                                           scan_tile_status status)
  {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < count) {  //
      tile_status[(base_tile_idx + thread_idx) % num_tiles] = status;
    }
  }

  __device__ inline void set_partial_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_partial + offset, value);
    __threadfence();
    cub::ThreadStore<cub::STORE_CG>(tile_status + offset, scan_tile_status::partial);
  }

  __device__ inline void set_inclusive_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_inclusive + offset, value);
    __threadfence();
    cub::ThreadStore<cub::STORE_CG>(tile_status + offset, scan_tile_status::inclusive);
  }

  __device__ inline T get_prefix(cudf::size_type tile_idx, scan_tile_status& status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;

    while ((status = cub::ThreadLoad<cub::LOAD_CG>(tile_status + offset)) ==
           scan_tile_status::invalid) {
      __threadfence();
    }

    if (status == scan_tile_status::partial) {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_partial + offset);
    } else {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_inclusive + offset);
    }
  }

  __device__ inline T get_inclusive_prefix(cudf::size_type tile_idx)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    while (cub::ThreadLoad<cub::LOAD_CG>(tile_status + offset) != scan_tile_status::inclusive) {
      __threadfence();
    }
    return cub::ThreadLoad<cub::LOAD_CG>(tile_inclusive + offset);
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_uvector<scan_tile_status> tile_status;
  rmm::device_uvector<T> tile_state_partial;
  rmm::device_uvector<T> tile_state_inclusive;

  scan_tile_state(cudf::size_type num_tiles,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : tile_status(rmm::device_uvector<scan_tile_status>(num_tiles, stream, mr)),
      tile_state_partial(rmm::device_uvector<T>(num_tiles, stream, mr)),
      tile_state_inclusive(rmm::device_uvector<T>(num_tiles, stream, mr))
  {
  }

  operator scan_tile_state_view<T>()
  {
    return scan_tile_state_view<T>{tile_status.size(),
                                   tile_status.data(),
                                   tile_state_partial.data(),
                                   tile_state_inclusive.data()};
  }

  inline void set_seed_async(T const seed, rmm::cuda_stream_view stream)
  {
    auto x = tile_status.size();
    auto y = scan_tile_status::inclusive;
    tile_state_inclusive.set_element_async(x - 1, seed, stream);
    tile_status.set_element_async(x - 1, y, stream);
  }

  // T back_element(rmm::cuda_stream_view stream) const { return tile_state.back_element(stream); }

  inline T get_inclusive_prefix(cudf::size_type tile_idx, rmm::cuda_stream_view stream) const
  {
    auto const offset = (tile_idx + tile_status.size()) % tile_status.size();
    return tile_state_inclusive.element(offset, stream);
  }
};

auto constexpr PARTIAL_AGGRIGATION_STRATEGY = 2;

// keep ITEMS_PER_TILE below input size to force multi-tile execution.
auto constexpr ITEMS_PER_THREAD = 32;
auto constexpr THREADS_PER_TILE = 128;
auto constexpr ITEMS_PER_TILE   = ITEMS_PER_THREAD * THREADS_PER_TILE;
auto constexpr TILES_PER_CHUNK  = 1024;
auto constexpr ITEMS_PER_CHUNK  = ITEMS_PER_TILE * TILES_PER_CHUNK;
// multibyte_split works by splitting up inputs in to 32 inputs (bytes) per thread, and transforming
// them in to data structures called "superstates". these superstates are created by searching a
// trie, but instead of a tradition trie where the search begins at a single node at the beginning,
// we allow our search to begin anywhere within the trie tree. The position within the trie tree is
// stored as a "partial match path", which indicates "we can get from here to there by a set of
// specific transitions". By scanning together superstates, we effectively know "we can get here
// from the beginning by following the inputs". By doing this, each thread knows exactly what state
// it begins in. From there, each thread can then take deterministic action. In this case, the
// deterministic action is counting and outputting delimiter offsets when a delimiter is found.

template <typename T>
struct scan_tile_state_callback {
  using WarpReduce = cub::WarpReduce<T>;

  struct _TempStorage {
    typename WarpReduce::TempStorage reduce;
    T exclusive_prefix;
  };

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline scan_tile_state_callback(TempStorage& temp_storage,
                                             scan_tile_state_view<T>& tile_state,
                                             cudf::size_type tile_idx)
    : _temp_storage(temp_storage.Alias()), _tile_state(tile_state), _tile_idx(tile_idx)
  {
  }

  __device__ inline T operator()(T const& block_aggregate)
  {
    if (threadIdx.x == 0) {
      _tile_state.set_partial_prefix(_tile_idx, block_aggregate);  //
    }

    auto predecessor_idx    = _tile_idx - 1 - threadIdx.x;
    auto predecessor_status = scan_tile_status::invalid;

    if constexpr (PARTIAL_AGGRIGATION_STRATEGY == 0) {
      if (threadIdx.x == 0) {
        _temp_storage.exclusive_prefix = _tile_state.get_inclusive_prefix(predecessor_idx);
      }
    }

    if constexpr (PARTIAL_AGGRIGATION_STRATEGY == 1) {
      // scan partials to form prefix

      auto window_partial = T{};

      if (threadIdx.x == 0) {
        do {
          auto predecessor_prefix = _tile_state.get_prefix(predecessor_idx, predecessor_status);
          window_partial          = predecessor_prefix + window_partial;
          predecessor_idx--;
        } while (predecessor_status != scan_tile_status::inclusive);

        _temp_storage.exclusive_prefix = window_partial;
      }
    }

    if constexpr (PARTIAL_AGGRIGATION_STRATEGY == 2) {
      auto window_partial = T{};
      if (threadIdx.x < 32) {
        do {
          auto predecessor_prefix = _tile_state.get_prefix(predecessor_idx, predecessor_status);

          window_partial =
            WarpReduce(_temp_storage.reduce)  //
              .TailSegmentedReduce(predecessor_prefix,
                                   predecessor_status == scan_tile_status::inclusive,
                                   [](T const& lhs, T const& rhs) { return rhs + lhs; }) +
            window_partial;
          predecessor_idx -= 32;
        } while (__all_sync(0xffffffff, predecessor_status != scan_tile_status::inclusive));
      }

      if (threadIdx.x == 0) {
        _temp_storage.exclusive_prefix = window_partial;  //
      }
    }

    if (threadIdx.x == 0) {
      _tile_state.set_inclusive_prefix(_tile_idx, _temp_storage.exclusive_prefix + block_aggregate);
    }

    __syncthreads();  // TODO: remove if unnecessary.

    return _temp_storage.exclusive_prefix;
  }

  _TempStorage& _temp_storage;
  scan_tile_state_view<T>& _tile_state;
  cudf::size_type _tile_idx;
};

struct PatternScan {
  typedef cub::BlockScan<superstate, THREADS_PER_TILE> BlockScan;
  typedef scan_tile_state_callback<superstate> BlockScanCallback;

  struct _TempStorage {
    typename BlockScan::TempStorage scan;
    typename BlockScanCallback::TempStorage scan_callback;
  };

  _TempStorage& _temp_storage;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline PatternScan(TempStorage& temp_storage) : _temp_storage(temp_storage.Alias()) {}

  __device__ inline void Scan(cudf::size_type tile_idx,
                              scan_tile_state_view<superstate> tile_state,
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

    auto prefix_callback = BlockScanCallback(_temp_storage.scan_callback, tile_state, tile_idx);

    BlockScan(_temp_storage.scan)
      .ExclusiveSum(thread_superstate, thread_superstate, prefix_callback);

    // transition from known state to known state
    thread_state[0] = trie.transition(thread_superstate.get(0), thread_data[0]);

    for (uint32_t i = 1; i < ITEMS_PER_THREAD; i++) {
      thread_state[i] = trie.transition(thread_state[i - 1], thread_data[i]);
    }
  }
};

__global__ void multibyte_split_init_kernel(cudf::size_type base_tile_idx,
                                            cudf::size_type num_tiles,
                                            scan_tile_state_view<superstate> tile_superstates,
                                            scan_tile_state_view<uint32_t> tile_output_offsets,
                                            scan_tile_status status = scan_tile_status::invalid)
{
  tile_superstates.initialize_status(base_tile_idx, num_tiles, status);
  tile_output_offsets.initialize_status(base_tile_idx, num_tiles, status);
}

__global__ void multibyte_split_kernel(cudf::size_type base_tile_idx,
                                       cudf::size_type num_tiles,
                                       scan_tile_state_view<superstate> tile_superstates,
                                       scan_tile_state_view<uint32_t> tile_output_offsets,
                                       cudf::io::text::trie_device_view trie,
                                       cudf::device_span<char const> data,
                                       cudf::device_span<int32_t> string_offsets)
{
  typedef cub::BlockScan<uint32_t, THREADS_PER_TILE> OffsetScan;
  typedef scan_tile_state_callback<uint32_t> OffsetScanCallback;

  __shared__ union {
    typename PatternScan::TempStorage pattern_scan;
    struct {
      typename OffsetScan::TempStorage offset_scan;
      typename OffsetScanCallback::TempStorage offset_scan_callback;
    };
  } temp_storage;

  int32_t const tile_idx   = base_tile_idx + blockIdx.x;
  int32_t const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t const data_begin = thread_idx * ITEMS_PER_THREAD;
  int32_t const num_valid  = data.size() - data_begin;
  int32_t const char_begin = base_tile_idx * ITEMS_PER_TILE;

  // STEP 1: Load inputs

  char thread_data[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {  //
    thread_data[i] = data[data_begin + i];
  }

  // STEP 2: Scan inputs to determine absolute thread states

  uint32_t thread_states[ITEMS_PER_THREAD];

  PatternScan(temp_storage.pattern_scan)  //
    .Scan(tile_idx, tile_superstates, trie, thread_data, thread_states);

  // STEP 3: Flag matches

  uint32_t thread_offsets[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_offsets[i] = i < num_valid and trie.is_match(thread_states[i]);
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  __syncthreads();  // required before temp_memory re-use

  auto prefix_callback =
    OffsetScanCallback(temp_storage.offset_scan_callback, tile_output_offsets, tile_idx);

  OffsetScan(temp_storage.offset_scan)
    .ExclusiveSum(thread_offsets, thread_offsets, prefix_callback);

  // Step 5: Assign string_offsets from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {
    auto const match_length = trie.get_match_length(thread_states[i]);

    if (match_length == 0) { continue; }

    auto const match_end   = char_begin + data_begin + i + 1;
    auto const match_begin = match_end - match_length;

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

  auto tile_superstates = scan_tile_state<superstate<16>>(num_tiles + 1, stream);
  auto tile_offsets     = scan_tile_state<uint32_t>(num_tiles + 1, stream);
  auto num_init_blocks  = ceil_div(num_tiles + 1, THREADS_PER_TILE);

  multibyte_split_init_kernel<<<num_init_blocks, THREADS_PER_TILE, 0, stream.value()>>>(  //
    0,
    num_tiles,
    tile_superstates,
    tile_offsets);

  tile_superstates.set_seed_async(superstate<16>(), stream);
  tile_offsets.set_seed_async(0, stream);

  multibyte_split_kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    0,
    num_tiles,
    tile_superstates,
    tile_offsets,
    trie.view(),
    cudf::device_span<char const>(input.data(), input.size()),
    cudf::device_span<cudf::size_type>(static_cast<size_type*>(nullptr), 0));

  // allocate string offsets

  auto num_results    = tile_offsets.get_inclusive_prefix(num_tiles - 1, stream);
  auto string_offsets = rmm::device_uvector<cudf::size_type>(num_results + 2, stream);
  auto const x        = string_offsets.size() - 1;
  auto const y        = input.size();

  // first and last element are set manually to zero and size of input, respectively.
  // kernel is only responsible for determining delimiter offsets
  string_offsets.set_element_to_zero_async(0, stream);
  string_offsets.set_element_async(x, y, stream);

  multibyte_split_init_kernel<<<num_init_blocks, THREADS_PER_TILE, 0, stream.value()>>>(  //
    0,
    num_tiles,
    tile_superstates,
    tile_offsets);

  tile_superstates.set_seed_async(superstate<16>(), stream);
  tile_offsets.set_seed_async(0, stream);

  // pattern-match and materialize string offsets

  multibyte_split_kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    0,
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

cudf::size_type scan_full_stream(cudf::io::text::device_istream& input,
                                 cudf::io::text::trie const& trie,
                                 scan_tile_state<superstate<16>>& tile_superstates,
                                 scan_tile_state<uint32_t>& tile_offsets,
                                 device_span<cudf::size_type> output_buffer,
                                 rmm::cuda_stream_view stream)
{
  uint32_t bytes_read;
  cudf::size_type bytes_total = 0;

  rmm::device_uvector<char> input_buffer(ITEMS_PER_CHUNK, stream);

  // this function can be updated to interleave two kernel executions, such that two input buffers

  multibyte_split_init_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, stream.value()>>>(  //
    -TILES_PER_CHUNK,
    TILES_PER_CHUNK,
    tile_superstates,
    tile_offsets,
    scan_tile_status::oob);

  tile_superstates.set_seed_async(superstate<16>(), stream);
  tile_offsets.set_seed_async(0, stream);

  for (auto base_tile_idx = 0; (bytes_read = input.read(input_buffer, stream)) > 0;
       base_tile_idx += TILES_PER_CHUNK) {
    bytes_total += bytes_read;

    // reset the next chunk of tile state
    multibyte_split_init_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, stream.value()>>>(  //
      base_tile_idx,
      TILES_PER_CHUNK,
      tile_superstates,
      tile_offsets);

    multibyte_split_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, stream.value()>>>(  //
      base_tile_idx,
      TILES_PER_CHUNK,
      tile_superstates,
      tile_offsets,
      trie.view(),
      device_span<char>(input_buffer).first(bytes_read),
      output_buffer);

    stream.synchronize();
  }

  return bytes_total;
}

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::device_istream& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto const trie       = cudf::io::text::trie::create(delimeters, stream);
  auto tile_superstates = scan_tile_state<superstate<16>>(TILES_PER_CHUNK * 2, stream);
  auto tile_offsets     = scan_tile_state<uint32_t>(TILES_PER_CHUNK * 2, stream);

  auto bytes_total =
    scan_full_stream(input,
                     trie,
                     tile_superstates,
                     tile_offsets,
                     cudf::device_span<cudf::size_type>(static_cast<size_type*>(nullptr), 0),
                     stream);

  // allocate string offsets

  auto num_tiles      = ceil_div(bytes_total, ITEMS_PER_TILE);
  auto num_results    = tile_offsets.get_inclusive_prefix(num_tiles - 1, stream);
  auto string_offsets = rmm::device_uvector<cudf::size_type>(num_results + 2, stream);

  // first and last element are set manually to zero and size of input, respectively.
  // kernel is only responsible for determining delimiter offsets
  auto const x = string_offsets.size() - 1;
  string_offsets.set_element_to_zero_async(0, stream);
  string_offsets.set_element_async(x, bytes_total, stream);

  // pattern-match and materialize string offsets
  input.reset();

  scan_full_stream(input,
                   trie,
                   tile_superstates,
                   tile_offsets,
                   cudf::device_span<cudf::size_type>(string_offsets).subspan(1, num_results),
                   stream);

  // copy chars
  auto string_chars = rmm::device_uvector<char>(bytes_total, stream);

  input.reset();
  input.read(string_chars, stream);

  // copy chars and offsets to make new strings column.
  auto result = cudf::make_strings_column(  //
    string_chars,
    string_offsets,
    {},
    0,
    stream,
    mr);

  // This synchronization is required to keep input_buffer in scope long enough to copy. Can be
  // by using `std::unique_ptr<column>` overload, or making a new one that accepts `device_uvector`.
  stream.synchronize();

  return result;
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::string_scalar const& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeters, rmm::cuda_stream_default, mr);
}

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::device_istream& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeters, rmm::cuda_stream_default, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
