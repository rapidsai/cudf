/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/io/text/detail/multistate.hpp>
#include <cudf/io/text/detail/tile_state.hpp>
#include <cudf/io/text/detail/trie.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cub/block/block_scan.cuh>

#include <iostream>
#include <memory>

namespace {

template <typename Dividend, typename Divisor>
inline constexpr auto ceil_div(Dividend dividend, Divisor divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

using cudf::io::text::detail::multistate;

auto constexpr ITEMS_PER_THREAD = 32;  // influences register pressure
auto constexpr THREADS_PER_TILE = 32;  // must be >= 32 for warp-reduce. bugged for > 32, needs fix
auto constexpr ITEMS_PER_TILE   = ITEMS_PER_THREAD * THREADS_PER_TILE;
auto constexpr TILES_PER_CHUNK  = 512;
// keep ITEMS_PER_CHUNK below input size to force multi-tile execution.
auto constexpr ITEMS_PER_CHUNK = ITEMS_PER_TILE * TILES_PER_CHUNK;

struct PatternScan {
  typedef cub::BlockScan<multistate, THREADS_PER_TILE> BlockScan;
  typedef cudf::io::text::detail::scan_tile_state_callback<multistate> BlockScanCallback;

  struct _TempStorage {
    typename BlockScan::TempStorage scan;
    typename BlockScanCallback::TempStorage scan_callback;
  };

  _TempStorage& _temp_storage;

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline PatternScan(TempStorage& temp_storage) : _temp_storage(temp_storage.Alias()) {}

  __device__ inline void Scan(cudf::size_type tile_idx,
                              cudf::io::text::detail::scan_tile_state_view<multistate> tile_state,
                              cudf::io::text::detail::trie_device_view trie,
                              char (&thread_data)[ITEMS_PER_THREAD],
                              uint32_t (&thread_state)[ITEMS_PER_THREAD])
  {
    auto thread_multistate = trie.transition_init(thread_data[0]);

    for (uint32_t i = 1; i < ITEMS_PER_THREAD; i++) {
      thread_multistate = trie.transition(thread_data[i], thread_multistate);
    }

    auto prefix_callback = BlockScanCallback(_temp_storage.scan_callback, tile_state, tile_idx);

    BlockScan(_temp_storage.scan)
      .ExclusiveSum(thread_multistate, thread_multistate, prefix_callback);

    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      thread_multistate = trie.transition(thread_data[i], thread_multistate);

      thread_state[i] = thread_multistate.max_tail();
    }
  }
};

// multibyte_split works by splitting up inputs in to 32 inputs (bytes) per thread, and transforming
// them in to data structures called "multistates". these multistates are created by searching a
// trie, but instead of a tradition trie where the search begins at a single node at the beginning,
// we allow our search to begin anywhere within the trie tree. The position within the trie tree is
// stored as a "partial match path", which indicates "we can get from here to there by a set of
// specific transitions". By scanning together multistates, we effectively know "we can get here
// from the beginning by following the inputs". By doing this, each thread knows exactly what state
// it begins in. From there, each thread can then take deterministic action. In this case, the
// deterministic action is counting and outputting delimiter offsets when a delimiter is found.

__global__ void multibyte_split_init_kernel(
  cudf::size_type base_tile_idx,
  cudf::size_type num_tiles,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<uint32_t> tile_output_offsets,
  cudf::io::text::detail::scan_tile_status status =
    cudf::io::text::detail::scan_tile_status::invalid)
{
  tile_multistates.initialize_status(base_tile_idx, num_tiles, status);
  tile_output_offsets.initialize_status(base_tile_idx, num_tiles, status);
}

__global__ void multibyte_split_kernel(
  cudf::size_type base_tile_idx,
  cudf::size_type num_tiles,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<uint32_t> tile_output_offsets,
  cudf::io::text::detail::trie_device_view trie,
  cudf::device_span<char const> data,
  cudf::device_span<int32_t> string_offsets,
  cudf::device_span<char> data_out)
{
  typedef cub::BlockScan<uint32_t, THREADS_PER_TILE> OffsetScan;
  typedef cudf::io::text::detail::scan_tile_state_callback<uint32_t> OffsetScanCallback;

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
    .Scan(tile_idx, tile_multistates, trie, thread_data, thread_states);

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

  // Step 5: Assign outputs from each thread using match offsets.

  if (data_out.size() > 0) {
    for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {
      data_out[data_begin + i] = thread_data[i];
    }
  }

  if (string_offsets.size() > 0) {
    for (int32_t i = 0; i < ITEMS_PER_THREAD and i < num_valid; i++) {
      if (trie.get_match_length(thread_states[i]) > 0) {
        auto const match_end              = char_begin + data_begin + i + 1;
        string_offsets[thread_offsets[i]] = match_end;
      }
    }
  }
}

}  // namespace

namespace cudf {
namespace io {
namespace text {
namespace detail {

template <typename T>
std::unique_ptr<column> create_column(rmm::device_uvector<T>&& values)
{
  auto size  = values.size();
  auto dtype = cudf::data_type{cudf::type_to_id<T>()};

  CUDF_EXPECTS(dtype.id() != type_id::EMPTY, "column type_id cannot be EMPTY");

  return std::make_unique<cudf::column>(dtype, size, values.release(), rmm::device_buffer(), 0);
}

std::unique_ptr<column> create_char_column(rmm::device_uvector<char>&& values)
{
  auto size  = values.size();
  auto dtype = cudf::data_type{type_id::INT8};

  return std::make_unique<cudf::column>(dtype, size, values.release(), rmm::device_buffer(), 0);
}

std::unique_ptr<column> create_strings_column(rmm::device_uvector<char>&& chars,
                                              rmm::device_uvector<int32_t>&& offsets,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto num_strings    = offsets.size() - 1;
  auto chars_column   = create_char_column(std::move(chars));
  auto offsets_column = create_column(std::move(offsets));

  return cudf::make_strings_column(
    num_strings, std::move(offsets_column), std::move(chars_column), 0, {}, stream, mr);
}

void fork_stream_to_pool(rmm::cuda_stream_view stream, rmm::cuda_stream_pool& stream_pool)
{
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);
  for (uint32_t i = 0; i < stream_pool.get_pool_size(); i++) {
    cudaStreamWaitEvent(stream_pool.get_stream(i), event, 0);
  }
  cudaEventDestroy(event);
}

void join_pool_to_stream(rmm::cuda_stream_pool& stream_pool, rmm::cuda_stream_view stream)
{
  cudaEvent_t event;
  cudaEventCreate(&event);
  for (uint32_t i = 0; i < stream_pool.get_pool_size(); i++) {
    cudaEventRecord(event, stream_pool.get_stream(i));
    cudaStreamWaitEvent(stream, event, 0);
  }
  cudaEventDestroy(event);
}

cudf::size_type multibyte_split_scan_full_source(cudf::io::text::data_chunk_source& source,
                                                 cudf::io::text::detail::trie const& trie,
                                                 scan_tile_state<multistate>& tile_multistates,
                                                 scan_tile_state<uint32_t>& tile_offsets,
                                                 device_span<cudf::size_type> output_buffer,
                                                 device_span<char> output_char_buffer,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::cuda_stream_pool& stream_pool)
{
  CUDF_FUNC_RANGE();
  cudf::size_type bytes_total = 0;

  // this function interleaves three kernel executions

  multibyte_split_init_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, stream.value()>>>(  //
    -TILES_PER_CHUNK,
    TILES_PER_CHUNK,
    tile_multistates,
    tile_offsets,
    cudf::io::text::detail::scan_tile_status::oob);

  auto multistate_seed = multistate();

  multistate_seed.enqueue(0, 0);

  tile_multistates.set_seed_async(multistate_seed, stream);
  tile_offsets.set_seed_async(0, stream);

  fork_stream_to_pool(stream, stream_pool);

  auto reader = source.create_reader();

  for (auto base_tile_idx = 0; true; base_tile_idx += TILES_PER_CHUNK) {
    auto chunk_stream = stream_pool.get_stream();
    auto chunk        = reader->get_next_chunk(ITEMS_PER_CHUNK, chunk_stream);

    if (chunk.size() == 0) { break; }

    bytes_total += chunk.size();

    // reset the next chunk of tile state
    multibyte_split_init_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, chunk_stream>>>(  //
      base_tile_idx,
      TILES_PER_CHUNK,
      tile_multistates,
      tile_offsets);
    multibyte_split_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, chunk_stream>>>(  //
      base_tile_idx,
      TILES_PER_CHUNK,
      tile_multistates,
      tile_offsets,
      trie.view(),
      chunk,
      output_buffer,
      output_char_buffer);
  }

  join_pool_to_stream(stream_pool, stream);

  return bytes_total;
}

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source& source,
                                              std::vector<std::string> const& delimiters,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const trie  = cudf::io::text::detail::trie::create(delimiters, stream);
  auto concurrency = 2;
  // must be at least 32 when using warp-reduce on partials
  // must be at least 1 more than max possible concurrent tiles
  // best when at least 32 more than max possible concurrent tiles, due to rolling `invalid`s
  auto num_tile_states  = std::max(32, TILES_PER_CHUNK * concurrency + 32);
  auto tile_multistates = scan_tile_state<multistate>(num_tile_states, stream);
  auto tile_offsets     = scan_tile_state<uint32_t>(num_tile_states, stream);

  auto stream_pool = rmm::cuda_stream_pool(concurrency);

  auto bytes_total =
    multibyte_split_scan_full_source(source,
                                     trie,
                                     tile_multistates,
                                     tile_offsets,
                                     cudf::device_span<int32_t>(static_cast<int32_t*>(nullptr), 0),
                                     cudf::device_span<char>(static_cast<char*>(nullptr), 0),
                                     stream,
                                     stream_pool);

  // allocate string offsets

  auto num_tiles      = ceil_div(bytes_total, ITEMS_PER_TILE);
  auto num_results    = tile_offsets.get_inclusive_prefix(num_tiles - 1, stream);
  auto string_offsets = rmm::device_uvector<int32_t>(num_results + 2, stream, mr);
  auto string_chars   = rmm::device_uvector<char>(bytes_total, stream, mr);

  // first and last element are set manually to zero and size of input, respectively.
  // kernel is only responsible for determining delimiter offsets
  auto const x = string_offsets.size() - 1;
  string_offsets.set_element_to_zero_async(0, stream);
  string_offsets.set_element_async(x, bytes_total, stream);

  multibyte_split_scan_full_source(
    source,
    trie,
    tile_multistates,
    tile_offsets,
    cudf::device_span<int32_t>(string_offsets).subspan(1, num_results),
    string_chars,
    stream,
    stream_pool);

  auto res = create_strings_column(std::move(string_chars), std::move(string_offsets), stream, mr);

  return res;
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source& source,
                                              std::vector<std::string> const& delimiters,
                                              rmm::mr::device_memory_resource* mr)
{
  auto stream = rmm::cuda_stream_default;
  auto result = detail::multibyte_split(source, delimiters, stream, mr);
  stream.synchronize();
  return result;
}

}  // namespace text
}  // namespace io
}  // namespace cudf
