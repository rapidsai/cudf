/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

// Can be removed once we use Thrust 1.16+
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wsizeof-array-div"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/io/text/detail/multistate.hpp>
#include <cudf/io/text/detail/tile_state.hpp>
#include <cudf/io/text/detail/trie.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <limits>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/transform.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#pragma GCC diagnostic pop

#include <memory>
#include <numeric>
#include <optional>

namespace cudf {

template <typename T>
class split_device_span {
 public:
  using element_type    = T;                  ///< The type of the elements in the span
  using value_type      = std::remove_cv<T>;  ///< Stored value type
  using size_type       = std::size_t;        ///< The type used for the size of the span
  using difference_type = std::ptrdiff_t;     ///< std::ptrdiff_t
  using pointer         = T*;                 ///< The type of the pointer returned by data()
  using iterator        = T*;                 ///< The type of the iterator returned by begin()
  using const_pointer   = T const*;           ///< The type of the pointer returned by data() const
  using reference       = T&;  ///< The type of the reference returned by operator[](size_type)
  using const_reference =
    T const&;  ///< The type of the reference returned by operator[](size_type) const

  explicit constexpr split_device_span(device_span<T> head, device_span<T> tail = {})
    : _head{head}, _tail{tail}
  {
  }

  [[nodiscard]] constexpr reference operator[](size_type i)
  {
    return i < _head.size() ? _head[i] : _tail[i - _head.size()];
  }

  [[nodiscard]] constexpr const_reference operator[](size_type i) const
  {
    return i < _head.size() ? _head[i] : _tail[i - _head.size()];
  }

  [[nodiscard]] constexpr size_type size() const { return _head.size() + _tail.size(); }

  [[nodiscard]] constexpr device_span<T> head() const { return _head; }

  [[nodiscard]] constexpr device_span<T> tail() const { return _tail; }

 private:
  device_span<T> _head;
  device_span<T> _tail;
};

}  // namespace cudf

namespace {

using cudf::io::text::detail::multistate;

int32_t constexpr ITEMS_PER_THREAD = 32;
int32_t constexpr THREADS_PER_TILE = 128;
int32_t constexpr ITEMS_PER_TILE   = ITEMS_PER_THREAD * THREADS_PER_TILE;
int32_t constexpr TILES_PER_CHUNK  = 1024;
int32_t constexpr ITEMS_PER_CHUNK  = ITEMS_PER_TILE * TILES_PER_CHUNK;

struct PatternScan {
  using BlockScan         = cub::BlockScan<multistate, THREADS_PER_TILE>;
  using BlockScanCallback = cudf::io::text::detail::scan_tile_state_callback<multistate>;

  struct _TempStorage {
    typename BlockScan::TempStorage scan;
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

    auto prefix_callback = BlockScanCallback(tile_state, tile_idx);

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
  cudf::io::text::detail::scan_tile_state_view<int64_t> tile_output_offsets,
  cudf::io::text::detail::scan_tile_status status =
    cudf::io::text::detail::scan_tile_status::invalid)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < num_tiles) {
    auto const tile_idx = base_tile_idx + thread_idx;
    tile_multistates.set_status(tile_idx, status);
    tile_output_offsets.set_status(tile_idx, status);
  }
}

__global__ void multibyte_split_seed_kernel(
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<int64_t> tile_output_offsets,
  multistate tile_multistate_seed,
  uint32_t tile_output_offset)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx == 0) {
    tile_multistates.set_inclusive_prefix(-1, tile_multistate_seed);
    tile_output_offsets.set_inclusive_prefix(-1, tile_output_offset);
  }
}

__global__ void multibyte_split_kernel(
  cudf::size_type base_tile_idx,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<int64_t> tile_output_offsets,
  cudf::io::text::detail::trie_device_view trie,
  cudf::device_span<char const> chunk_input_chars,
  cudf::device_span<int64_t> abs_output_delimiter_offsets)
{
  using InputLoad =
    cub::BlockLoad<char, THREADS_PER_TILE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
  using OffsetScan         = cub::BlockScan<int64_t, THREADS_PER_TILE>;
  using OffsetScanCallback = cudf::io::text::detail::scan_tile_state_callback<int64_t>;

  __shared__ union {
    typename InputLoad::TempStorage input_load;
    typename PatternScan::TempStorage pattern_scan;
    typename OffsetScan::TempStorage offset_scan;
  } temp_storage;

  int32_t const tile_idx            = base_tile_idx + blockIdx.x;
  int32_t const tile_input_offset   = blockIdx.x * ITEMS_PER_TILE;
  int32_t const thread_input_offset = tile_input_offset + threadIdx.x * ITEMS_PER_THREAD;
  int32_t const thread_input_size   = chunk_input_chars.size() - thread_input_offset;

  // STEP 1: Load inputs

  char thread_chars[ITEMS_PER_THREAD];

  InputLoad(temp_storage.input_load)
    .Load(chunk_input_chars.data() + tile_input_offset,
          thread_chars,
          chunk_input_chars.size() - tile_input_offset);

  // STEP 2: Scan inputs to determine absolute thread states

  uint32_t thread_states[ITEMS_PER_THREAD];

  __syncthreads();  // required before temp_memory re-use
  PatternScan(temp_storage.pattern_scan)
    .Scan(tile_idx, tile_multistates, trie, thread_chars, thread_states);

  // STEP 3: Flag matches

  int64_t thread_offsets[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_offsets[i] = i < thread_input_size and trie.is_match(thread_states[i]);
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  auto prefix_callback = OffsetScanCallback(tile_output_offsets, tile_idx);

  __syncthreads();  // required before temp_memory re-use
  OffsetScan(temp_storage.offset_scan)
    .ExclusiveSum(thread_offsets, thread_offsets, prefix_callback);

  // Step 5: Assign outputs from each thread using match offsets.

  if (abs_output_delimiter_offsets.size() > 0) {
    for (int32_t i = 0; i < ITEMS_PER_THREAD and i < thread_input_size; i++) {
      if (trie.is_match(thread_states[i])) {
        auto const match_end =
          static_cast<int64_t>(base_tile_idx) * ITEMS_PER_TILE + thread_input_offset + i + 1;
        abs_output_delimiter_offsets[thread_offsets[i]] = match_end;
      }
    }
  }
}

struct singlepass_offset {
  // magnitude stores the offset, sign bit stores whether we are past the end of the last delimiter
  int64_t value;

  constexpr void init(bool is_delim, bool is_past_range)
  {
    value = is_past_range ? -is_delim : is_delim;
  }

  [[nodiscard]] constexpr int64_t offset() const { return value < 0 ? -value : value; }

  [[nodiscard]] constexpr bool is_past_end() { return value < 0; }

  friend constexpr singlepass_offset operator+(singlepass_offset lhs, singlepass_offset rhs)
  {
    auto const past_end = lhs.is_past_end() || rhs.is_past_end();
    auto const offset   = lhs.offset() + (lhs.is_past_end() ? 0 : rhs.offset());
    return {past_end ? -offset : offset};
  }
};

__global__ void multibyte_split_singlepass_init_kernel(
  cudf::size_type base_tile_idx,
  cudf::size_type num_tiles,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<singlepass_offset> tile_output_offsets,
  cudf::io::text::detail::scan_tile_status status =
    cudf::io::text::detail::scan_tile_status::invalid)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < num_tiles) {
    auto const tile_idx = base_tile_idx + thread_idx;
    tile_multistates.set_status(tile_idx, status);
    tile_output_offsets.set_status(tile_idx, status);
  }
}

__global__ void multibyte_split_singlepass_seed_kernel(
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<singlepass_offset> tile_output_offsets,
  multistate tile_multistate_seed,
  singlepass_offset tile_output_offset)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx == 0) {
    tile_multistates.set_inclusive_prefix(-1, tile_multistate_seed);
    tile_output_offsets.set_inclusive_prefix(-1, tile_output_offset);
  }
}

__global__ void multibyte_split_singlepass_kernel(
  cudf::size_type base_tile_idx,
  int64_t base_input_offset,
  int64_t base_offset_offset,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<singlepass_offset> tile_output_offsets,
  cudf::io::text::detail::trie_device_view trie,
  cudf::device_span<char const> chunk_input_chars,
  int64_t byte_range_end,
  cudf::split_device_span<int64_t> output_offsets)
{
  using InputLoad =
    cub::BlockLoad<char, THREADS_PER_TILE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
  using OffsetScan         = cub::BlockScan<singlepass_offset, THREADS_PER_TILE>;
  using OffsetScanCallback = cudf::io::text::detail::scan_tile_state_callback<singlepass_offset>;

  __shared__ union {
    typename InputLoad::TempStorage input_load;
    typename PatternScan::TempStorage pattern_scan;
    typename OffsetScan::TempStorage offset_scan;
  } temp_storage;

  int32_t const tile_idx            = base_tile_idx + blockIdx.x;
  int32_t const tile_input_offset   = blockIdx.x * ITEMS_PER_TILE;
  int32_t const thread_input_offset = tile_input_offset + threadIdx.x * ITEMS_PER_THREAD;
  int32_t const thread_input_size   = chunk_input_chars.size() - thread_input_offset;

  // STEP 1: Load inputs

  char thread_chars[ITEMS_PER_THREAD];

  InputLoad(temp_storage.input_load)
    .Load(chunk_input_chars.data() + tile_input_offset,
          thread_chars,
          chunk_input_chars.size() - tile_input_offset);

  // STEP 2: Scan inputs to determine absolute thread states

  uint32_t thread_states[ITEMS_PER_THREAD];

  __syncthreads();  // required before temp_memory re-use
  PatternScan(temp_storage.pattern_scan)
    .Scan(tile_idx, tile_multistates, trie, thread_chars, thread_states);

  // STEP 3: Flag matches

  singlepass_offset thread_offsets[ITEMS_PER_THREAD];

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    bool const match      = i < thread_input_size and trie.is_match(thread_states[i]);
    auto const global_idx = base_input_offset + thread_input_offset + i;
    bool const past_range = global_idx + 1 >= byte_range_end;
    thread_offsets[i].init(match, past_range);
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  auto prefix_callback = OffsetScanCallback(tile_output_offsets, tile_idx);

  __syncthreads();  // required before temp_memory re-use
  OffsetScan(temp_storage.offset_scan)
    .ExclusiveSum(thread_offsets, thread_offsets, prefix_callback);

  // Step 5: Assign outputs from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < thread_input_size; i++) {
    if (trie.is_match(thread_states[i]) && !thread_offsets[i].is_past_end()) {
      auto const match_end = base_input_offset + thread_input_offset + i + 1;
      output_offsets[thread_offsets[i].offset() - base_offset_offset] = match_end;
    }
  }
}

bool delimiter_is_local(std::string const& delimiter)
{
  for (size_t i = 1; i <= delimiter.size() - 1; i++) {
    // if a true prefix is also a true suffix: we need global context
    if (delimiter.substr(0, i) == delimiter.substr(i)) { return false; }
  }
  // otherwise local context suffices
  return true;
}

}  // namespace

namespace cudf {
namespace io {
namespace text {
namespace detail {

void fork_stream(std::vector<rmm::cuda_stream_view> streams, rmm::cuda_stream_view stream)
{
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);
  for (uint32_t i = 0; i < streams.size(); i++) {
    cudaStreamWaitEvent(streams[i], event, 0);
  }
  cudaEventDestroy(event);
}

void join_stream(std::vector<rmm::cuda_stream_view> streams, rmm::cuda_stream_view stream)
{
  cudaEvent_t event;
  cudaEventCreate(&event);
  for (uint32_t i = 0; i < streams.size(); i++) {
    cudaEventRecord(event, streams[i]);
    cudaStreamWaitEvent(stream, event, 0);
  }
  cudaEventDestroy(event);
}

std::vector<rmm::cuda_stream_view> get_streams(int32_t count, rmm::cuda_stream_pool& stream_pool)
{
  auto streams = std::vector<rmm::cuda_stream_view>();
  for (int32_t i = 0; i < count; i++) {
    streams.emplace_back(stream_pool.get_stream());
  }
  return streams;
}

int64_t multibyte_split_scan_full_source(cudf::io::text::data_chunk_source const& source,
                                         cudf::io::text::detail::trie const& trie,
                                         scan_tile_state<multistate>& tile_multistates,
                                         scan_tile_state<int64_t>& tile_offsets,
                                         device_span<int64_t> output_buffer,
                                         rmm::cuda_stream_view stream,
                                         std::vector<rmm::cuda_stream_view> const& streams)
{
  CUDF_FUNC_RANGE();
  int64_t chunk_offset = 0;

  multibyte_split_init_kernel<<<TILES_PER_CHUNK, THREADS_PER_TILE, 0, stream.value()>>>(  //
    -TILES_PER_CHUNK,
    TILES_PER_CHUNK,
    tile_multistates,
    tile_offsets,
    cudf::io::text::detail::scan_tile_status::oob);

  auto multistate_seed = multistate();
  multistate_seed.enqueue(0, 0);  // this represents the first state in the pattern.

  // Seeding the tile state with an identity value allows the 0th tile to follow the same logic as
  // the Nth tile, assuming it can look up an inclusive prefix. Without this seed, the 0th block
  // would have to follow separate logic.
  multibyte_split_seed_kernel<<<1, 1, 0, stream.value()>>>(  //
    tile_multistates,
    tile_offsets,
    multistate_seed,
    0);

  fork_stream(streams, stream);

  auto reader = source.create_reader();

  cudaEvent_t last_launch_event;
  cudaEventCreate(&last_launch_event);

  for (int32_t i = 0; true; i++) {
    auto base_tile_idx = i * TILES_PER_CHUNK;
    auto chunk_stream  = streams[i % streams.size()];
    auto chunk         = reader->get_next_chunk(ITEMS_PER_CHUNK, chunk_stream);

    if (chunk->size() == 0) { break; }

    auto tiles_in_launch =
      cudf::util::div_rounding_up_safe(chunk->size(), static_cast<std::size_t>(ITEMS_PER_TILE));

    // reset the next chunk of tile state
    multibyte_split_init_kernel<<<tiles_in_launch, THREADS_PER_TILE, 0, chunk_stream>>>(  //
      base_tile_idx,
      tiles_in_launch,
      tile_multistates,
      tile_offsets);

    cudaStreamWaitEvent(chunk_stream, last_launch_event, 0);

    multibyte_split_kernel<<<tiles_in_launch, THREADS_PER_TILE, 0, chunk_stream>>>(  //
      base_tile_idx,
      tile_multistates,
      tile_offsets,
      trie.view(),
      *chunk,
      output_buffer);

    cudaEventRecord(last_launch_event, chunk_stream);

    chunk_offset += chunk->size();

    chunk.reset();
  }

  cudaEventDestroy(last_launch_event);

  join_stream(streams, stream);

  return chunk_offset;
}

template <typename T>
class output_chunks {
 public:
  using element_type = T;                  ///< The type of the elements in the span
  using value_type   = std::remove_cv<T>;  ///< Stored value type
  using size_type    = std::size_t;        ///< The type used for the size of the span

  output_chunks(size_type initial_size,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr)
    : _size{}
  {
    assert(initial_size > 0);
    _chunks.emplace_back(0, stream, mr);
    _chunks.back().reserve(initial_size, stream);
  }

  output_chunks& operator=(output_chunks&&) = delete;
  output_chunks(output_chunks&&)            = delete;

  [[nodiscard]] split_device_span<element_type> next_output(size_type max_size,
                                                            rmm::cuda_stream_view stream)
  {
    device_span<element_type> head_span{_chunks.back().data() + _chunks.back().size(),
                                        _chunks.back().capacity() - _chunks.back().size()};
    if (head_span.size() >= max_size) { return split_device_span<element_type>{head_span}; }
    // growth and padding factor 2, reduces the number of allocations if max_size is overestimate
    auto const next_chunk_size = 2 * std::max(_chunks.back().capacity(), max_size);
    _chunks.emplace_back(0, stream, _chunks.back().memory_resource());
    _chunks.back().reserve(next_chunk_size, stream);
    device_span<element_type> tail_span{_chunks.back().data(), _chunks.back().capacity()};
    assert(head_span.size() + tail_span.size() >= max_size);
    return split_device_span<element_type>{head_span, tail_span};
  }

  void advance_output(size_type actual_size)
  {
    // this dummy stream is used for resizing, since we know we won't reallocate,
    // providing a the correct stream is not necessary.
    rmm::cuda_stream_view dummy_stream{};
    if (_chunks.size() < 2) {
      auto const new_size = _chunks.back().size() + actual_size;
      assert(new_size <= _chunks.back().capacity());
      _chunks.back().resize(new_size, dummy_stream);
    } else {
      auto& tail              = _chunks.back();
      auto& prev              = *(_chunks.end() - 2);
      auto const prev_advance = std::min(actual_size, prev.capacity() - prev.size());
      auto const tail_advance = actual_size - prev_advance;
      assert(tail.size() + tail_advance <= tail.capacity());
      prev.resize(prev.size() + prev_advance, dummy_stream);
      tail.resize(tail.size() + tail_advance, dummy_stream);
    }
    _size += actual_size;
  }

  [[nodiscard]] const rmm::device_uvector<element_type>& first_chunk() const
  {
    return _chunks.front();
  }

  [[nodiscard]] const rmm::device_uvector<element_type>& last_nonempty_chunk() const
  {
    return _chunks.back().is_empty() ? *(_chunks.end() - 2) : _chunks.back();
  }

  [[nodiscard]] size_type size() const
  {
    assert(_size ==
           std::accumulate(
             _chunks.begin(), _chunks.end(), size_type{}, [](size_type acc, const auto& chunk) {
               return acc + chunk.size();
             }));
    return _size;
  }

  rmm::device_uvector<element_type> collect(rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr) const
  {
    rmm::device_uvector<element_type> output{size(), stream, mr};
    auto output_it = output.begin();
    for (auto const& chunk : _chunks) {
      output_it = thrust::copy(
        rmm::exec_policy_nosync(stream), chunk.begin(), chunk.begin() + chunk.size(), output_it);
    }
    return output;
  }

 private:
  size_type _size;
  std::vector<rmm::device_uvector<element_type>> _chunks;
};

std::unique_ptr<cudf::column> multibyte_split_singlepass(
  cudf::io::text::data_chunk_source const& source,
  std::string const& delimiter,
  byte_range_info byte_range,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_pool& stream_pool)
{
  CUDF_FUNC_RANGE();

  if (byte_range.size() == 0) { return make_empty_column(type_id::STRING); }

  auto const trie = cudf::io::text::detail::trie::create({delimiter}, stream);

  CUDF_EXPECTS(trie.max_duplicate_tokens() < multistate::max_segment_count,
               "delimiter contains too many duplicate tokens to produce a deterministic result.");

  CUDF_EXPECTS(trie.size() < multistate::max_segment_value,
               "delimiter contains too many total tokens to produce a deterministic result.");

  auto concurrency = 2;
  auto streams     = get_streams(concurrency, stream_pool);

  // must be at least 32 when using warp-reduce on partials
  // must be at least 1 more than max possible concurrent tiles
  // best when at least 32 more than max possible concurrent tiles, due to rolling `invalid`s
  auto num_tile_states  = std::max(32, TILES_PER_CHUNK * concurrency + 32);
  auto tile_multistates = scan_tile_state<multistate>(num_tile_states, stream);
  auto tile_offsets     = scan_tile_state<singlepass_offset>(num_tile_states, stream);

  multibyte_split_singlepass_init_kernel<<<TILES_PER_CHUNK,
                                           THREADS_PER_TILE,
                                           0,
                                           stream.value()>>>(  //
    -TILES_PER_CHUNK,
    TILES_PER_CHUNK,
    tile_multistates,
    tile_offsets,
    cudf::io::text::detail::scan_tile_status::oob);

  auto multistate_seed = multistate();
  multistate_seed.enqueue(0, 0);  // this represents the first state in the pattern.

  // Seeding the tile state with an identity value allows the 0th tile to follow the same logic as
  // the Nth tile, assuming it can look up an inclusive prefix. Without this seed, the 0th block
  // would have to follow separate logic.
  multibyte_split_singlepass_seed_kernel<<<1, 1, 0, stream.value()>>>(  //
    tile_multistates,
    tile_offsets,
    multistate_seed,
    {0});

  auto reader               = source.create_reader();
  auto chunk_offset         = std::max<int64_t>(0, byte_range.offset() - delimiter.size());
  auto const byte_range_end = byte_range.offset() + byte_range.size();
  reader->skip_bytes(chunk_offset);
  output_chunks<int64_t> offset_storage(2 * ITEMS_PER_CHUNK, stream, mr);
  output_chunks<char> char_storage(2 * ITEMS_PER_CHUNK, stream, mr);

  fork_stream(streams, stream);

  cudaEvent_t last_launch_event;
  cudaEventCreate(&last_launch_event);

  auto& read_stream     = streams[0];
  auto& scan_stream     = streams[1];
  auto chunk            = reader->get_next_chunk(ITEMS_PER_CHUNK, read_stream);
  int64_t base_tile_idx = 0;
  std::optional<int64_t> first_offset;
  std::optional<int64_t> last_offset;
  if (byte_range.offset() == 0) { first_offset = 0; }
  std::swap(read_stream, scan_stream);

  while (chunk->size() > 0) {
    // if we found the last delimiter, or didn't find delimiters inside the byte range at all: abort
    if (last_offset || (!first_offset && chunk_offset >= byte_range_end)) { break; }

    auto tiles_in_launch =
      cudf::util::div_rounding_up_safe(chunk->size(), static_cast<std::size_t>(ITEMS_PER_TILE));

    cudaStreamWaitEvent(scan_stream.value(), last_launch_event);

    auto offset_output = offset_storage.next_output(ITEMS_PER_CHUNK, scan_stream);

    // reset the next chunk of tile state
    multibyte_split_singlepass_init_kernel<<<tiles_in_launch,
                                             THREADS_PER_TILE,
                                             0,
                                             scan_stream.value()>>>(  //
      base_tile_idx,
      tiles_in_launch,
      tile_multistates,
      tile_offsets);

    multibyte_split_singlepass_kernel<<<tiles_in_launch,
                                        THREADS_PER_TILE,
                                        0,
                                        scan_stream.value()>>>(  //
      base_tile_idx,
      chunk_offset,
      offset_storage.size(),
      tile_multistates,
      tile_offsets,
      trie.view(),
      *chunk,
      byte_range_end,
      offset_output);

    // load the next chunk
    auto next_chunk = reader->get_next_chunk(ITEMS_PER_CHUNK, read_stream);
    // while that is running, determine how many offsets we output (synchronizes)
    auto next_tile_offset =
      tile_offsets.get_inclusive_prefix(base_tile_idx + tiles_in_launch - 1, scan_stream);
    offset_storage.advance_output(next_tile_offset.offset() - offset_storage.size());
    // determine if we found the first or last field offset for the byte range
    if (next_tile_offset.offset() > 0 && !first_offset) {
      first_offset = offset_storage.first_chunk().front_element(scan_stream);
    }
    if (next_tile_offset.is_past_end()) {
      last_offset = offset_storage.last_nonempty_chunk().back_element(scan_stream);
    }
    // copy over the characters we need, if we already encountered the first field delimiter
    if (first_offset) {
      auto const begin    = chunk->data() + std::max<int64_t>(0, *first_offset - chunk_offset);
      auto const sentinel = last_offset.value_or(std::numeric_limits<int64_t>::max());
      auto const end = chunk->data() + std::min<int64_t>(sentinel - chunk_offset, chunk->size());
      auto const output_size = end - begin;
      auto char_output       = char_storage.next_output(output_size, scan_stream);
      auto const split       = begin + std::min<int64_t>(output_size, char_output.head().size());
      thrust::copy(rmm::exec_policy_nosync(scan_stream), begin, split, char_output.head().begin());
      thrust::copy(rmm::exec_policy_nosync(scan_stream), split, end, char_output.tail().begin());
      char_storage.advance_output(output_size);
    }

    cudaEventRecord(last_launch_event, scan_stream.value());

    std::swap(read_stream, scan_stream);
    base_tile_idx += TILES_PER_CHUNK;
    chunk_offset += chunk->size();
    chunk = std::move(next_chunk);
  }

  cudaEventDestroy(last_launch_event);

  join_stream(streams, stream);

  // if we didn't find a delimiter at all, or the first delimiter was also the last: empty output
  if (!first_offset.has_value() || first_offset == last_offset) {
    return make_empty_column(type_id::STRING);
  }

  auto chars          = char_storage.collect(stream, mr);
  auto global_offsets = offset_storage.collect(stream, mr);

  bool const insert_begin = *first_offset == 0;
  bool const insert_end   = !last_offset.has_value();
  rmm::device_uvector<int32_t> offsets{
    global_offsets.size() + insert_begin + insert_end, stream, mr};
  if (insert_begin) { offsets.set_element_to_zero_async(0, stream); }
  if (insert_end) { offsets.set_element(offsets.size() - 1, chunk_offset - *first_offset, stream); }
  thrust::transform(rmm::exec_policy_nosync(stream),
                    global_offsets.begin(),
                    global_offsets.end(),
                    offsets.begin() + insert_begin,
                    [baseline = *first_offset] __device__(int64_t global_offset) {
                      return static_cast<int32_t>(global_offset - baseline);
                    });

  auto string_count = offsets.size() - 1;

  return cudf::make_strings_column(string_count, std::move(offsets), std::move(chars));
}

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string const& delimiter,
                                              byte_range_info byte_range,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr,
                                              rmm::cuda_stream_pool& stream_pool)
{
  CUDF_FUNC_RANGE();

  if (byte_range.size() == 0) { return make_empty_column(type_id::STRING); }

  if (delimiter_is_local(delimiter)) {
    return multibyte_split_singlepass(source, delimiter, byte_range, stream, mr, stream_pool);
  }

  auto const trie = cudf::io::text::detail::trie::create({delimiter}, stream);

  CUDF_EXPECTS(trie.max_duplicate_tokens() < multistate::max_segment_count,
               "delimiter contains too many duplicate tokens to produce a deterministic result.");

  CUDF_EXPECTS(trie.size() < multistate::max_segment_value,
               "delimiter contains too many total tokens to produce a deterministic result.");

  auto concurrency = 2;
  // must be at least 32 when using warp-reduce on partials
  // must be at least 1 more than max possible concurrent tiles
  // best when at least 32 more than max possible concurrent tiles, due to rolling `invalid`s
  auto num_tile_states  = std::max(32, TILES_PER_CHUNK * concurrency + 32);
  auto tile_multistates = scan_tile_state<multistate>(num_tile_states, stream);
  auto tile_offsets     = scan_tile_state<int64_t>(num_tile_states, stream);

  auto streams = get_streams(concurrency, stream_pool);

  auto bytes_total =
    multibyte_split_scan_full_source(source,
                                     trie,
                                     tile_multistates,
                                     tile_offsets,
                                     cudf::device_span<int64_t>(static_cast<int64_t*>(nullptr), 0),
                                     stream,
                                     streams);

  if (bytes_total == 0) { return make_empty_column(type_id::STRING); }

  // allocate results
  auto num_tiles =
    cudf::util::div_rounding_up_safe(bytes_total, static_cast<int64_t>(ITEMS_PER_TILE));
  auto num_results = tile_offsets.get_inclusive_prefix(num_tiles - 1, stream);

  auto string_offsets = rmm::device_uvector<int64_t>(num_results + 2, stream);

  // first and last element are set manually to zero and size of input, respectively.
  // kernel is only responsible for determining delimiter offsets
  string_offsets.set_element_to_zero_async(0, stream);
  string_offsets.set_element_async(string_offsets.size() - 1, bytes_total, stream);

  // kernel needs to find first and last relevant offset., as well as count of relevant offsets.

  multibyte_split_scan_full_source(
    source,
    trie,
    tile_multistates,
    tile_offsets,
    cudf::device_span<int64_t>(string_offsets).subspan(1, num_results),
    stream,
    streams);

  // String offsets point to the first character of a field
  // This finds the first field whose first character starts inside or after the byte range
  auto relevant_offsets_begin = thrust::lower_bound(rmm::exec_policy(stream),
                                                    string_offsets.begin(),
                                                    string_offsets.end() - 1,
                                                    byte_range.offset());

  // This finds the first field beginning after the byte range.
  // We shift it by 1 to also copy this last offset
  auto relevant_offsets_end = 1 + thrust::lower_bound(rmm::exec_policy(stream),
                                                      string_offsets.begin(),
                                                      string_offsets.end() - 1,
                                                      byte_range.offset() + byte_range.size());

  // The above logic works if there are no duplicate string_offsets entries.
  // The only way we can get duplicates is if the input ends with a delimiter.
  // relevant_offsets_end should then point to the last entry, not the second-to-last, which can
  // happen if byte_range.offset() + byte_range.size() matches the input size exactly.
  // Without this adjustment, string_offsets_out would be missing the last element.
  bool last_field_empty = string_offsets.size() >= 2 &&
                          string_offsets.element(string_offsets.size() - 2, stream) == bytes_total;
  bool byte_range_exact_end = byte_range.offset() + byte_range.size() == bytes_total;
  if (last_field_empty && byte_range_exact_end) { ++relevant_offsets_end; }

  auto string_offsets_out_size = relevant_offsets_end - relevant_offsets_begin;

  auto string_offsets_out = rmm::device_uvector<int32_t>(string_offsets_out_size, stream, mr);

  auto relevant_offset_first =
    string_offsets.element(relevant_offsets_begin - string_offsets.begin(), stream);
  auto relevant_offset_last =
    string_offsets.element(relevant_offsets_end - string_offsets.begin() - 1, stream);

  auto string_chars_size = relevant_offset_last - relevant_offset_first;
  auto string_chars      = rmm::device_uvector<char>(string_chars_size, stream, mr);

  // copy relevant offsets and adjust them to be zero-based.
  thrust::transform(rmm::exec_policy(stream),
                    relevant_offsets_begin,
                    relevant_offsets_end,
                    string_offsets_out.begin(),
                    [relevant_offset_first] __device__(int64_t offset) {
                      return static_cast<int32_t>(offset - relevant_offset_first);
                    });

  auto reader = source.create_reader();
  reader->skip_bytes(relevant_offset_first);

  for (int32_t i = 0; i < string_chars_size; i += ITEMS_PER_CHUNK) {
    auto const read_size   = std::min<int64_t>(ITEMS_PER_CHUNK, string_chars_size - i);
    auto const chunk_bytes = reader->get_next_chunk(read_size, stream);

    thrust::copy(rmm::exec_policy(stream),
                 chunk_bytes->data(),
                 chunk_bytes->data() + chunk_bytes->size(),
                 string_chars.begin() + i);
  }

  auto string_count = string_offsets_out.size() - 1;

  return cudf::make_strings_column(
    string_count, std::move(string_offsets_out), std::move(string_chars));
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string const& delimiter,
                                              std::optional<byte_range_info> byte_range,
                                              rmm::mr::device_memory_resource* mr)
{
  auto stream      = cudf::default_stream_value;
  auto stream_pool = rmm::cuda_stream_pool(2);

  auto result = detail::multibyte_split(
    source, delimiter, byte_range.value_or(create_byte_range_info_max()), stream, mr, stream_pool);

  return result;
}

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string const& delimiter,
                                              rmm::mr::device_memory_resource* mr)
{
  return multibyte_split(source, delimiter, std::nullopt, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
