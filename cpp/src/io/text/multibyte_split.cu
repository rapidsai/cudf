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
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/copy.h>
#include <thrust/transform.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#pragma GCC diagnostic pop

#include <memory>
#include <numeric>
#include <optional>

namespace cudf {

/**
 * @brief A device span consisting of two separate device_spans acting as if they were part of a
 * single span. The first head.size() entries are served from the first span, the remaining
 * tail.size() entries are served from the second span.
 *
 * @tparam T The type of elements in the span.
 */
template <typename T>
class split_device_span {
 public:
  explicit constexpr split_device_span(device_span<T> head, device_span<T> tail = {})
    : _head{head}, _tail{tail}
  {
  }

  [[nodiscard]] constexpr T& operator[](size_type i)
  {
    return i < _head.size() ? _head[i] : _tail[i - _head.size()];
  }

  [[nodiscard]] constexpr const T& operator[](size_type i) const
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

// This struct provides output offsets that are only incremented until a cutoff point.
struct cutoff_offset {
  // magnitude stores the offset, sign bit stores whether we are past the cutoff
  int64_t value = 0;

  constexpr cutoff_offset() = default;

  constexpr cutoff_offset(int64_t offset, bool is_past_cutoff)
    : value{is_past_cutoff ? -offset : offset}
  {
  }

  [[nodiscard]] constexpr int64_t offset() const { return value < 0 ? -value : value; }

  [[nodiscard]] constexpr bool is_past_end() { return value < 0; }

  friend constexpr cutoff_offset operator+(cutoff_offset lhs, cutoff_offset rhs)
  {
    auto const past_end = lhs.is_past_end() or rhs.is_past_end();
    auto const offset   = lhs.offset() + (lhs.is_past_end() ? 0 : rhs.offset());
    return cutoff_offset{offset, past_end};
  }
};

__global__ void multibyte_split_init_kernel(
  cudf::size_type base_tile_idx,
  cudf::size_type num_tiles,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<cutoff_offset> tile_output_offsets,
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
  cudf::io::text::detail::scan_tile_state_view<cutoff_offset> tile_output_offsets,
  multistate tile_multistate_seed,
  cutoff_offset tile_output_offset)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx == 0) {
    tile_multistates.set_inclusive_prefix(-1, tile_multistate_seed);
    tile_output_offsets.set_inclusive_prefix(-1, tile_output_offset);
  }
}

__global__ void multibyte_split_kernel(
  cudf::size_type base_tile_idx,
  int64_t base_input_offset,
  int64_t base_offset_offset,
  cudf::io::text::detail::scan_tile_state_view<multistate> tile_multistates,
  cudf::io::text::detail::scan_tile_state_view<cutoff_offset> tile_output_offsets,
  cudf::io::text::detail::trie_device_view trie,
  cudf::device_span<char const> chunk_input_chars,
  int64_t byte_range_end,
  cudf::split_device_span<int64_t> output_offsets)
{
  using InputLoad =
    cub::BlockLoad<char, THREADS_PER_TILE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
  using OffsetScan         = cub::BlockScan<cutoff_offset, THREADS_PER_TILE>;
  using OffsetScanCallback = cudf::io::text::detail::scan_tile_state_callback<cutoff_offset>;

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

  cutoff_offset thread_offset;

  for (int32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    auto const is_match      = i < thread_input_size and trie.is_match(thread_states[i]);
    auto const match_end     = base_input_offset + thread_input_offset + i + 1;
    auto const is_past_range = match_end >= byte_range_end;
    thread_offset            = thread_offset + cutoff_offset{is_match, is_past_range};
  }

  // STEP 4: Scan flags to determine absolute thread output offset

  auto prefix_callback = OffsetScanCallback(tile_output_offsets, tile_idx);

  __syncthreads();  // required before temp_memory re-use
  OffsetScan(temp_storage.offset_scan).ExclusiveSum(thread_offset, thread_offset, prefix_callback);

  // Step 5: Assign outputs from each thread using match offsets.

  for (int32_t i = 0; i < ITEMS_PER_THREAD and i < thread_input_size; i++) {
    if (trie.is_match(thread_states[i]) and not thread_offset.is_past_end()) {
      auto const match_end     = base_input_offset + thread_input_offset + i + 1;
      auto const is_past_range = match_end >= byte_range_end;
      output_offsets[thread_offset.offset() - base_offset_offset] = match_end;
      thread_offset = thread_offset + cutoff_offset{true, is_past_range};
    }
  }
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

/**
 * @brief A chunked storage class that provides preallocated memory for algorithms with known
 * worst-case output size. It provides functionality to retrieve the next chunk to write to, for
 * reporting how much memory was actually written and for gathering all previously written outputs
 * into a single contiguous vector.
 *
 * @tparam T The output element type.
 */
template <typename T>
class output_builder {
 public:
  using size_type = typename rmm::device_uvector<T>::size_type;

  /**
   * @brief Initializes an output builder with given worst-case output size and stream.
   *
   * @param max_write_size the maximum number of elements that will be written into a
   *                       split_device_span returned from `next_output`.
   * @param stream the stream used to allocate the first chunk of memory.
   * @param mr optional, the memory resource to use for allocation.
   */
  output_builder(size_type max_write_size,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : _size{0}, _max_write_size{max_write_size}
  {
    CUDF_EXPECTS(max_write_size > 0, "Internal error");
    _chunks.emplace_back(0, stream, mr);
    _chunks.back().reserve(max_write_size * 2, stream);
  }

  output_builder(output_builder&&)      = delete;
  output_builder(const output_builder&) = delete;
  output_builder& operator=(output_builder&&) = delete;
  output_builder& operator=(const output_builder&) = delete;

  /**
   * @brief Returns the next free chunk of `max_write_size` elements from the underlying storage.
   * Must be followed by a call to `advance_output` after the memory has been written to.
   *
   * @param stream The stream to allocate a new chunk of memory with, if necessary.
   *               This should be the stream that will write to the `split_device_span`.
   * @return A `split_device_span` starting directly after the last output and providing at least
   *         `max_write_size` entries of storage.
   */
  [[nodiscard]] split_device_span<T> next_output(rmm::cuda_stream_view stream)
  {
    auto head_it   = _chunks.end() - (_chunks.size() > 1 and _chunks.back().is_empty() ? 2 : 1);
    auto head_span = get_free_span(*head_it);
    if (head_span.size() >= _max_write_size) { return split_device_span<T>{head_span}; }
    if (head_it == _chunks.end() - 1) {
      // insert a new vector of double size
      auto const next_chunk_size = 2 * _chunks.back().capacity();
      _chunks.emplace_back(0, stream, _chunks.back().memory_resource());
      _chunks.back().reserve(next_chunk_size, stream);
    }
    auto tail_span = get_free_span(_chunks.back());
    CUDF_EXPECTS(head_span.size() + tail_span.size() >= _max_write_size, "Internal error");
    return split_device_span<T>{head_span, tail_span};
  }

  /**
   * @brief Advances the output sizes after a `split_device_span` returned from `next_output` was
   *        written to.
   *
   * @param actual_size The number of elements that were written to the result of the previous
   *                    `next_output` call.
   */
  void advance_output(size_type actual_size)
  {
    CUDF_EXPECTS(actual_size <= _max_write_size, "Internal error");
    if (_chunks.size() < 2) {
      auto const new_size = _chunks.back().size() + actual_size;
      inplace_resize(_chunks.back(), new_size);
    } else {
      auto& tail              = _chunks.back();
      auto& prev              = _chunks.rbegin()[1];
      auto const prev_advance = std::min(actual_size, prev.capacity() - prev.size());
      auto const tail_advance = actual_size - prev_advance;
      inplace_resize(prev, prev.size() + prev_advance);
      inplace_resize(tail, tail.size() + tail_advance);
    }
    _size += actual_size;
  }

  /**
   * @brief Returns the first element that was written to the output.
   *        Requires a previous call to `next_output` and `advance_output` and `size() > 0`.
   * @param stream The stream used to access the element.
   * @return The first element that was written to the output.
   */
  [[nodiscard]] T front_element(rmm::cuda_stream_view stream) const
  {
    return _chunks.front().front_element(stream);
  }

  /**
   * @brief Returns the last element that was written to the output.
   *        Requires a previous call to `next_output` and `advance_output` and `size() > 0`.
   * @param stream The stream used to access the element.
   * @return The last element that was written to the output.
   */
  [[nodiscard]] T back_element(rmm::cuda_stream_view stream) const
  {
    auto const& last_nonempty_chunk =
      _chunks.size() > 1 and _chunks.back().is_empty() ? _chunks.rbegin()[1] : _chunks.back();
    return last_nonempty_chunk.back_element(stream);
  }

  [[nodiscard]] size_type size() const { return _size; }

  /**
   * @brief Gathers all previously written outputs into a single contiguous vector.
   *
   * @param stream The stream used to allocate and gather the output vector. All previous write
   *               operations to the output buffer must have finished or happened on this stream.
   * @param mr The memory resource used to allocate the output vector.
   * @return The output vector.
   */
  rmm::device_uvector<T> gather(rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr) const
  {
    rmm::device_uvector<T> output{size(), stream, mr};
    auto output_it = output.begin();
    for (auto const& chunk : _chunks) {
      output_it = thrust::copy(
        rmm::exec_policy_nosync(stream), chunk.begin(), chunk.begin() + chunk.size(), output_it);
    }
    return output;
  }

 private:
  /**
   * @brief Resizes a vector without reallocating
   *
   * @param vector The vector
   * @param new_size The new size. Must be smaller than the vector's capacity
   */
  static void inplace_resize(rmm::device_uvector<T>& vector, size_type new_size)
  {
    CUDF_EXPECTS(new_size <= vector.capacity(), "Internal error");
    vector.resize(new_size, rmm::cuda_stream_view{});
  }

  /**
   * @brief Returns the span consisting of all currently unused elements in the vector
   * (`i >= size() and i < capacity()`).
   *
   * @param vector The vector.
   * @return The span of unused elements.
   */
  static device_span<T> get_free_span(rmm::device_uvector<T>& vector)
  {
    return device_span<T>{vector.data() + vector.size(), vector.capacity() - vector.size()};
  }

  size_type _size;
  size_type _max_write_size;
  std::vector<rmm::device_uvector<T>> _chunks;
};

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::data_chunk_source const& source,
                                              std::string const& delimiter,
                                              byte_range_info byte_range,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr,
                                              rmm::cuda_stream_pool& stream_pool)
{
  CUDF_FUNC_RANGE();

  if (byte_range.empty()) { return make_empty_column(type_id::STRING); }

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
  auto tile_offsets     = scan_tile_state<cutoff_offset>(num_tile_states, stream);

  multibyte_split_init_kernel<<<TILES_PER_CHUNK,
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
  multibyte_split_seed_kernel<<<1, 1, 0, stream.value()>>>(  //
    tile_multistates,
    tile_offsets,
    multistate_seed,
    {});

  auto reader               = source.create_reader();
  auto chunk_offset         = std::max<int64_t>(0, byte_range.offset() - delimiter.size());
  auto const byte_range_end = byte_range.offset() + byte_range.size();
  reader->skip_bytes(chunk_offset);
  output_builder<int64_t> offset_storage(ITEMS_PER_CHUNK / delimiter.size() + 1, stream);
  output_builder<char> char_storage(ITEMS_PER_CHUNK, stream);

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
    if (last_offset.has_value() or
        (not first_offset.has_value() and chunk_offset >= byte_range_end)) {
      break;
    }

    auto tiles_in_launch =
      cudf::util::div_rounding_up_safe(chunk->size(), static_cast<std::size_t>(ITEMS_PER_TILE));

    auto offset_output = offset_storage.next_output(scan_stream);

    // reset the next chunk of tile state
    multibyte_split_init_kernel<<<tiles_in_launch,
                                  THREADS_PER_TILE,
                                  0,
                                  scan_stream.value()>>>(  //
      base_tile_idx,
      tiles_in_launch,
      tile_multistates,
      tile_offsets);

    cudaStreamWaitEvent(scan_stream.value(), last_launch_event);

    multibyte_split_kernel<<<tiles_in_launch,
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
    if (next_tile_offset.offset() > 0 and not first_offset) {
      first_offset = offset_storage.front_element(scan_stream);
    }
    if (next_tile_offset.is_past_end()) { last_offset = offset_storage.back_element(scan_stream); }
    // copy over the characters we need, if we already encountered the first field delimiter
    if (first_offset.has_value()) {
      auto const begin    = chunk->data() + std::max<int64_t>(0, *first_offset - chunk_offset);
      auto const sentinel = last_offset.value_or(std::numeric_limits<int64_t>::max());
      auto const end = chunk->data() + std::min<int64_t>(sentinel - chunk_offset, chunk->size());
      auto const output_size = end - begin;
      auto char_output       = char_storage.next_output(scan_stream);
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

  // if the input was empty, we didn't find a delimiter at all,
  // or the first delimiter was also the last: empty output
  if (chunk_offset == 0 or not first_offset.has_value() or first_offset == last_offset) {
    return make_empty_column(type_id::STRING);
  }

  auto chars          = char_storage.gather(stream, mr);
  auto global_offsets = offset_storage.gather(stream, mr);

  bool const insert_begin = *first_offset == 0;
  bool const insert_end   = not last_offset.has_value() or last_offset == chunk_offset;
  rmm::device_uvector<int32_t> offsets{
    global_offsets.size() + insert_begin + insert_end, stream, mr};
  if (insert_begin) { offsets.set_element_to_zero_async(0, stream); }
  if (insert_end) { offsets.set_element(offsets.size() - 1, chunk_offset - *first_offset, stream); }
  thrust::transform(rmm::exec_policy(stream),
                    global_offsets.begin(),
                    global_offsets.end(),
                    offsets.begin() + insert_begin,
                    [baseline = *first_offset] __device__(int64_t global_offset) {
                      return static_cast<int32_t>(global_offset - baseline);
                    });

  auto string_count = offsets.size() - 1;

  return cudf::make_strings_column(string_count, std::move(offsets), std::move(chars));
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
