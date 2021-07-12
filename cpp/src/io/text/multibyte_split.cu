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

struct trie_state {
  uint8_t placeholder;
};

using superstate = cudf::io::text::superstate<16>;

// keep BYTES_PER_TILE below input size to force multi-tile execution.
auto constexpr BYTES_PER_THREAD = 2;
auto constexpr THREADS_PER_TILE = 32;
auto constexpr BYTES_PER_TILE   = BYTES_PER_THREAD * THREADS_PER_TILE;
auto constexpr TILES_PER_CHUNK  = 1024;
auto constexpr BYTES_PER_CHUNK  = BYTES_PER_TILE * TILES_PER_CHUNK;
// multibyte_split works by splitting up inputs in to 32 inputs (bytes) per thread, and transforming
// them in to data structures called "superstates". these superstates are created by searching a
// trie, but instead of a tradition trie where the search begins at a single node at the beginning,
// we allow our search to begin anywhere within the trie tree. The position within the trie tree is
// stored as a "partial match path", which indicates "we can get from here to there by a set of
// specific transitions". By scanning together superstates, we effectively know "we can get here
// from the beginning by following the inputs". By doing this, each thread knows exactly what state
// it begins in. From there, each thread can then take deterministic action. In this case, the
// deterministic action is counting and outputting delimiter offsets when a delimiter is found.

template <uint32_t BYTES_PER_THREAD>
__global__ void multibyte_split_kernel(cudf::io::text::trie_device_view trie,
                                       cudf::device_span<char const> data,
                                       uint32_t* result_count,
                                       cudf::device_span<int32_t> results)
{
  typedef cub::BlockScan<superstate, THREADS_PER_TILE> SuperstateBlockScan;
  typedef cub::BlockScan<uint32_t, THREADS_PER_TILE> OffsetBlockScan;

  __shared__ union {
    typename SuperstateBlockScan::TempStorage superstate_scan;
    typename OffsetBlockScan::TempStorage offset_scan;
  } temp_storage;

  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto const data_begin = thread_idx * BYTES_PER_THREAD;
  auto data_end         = data_begin + BYTES_PER_THREAD;

  if (data_end > data.size()) { data_end = data.size(); }

  // STEP 1 + 2: Load inputs, transform to individual superstates

  superstate thread_superstates[BYTES_PER_THREAD];

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    auto const element_idx = data_begin + i;
    if (element_idx >= data.size()) {
      // this check is not necessary if we gaurantee no OOB accesses, which we can do because of
      // the batch-read/batch-process approach. Keeping the check in for now, though.
      thread_superstates[i] = superstate();
    } else {
      thread_superstates[i] = superstate().apply([&](uint8_t state) {  //
        return trie.transition(state, data[element_idx]);
      });
    }
  }

  // STEP 3: Scan superstates can to produce absolute thread states.

  __syncthreads();
  SuperstateBlockScan(temp_storage.superstate_scan)
    .InclusiveScan(  //
      thread_superstates,
      thread_superstates,
      [](superstate const& lhs, superstate const& rhs) { return lhs + rhs; });

  // STEP 4: Populate match flags

  uint32_t thread_offsets[BYTES_PER_THREAD];

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    thread_offsets[i] = trie.is_match(thread_superstates[i].get(0));
  }

  // STEP 5: Scan match flags to produce match offsets

  uint32_t matches_in_block;

  __syncthreads();
  OffsetBlockScan(temp_storage.offset_scan)
    .ExclusiveScan(
      thread_offsets,
      thread_offsets,
      [](uint32_t const& lhs, uint32_t const& rhs) { return lhs + rhs; },
      matches_in_block);

  // Step 6: Assign final block-aggregate match offset as the total number of matches.

  if (threadIdx.x == 0) { *result_count = matches_in_block; }

  // Step 7: Assign results from each thread using match offsets.

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    auto const match_length = trie.get_match_length(thread_superstates[i].get(0));

    if (match_length == 0) { continue; }

    auto const match_end   = data_begin + i + 1;
    auto const match_begin = match_end - match_length;

    printf("bid(%2u) tid(%2u) byte(%2u): %c %2u - [%3u, %3u)\n",  //
           blockIdx.x,
           threadIdx.x,
           i,
           data[data_begin + i],
           thread_offsets[i],
           match_begin,
           match_end);

    if (results.size() > 0) { results[thread_offsets[i]] = match_end; }
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
  // auto input_buffer     = rmm::device_uvector<char>(BYTES_PER_CHUNK, stream);
  // auto const input_span = cudf::device_span<char>(input_buffer);

  // TODO: call state initalization kernels

  auto const trie = cudf::io::text::trie::create(delimeters, stream);

  auto num_results = rmm::device_scalar<uint32_t>(0, stream);
  auto num_tiles   = ceil_div(input.size(), BYTES_PER_TILE);

  auto offsets = rmm::device_uvector<cudf::size_type>(0, stream);

  // count the results

  auto kernel = multibyte_split_kernel<BYTES_PER_THREAD>;
  kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    trie.view(),
    cudf::device_span<char const>(input.data(), input.size()),
    num_results.data(),
    offsets);

  auto host_num_results = num_results.value(stream);

  stream.synchronize();

  std::cout << "num results: " << host_num_results << std::endl;

  // allocate the results

  offsets = rmm::device_uvector<cudf::size_type>(host_num_results + 2, stream);
  offsets.set_element_to_zero_async(0, stream);
  cudf::size_type const x = offsets.size() - 1;
  cudf::size_type const y = input.size();
  offsets.set_element_async(x, y, stream);

  // materialize the results

  kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
    trie.view(),
    cudf::device_span<char const>(input.data(), input.size()),
    num_results.data(),
    cudf::device_span<cudf::size_type>(offsets.data() + 1, host_num_results));

  stream.synchronize();

  // TODO: call state finalization kernels

  return cudf::make_strings_column(  //
    cudf::device_span<char const>(input.data(), input.size()),
    offsets);

  CUDF_FAIL();

  /*
  std::unique_ptr<column> make_strings_column(
  cudf::device_span<char const> strings,
  cudf::device_span<size_type const> offsets,
  cudf::device_span<bitmask_type const> null_mask = {},
  size_type null_count                            = cudf::UNKNOWN_NULL_COUNT,
  rmm::cuda_stream_view stream                    = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr             = rmm::mr::get_current_device_resource());
  */
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
