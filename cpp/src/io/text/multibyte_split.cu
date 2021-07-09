#include <cudf/column/column.hpp>
#include <cudf/io/text/input_stream.hpp>
#include <cudf/io/text/superstate.hpp>
#include <cudf/io/text/trie.hpp>
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

auto constexpr BYTES_PER_THREAD = 8;
auto constexpr THREADS_PER_TILE = 32;
auto constexpr BYTES_PER_TILE   = BYTES_PER_THREAD * THREADS_PER_TILE;
auto constexpr TILES_PER_CHUNK  = 1024;
auto constexpr BYTES_PER_CHUNK  = BYTES_PER_TILE * TILES_PER_CHUNK;

struct BlockPrefixCallbackOp {
  // Running prefix
  superstate running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(superstate running_total) : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ superstate operator()(superstate const& block_aggregate)
  {
    superstate old_prefix = running_total;
    running_total         = old_prefix + block_aggregate;
    return old_prefix;
  }

  static rmm::device_uvector<superstate> create_temp_storage(uint32_t num_elements,
                                                             rmm::cuda_stream_view stream)
  {
    auto num_prefixes = ceil_div(num_elements, BYTES_PER_TILE);

    return rmm::device_uvector<superstate>(num_prefixes, stream);
  }
};

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
                                       cudf::device_span<char> data,
                                       uint32_t* result_count)
{
  typedef cub::BlockScan<superstate, THREADS_PER_TILE> SuperstateBlockScan;
  typedef cub::BlockScan<uint32_t, THREADS_PER_TILE> ResultOffsetBlockScan;

  __shared__ union {
    typename SuperstateBlockScan::TempStorage superstate_scan;
    typename ResultOffsetBlockScan::TempStorage result_offset_scan;
  } temp_storage;

  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto const data_begin = thread_idx * BYTES_PER_THREAD;
  auto data_end         = data_begin + BYTES_PER_THREAD;

  if (data_end > data.size()) { data_end = data.size(); }

  superstate thread_data[BYTES_PER_THREAD];

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    auto const element_idx = data_begin + i;
    if (element_idx >= data.size()) {
      // this check is not necessary if we gaurantee no OOB accesses, which we can do because of
      // the batch-read/batch-process approach. Keeping the check in for now, though.
      thread_data[i] = superstate();
    } else {
      thread_data[i] = superstate().apply([&](uint8_t state) {  //
        return trie.transition(state, data[element_idx]);
      });
    }
  }

  BlockPrefixCallbackOp prefix_op({});

  __syncthreads();

  SuperstateBlockScan(temp_storage.superstate_scan)
    .InclusiveScan(  //
      thread_data,
      thread_data,
      [](superstate const& lhs, superstate const& rhs) { return lhs + rhs; },
      prefix_op);

  __syncthreads();

  uint32_t thread_offsets[BYTES_PER_THREAD];

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    auto const element_idx = data_begin + i;
    if (element_idx < data.size()) {
      thread_offsets[i] = trie.is_match(thread_data[i].get(0));
    } else {
      thread_offsets[i] = false;
    }
  }

  uint32_t matches_in_block;

  ResultOffsetBlockScan(temp_storage.result_offset_scan)
    .ExclusiveScan(
      thread_offsets,
      thread_offsets,
      [](uint32_t const& lhs, uint32_t const& rhs) { return lhs + rhs; },
      matches_in_block);

  if (threadIdx.x == 0) { *result_count = matches_in_block; }

  // for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
  //   auto const element_idx = data_begin + i;
  //   if (element_idx < data.size()) {
  //     thread_offsets[i] = trie.is_match(thread_data[i].get(0));
  //   } else {
  //     thread_offsets[i] = false;
  //   }
  // }
}

}  // namespace

namespace cudf {
namespace io {
namespace text {
namespace detail {

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::input_stream& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto input_buffer     = rmm::device_uvector<char>(BYTES_PER_CHUNK, stream);
  auto const input_span = cudf::device_span<char>(input_buffer);

  // TODO: call state initalization kernels

  auto const trie = cudf::io::text::trie::create(delimeters, stream);

  auto num_results = rmm::device_scalar<uint32_t>(0, stream);

  while (true) {
    uint32_t num_bytes_read = input.readsome(input_span, stream);

    if (num_bytes_read == 0) {
      // if there's no more data to read, we're done.
      break;
    }

    auto num_tiles = ceil_div(num_bytes_read, BYTES_PER_TILE);

    auto kernel = multibyte_split_kernel<BYTES_PER_THREAD>;
    kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(  //
      trie.view(),
      input_span.first(num_bytes_read),
      num_results.data());
  }

  auto host_num_results = num_results.value(stream);

  stream.synchronize();

  std::cout << "num results: " << host_num_results << std::endl;

  // TODO: call state finalization kernels

  CUDF_FAIL();
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::input_stream& input,
                                              std::vector<std::string> const& delimeters,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeters, rmm::cuda_stream_default, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
