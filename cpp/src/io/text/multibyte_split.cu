#include <cudf/column/column.hpp>
#include <cudf/io/text/input_stream.hpp>
#include <cudf/io/text/trie.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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

template <uint32_t BYTES_PER_THREAD>
__global__ void multibyte_split_kernel(cudf::device_span<char> data)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto const data_begin = thread_idx * BYTES_PER_THREAD;
  auto data_end         = data_begin + BYTES_PER_THREAD;

  if (data_end > data.size()) { data_end = data.size(); }

  if (data_end < data.size()) {  //
    printf("bid(%i) tid(%i)    : whole\n", blockIdx.x, threadIdx.x);
  } else if (data_begin < data.size()) {
    printf("bid(%i) tid(%i)    : partial\n", blockIdx.x, threadIdx.x);
  }

  for (uint32_t i = data_begin; i < data_end; i++) {
    printf("bid(%i) tid(%i) %3i: %c\n", blockIdx.x, threadIdx.x, i, data[i]);
  }
}

}  // namespace

namespace cudf {
namespace io {
namespace text {
namespace detail {

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::input_stream& input,
                                              std::string delimeter,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto constexpr BYTES_PER_THREAD = 32;
  auto constexpr THREADS_PER_TILE = 256;
  auto constexpr BYTES_PER_TILE   = BYTES_PER_THREAD * THREADS_PER_TILE;
  auto constexpr TILES_PER_CHUNK  = 1024;
  auto constexpr BYTES_PER_CHUNK  = BYTES_PER_TILE * TILES_PER_CHUNK;

  auto input_buffer     = rmm::device_uvector<char>(BYTES_PER_CHUNK, stream);
  auto const input_span = cudf::device_span<char>(input_buffer);

  // TODO: call state initalization kernels

  while (true) {
    uint32_t num_bytes_read = input.readsome(input_span, stream);

    if (num_bytes_read == 0) {
      // if there's no more data to read, we're done.
      break;
    }

    auto num_tiles = ceil_div(num_bytes_read, BYTES_PER_TILE);

    auto kernel = multibyte_split_kernel<BYTES_PER_THREAD>;
    kernel<<<num_tiles, THREADS_PER_TILE, 0, stream.value()>>>(input_span.first(num_bytes_read));
  }

  // TODO: call state finalization kernels

  CUDF_FAIL();
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::input_stream& input,
                                              std::string delimeter,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeter, rmm::cuda_stream_default, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
