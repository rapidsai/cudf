#include <cudf/column/column.hpp>
#include <cudf/io/text/input_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <bitset>
#include <iostream>
#include <memory>

namespace {

__global__ void multibyte_split_kernel(cudf::device_span<char> data)
{
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < data.size()) {
    printf("bid(%i) tid(%i) %c\n",
           static_cast<int32_t>(blockIdx.x),
           static_cast<int32_t>(threadIdx.x),
           data[thread_idx]);
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
  auto constexpr bytes_per_thread  = 32;
  auto constexpr threads_per_block = 1024;
  auto constexpr blocks_per_pass   = 1;
  auto constexpr bytes_per_pass    = bytes_per_thread * threads_per_block * blocks_per_pass;

  auto input_buffer_a = rmm::device_uvector<char>(bytes_per_pass, stream);
  auto stream_a       = stream;

  auto input_buffer_b = rmm::device_uvector<char>(bytes_per_pass, stream);
  auto stream_b       = stream;

  uint32_t bytes_read = 0;

  while (true) {
    stream_a.synchronize();

    auto bytes_read = input.readsome(input_buffer_a, stream_a);

    if (bytes_read == 0) {
      break;  // nothing left to process.
    }

    multibyte_split_kernel<<<blocks_per_pass, threads_per_block, 0, stream_a.value()>>>(
      cudf::device_span<char>(input_buffer_a).first(bytes_read));

    std::swap(stream_a, stream_b);
    std::swap(input_buffer_a, input_buffer_b);
  }

  stream_b.synchronize();

  CUDF_FAIL();
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(cudf::io::text::input_stream& input,
                                              std::string delimeter,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::multibyte_split(input, delimeter, {}, mr);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
