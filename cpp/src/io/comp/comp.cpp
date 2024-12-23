/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "comp.hpp"

#include "gpuinflate.hpp"
#include "io/utilities/getenv_or.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "nvcomp_adapter.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <BS_thread_pool.hpp>
#include <zlib.h>  // GZIP compression

namespace cudf::io::detail {

namespace {

auto& h_comp_pool()
{
  static BS::thread_pool pool(std::thread::hardware_concurrency());
  return pool;
}

std::optional<nvcomp::compression_type> to_nvcomp_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::SNAPPY: return nvcomp::compression_type::SNAPPY;
    case compression_type::ZSTD: return nvcomp::compression_type::ZSTD;
    case compression_type::LZ4: return nvcomp::compression_type::LZ4;
    case compression_type::ZLIB: return nvcomp::compression_type::DEFLATE;
    default: return std::nullopt;
  }
}

/**
 * @brief GZIP host compressor (includes header)
 */
std::vector<std::uint8_t> compress_gzip(host_span<uint8_t const> src)
{
  z_stream zs;
  zs.zalloc   = Z_NULL;
  zs.zfree    = Z_NULL;
  zs.opaque   = Z_NULL;
  zs.avail_in = src.size();
  zs.next_in  = reinterpret_cast<unsigned char*>(const_cast<unsigned char*>(src.data()));

  std::vector<uint8_t> dst;
  zs.avail_out = 0;
  zs.next_out  = nullptr;

  constexpr int windowbits    = 15;
  constexpr int gzip_encoding = 16;
  int ret                     = deflateInit2(
    &zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowbits | gzip_encoding, 8, Z_DEFAULT_STRATEGY);
  CUDF_EXPECTS(ret == Z_OK, "GZIP DEFLATE compression initialization failed.");

  uint32_t const estcomplen = deflateBound(&zs, src.size());
  dst.resize(estcomplen);
  zs.avail_out = estcomplen;
  zs.next_out  = dst.data();

  ret = deflate(&zs, Z_FINISH);
  CUDF_EXPECTS(ret == Z_STREAM_END, "GZIP DEFLATE compression failed due to insufficient space!");
  dst.resize(std::distance(dst.data(), zs.next_out));

  ret = deflateEnd(&zs);
  CUDF_EXPECTS(ret == Z_OK, "GZIP DEFLATE compression failed at deallocation");

  return dst;
}

/**
 * @brief SNAPPY device compressor
 */
std::vector<std::uint8_t> compress_snappy(host_span<uint8_t const> src,
                                          rmm::cuda_stream_view stream)
{
  auto const d_src =
    cudf::detail::make_device_uvector_async(src, stream, cudf::get_current_device_resource_ref());
  cudf::detail::hostdevice_vector<device_span<uint8_t const>> inputs(1, stream);
  inputs[0] = d_src;
  inputs.host_to_device_async(stream);

  auto dst_size = compress_max_output_chunk_size(nvcomp::compression_type::SNAPPY, src.size());
  rmm::device_uvector<uint8_t> d_dst(dst_size, stream);
  cudf::detail::hostdevice_vector<device_span<uint8_t>> outputs(1, stream);
  outputs[0] = d_dst;
  outputs.host_to_device_async(stream);

  cudf::detail::hostdevice_vector<compression_result> hd_status(1, stream);
  hd_status[0] = {};
  hd_status.host_to_device_async(stream);

  nvcomp::batched_compress(nvcomp::compression_type::SNAPPY, inputs, outputs, hd_status, stream);

  hd_status.device_to_host_sync(stream);
  CUDF_EXPECTS(hd_status[0].status == compression_status::SUCCESS, "snappy compression failed");
  return cudf::detail::make_std_vector_sync<uint8_t>(d_dst, stream);
}

void device_compress(compression_type compression,
                     device_span<device_span<uint8_t const> const> inputs,
                     device_span<device_span<uint8_t> const> outputs,
                     device_span<compression_result> results,
                     rmm::cuda_stream_view stream)
{
  auto const nvcomp_type = to_nvcomp_compression(compression);
  auto nvcomp_disabled   = nvcomp_type.has_value() ? nvcomp::is_compression_disabled(*nvcomp_type)
                                                   : "invalid compression type";
  if (not nvcomp_disabled) {
    return nvcomp::batched_compress(*nvcomp_type, inputs, outputs, results, stream);
  }

  switch (compression) {
    case compression_type::SNAPPY: return gpu_snap(inputs, outputs, results, stream);
    default: CUDF_FAIL("Compression error: " + nvcomp_disabled.value());
  }
}

void host_compress(compression_type compression,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<compression_result> results,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (compression == compression_type::NONE) { return; }
  auto const num_blocks = inputs.size();
  auto h_results        = cudf::detail::make_host_vector<compression_result>(num_blocks, stream);
  auto const h_inputs   = cudf::detail::make_host_vector_async(inputs, stream);
  auto const h_outputs  = cudf::detail::make_host_vector_async(outputs, stream);
  stream.synchronize();

  std::vector<std::future<size_t>> tasks;
  auto streams = cudf::detail::fork_streams(stream, h_comp_pool().get_thread_count());
  for (size_t i = 0; i < num_blocks; ++i) {
    auto cur_stream = streams[i % streams.size()];
    auto task = [d_in = h_inputs[i], d_out = h_outputs[i], cur_stream, compression]() -> size_t {
      auto const h_in  = cudf::detail::make_host_vector_sync(d_in, cur_stream);
      auto const h_out = compress(compression, h_in, cur_stream);
      cudf::detail::cuda_memcpy<uint8_t>(d_out.subspan(0, h_out.size()), h_out, cur_stream);
      return h_out.size();
    };
    tasks.emplace_back(h_comp_pool().submit_task(std::move(task)));
  }

  for (auto i = 0ul; i < num_blocks; ++i) {
    h_results[i] = {tasks[i].get(), compression_status::SUCCESS};
  }
  cudf::detail::cuda_memcpy_async<compression_result>(results, h_results, stream);
}

[[nodiscard]] bool host_compression_supported(compression_type compression)
{
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool device_compression_supported(compression_type compression)
{
  auto const nvcomp_type = to_nvcomp_compression(compression);
  switch (compression) {
    case compression_type::LZ4:
    case compression_type::ZLIB:
    case compression_type::ZSTD: return not nvcomp::is_compression_disabled(nvcomp_type.value());
    case compression_type::SNAPPY:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool use_host_compression(
  compression_type compression,
  [[maybe_unused]] device_span<device_span<uint8_t const> const> inputs,
  [[maybe_unused]] device_span<device_span<uint8_t> const> outputs)
{
  CUDF_EXPECTS(
    not host_compression_supported(compression) or device_compression_supported(compression),
    "Unsupported compression type");
  if (not host_compression_supported(compression)) { return false; }
  if (not device_compression_supported(compression)) { return true; }
  // If both host and device compression are supported, use the host if the env var is set
  return getenv_or("LIBCUDF_USE_HOST_COMPRESSION", 0);
}

}  // namespace

std::optional<size_t> compress_max_allowed_block_size(compression_type compression)
{
  if (auto nvcomp_type = to_nvcomp_compression(compression);
      nvcomp_type.has_value() and not nvcomp::is_compression_disabled(*nvcomp_type)) {
    return nvcomp::compress_max_allowed_chunk_size(*nvcomp_type);
  }
  return std::nullopt;
}

[[nodiscard]] size_t compress_required_block_alignment(compression_type compression)
{
  auto nvcomp_type = to_nvcomp_compression(compression);
  if (compression == compression_type::NONE or not nvcomp_type.has_value() or
      nvcomp::is_compression_disabled(*nvcomp_type)) {
    return 1ul;
  }

  return nvcomp::required_alignment(*nvcomp_type);
}

[[nodiscard]] size_t max_compressed_size(compression_type compression, uint32_t uncompressed_size)
{
  if (compression == compression_type::NONE) { return uncompressed_size; }

  if (auto nvcomp_type = to_nvcomp_compression(compression); nvcomp_type.has_value()) {
    return nvcomp::compress_max_output_chunk_size(*nvcomp_type, uncompressed_size);
  }
  CUDF_FAIL("Unsupported compression type");
}

std::vector<std::uint8_t> compress(compression_type compression,
                                   host_span<uint8_t const> src,
                                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  switch (compression) {
    case compression_type::GZIP: return compress_gzip(src);
    case compression_type::SNAPPY: return compress_snappy(src, stream);
    default: CUDF_FAIL("Unsupported compression type");
  }
}

void compress(compression_type compression,
              device_span<device_span<uint8_t const> const> inputs,
              device_span<device_span<uint8_t> const> outputs,
              device_span<compression_result> results,
              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (use_host_compression(compression, inputs, outputs)) {
    return host_compress(compression, inputs, outputs, results, stream);
  } else {
    return device_compress(compression, inputs, outputs, results, stream);
  }
}

}  // namespace cudf::io::detail
