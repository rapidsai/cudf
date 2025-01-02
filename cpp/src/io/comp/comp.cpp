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

#include "io/utilities/hostdevice_vector.hpp"
#include "nvcomp_adapter.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <zlib.h>  // GZIP compression

namespace cudf::io::detail {

namespace {

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

}  // namespace

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

}  // namespace cudf::io::detail
