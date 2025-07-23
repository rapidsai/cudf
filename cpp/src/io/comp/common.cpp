/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "common_internal.hpp"
#include "nvcomp_adapter.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/types.hpp>

namespace cudf::io::detail {

[[nodiscard]] std::optional<nvcomp::compression_type> to_nvcomp_compression(
  compression_type compression)
{
  switch (compression) {
    case compression_type::GZIP: return nvcomp::compression_type::GZIP;
    case compression_type::LZ4: return nvcomp::compression_type::LZ4;
    case compression_type::SNAPPY: return nvcomp::compression_type::SNAPPY;
    case compression_type::ZLIB: return nvcomp::compression_type::DEFLATE;
    case compression_type::ZSTD: return nvcomp::compression_type::ZSTD;
    default: return std::nullopt;
  }
}

[[nodiscard]] std::string compression_type_name(compression_type compression)
{
  switch (compression) {
    case compression_type::NONE: return "NONE";
    case compression_type::AUTO: return "AUTO";
    case compression_type::SNAPPY: return "SNAPPY";
    case compression_type::GZIP: return "GZIP";
    case compression_type::BZIP2: return "BZIP2";
    case compression_type::BROTLI: return "BROTLI";
    case compression_type::ZIP: return "ZIP";
    case compression_type::XZ: return "XZ";
    case compression_type::ZLIB: return "ZLIB";
    case compression_type::LZ4: return "LZ4";
    case compression_type::LZO: return "LZO";
    case compression_type::ZSTD: return "ZSTD";
    default:
      CUDF_FAIL("Invalid compression type: " + std::to_string(static_cast<int>(compression)));
  }
}

[[nodiscard]] size_t find_split_index(device_span<device_span<uint8_t const> const> inputs,
                                      host_engine_state host_state,
                                      size_t auto_mode_threshold,
                                      size_t hybrid_mode_target_ratio,
                                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (host_state == host_engine_state::OFF or inputs.empty()) { return 0; }
  if (host_state == host_engine_state::ON) { return inputs.size(); }

  if (host_state == host_engine_state::AUTO) {
    return inputs.size() < auto_mode_threshold ? inputs.size() : 0;
  }

  if (host_state == host_engine_state::HYBRID) {
    auto const h_inputs    = cudf::detail::make_host_vector(inputs, stream);
    size_t total_host_size = 0;
    for (size_t i = 0; i < h_inputs.size(); ++i) {
      if (total_host_size >= hybrid_mode_target_ratio * h_inputs[i].size()) { return i; }
      total_host_size += h_inputs[i].size();
    }
    return inputs.size();  // No split
  }

  CUDF_FAIL("Invalid host engine state for compression: " +
            std::to_string(static_cast<uint8_t>(host_state)));
}

}  // namespace cudf::io::detail
