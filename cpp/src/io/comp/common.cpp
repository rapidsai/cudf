/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_internal.hpp"
#include "nvcomp_adapter.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/types.hpp>

namespace cudf::io::detail {

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

}  // namespace cudf::io::detail
