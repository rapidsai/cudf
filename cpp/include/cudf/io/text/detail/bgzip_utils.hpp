/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <fstream>

namespace CUDF_EXPORT cudf {
namespace io::text::detail::bgzip {

struct header {
  int block_size;
  int extra_length;
  [[nodiscard]] int data_size() const { return block_size - extra_length - 20; }
};

struct footer {
  uint32_t crc;
  uint32_t decompressed_size;
};

/**
 * @brief Reads the full BGZIP header from the given input stream. Afterwards, the stream position
 *        is at the first data byte.
 *
 * @param input_stream The input stream
 * @return The header storing the compressed size and extra subfield length
 */
header read_header(std::istream& input_stream);

/**
 * @brief Reads the full BGZIP footer from the given input stream. Afterwards, the stream position
 *        is after the last footer byte.
 *
 * @param input_stream The input stream
 * @return The footer storing uncompressed size and CRC32
 */
footer read_footer(std::istream& input_stream);

/**
 * @brief Writes a header for data of the given compressed size to the given stream.
 *
 * @param output_stream The output stream
 * @param compressed_size The size of the compressed data
 * @param pre_size_subfields Any GZIP extra subfields (need to be valid) to be placed before the
 *                           BGZIP block size subfield
 * @param post_size_subfields Any subfields to be placed after the BGZIP block size subfield
 */
void write_header(std::ostream& output_stream,
                  uint16_t compressed_size,
                  host_span<char const> pre_size_subfields,
                  host_span<char const> post_size_subfields);

/**
 * @brief Writes a footer for the given uncompressed data to the given stream.
 *
 * @param output_stream The output stream
 * @param data The data for which uncompressed size and CRC32 will be computed and written
 */
void write_footer(std::ostream& output_stream, host_span<char const> data);

/**
 * @brief Writes the given data to the given stream as an uncompressed deflate block with BZGIP
 *        header and footer.
 *
 * @param output_stream The output stream
 * @param data The uncompressed data
 * @param pre_size_subfields Any GZIP extra subfields (need to be valid) to be placed before the
 *                           BGZIP block size subfield
 * @param post_size_subfields Any subfields to be placed after the BGZIP block size subfield
 */
void write_uncompressed_block(std::ostream& output_stream,
                              host_span<char const> data,
                              host_span<char const> pre_size_subfields  = {},
                              host_span<char const> post_size_subfields = {});

/**
 * @brief Writes the given data to the given stream as a compressed deflate block with BZGIP
 *        header and footer.
 *
 * @param output_stream The output stream
 * @param data The uncompressed data
 * @param pre_size_subfields Any GZIP extra subfields (need to be valid) to be placed before the
 *                           BGZIP block size subfield
 * @param post_size_subfields Any subfields to be placed after the BGZIP block size subfield
 */
void write_compressed_block(std::ostream& output_stream,
                            host_span<char const> data,
                            host_span<char const> pre_size_subfields  = {},
                            host_span<char const> post_size_subfields = {});

}  // namespace io::text::detail::bgzip
}  // namespace CUDF_EXPORT cudf
