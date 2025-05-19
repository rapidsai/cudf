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

#include <cudf/io/text/detail/bgzip_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <zlib.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>

namespace cudf::io::text::detail::bgzip {
namespace {

template <typename IntType>
IntType read_int(char* data)
{
  IntType result{};
  // we assume little-endian
  std::memcpy(&result, &data[0], sizeof(result));
  return result;
}

template <typename T>
void write_int(std::ostream& output_stream, T val)
{
  std::array<char, sizeof(T)> bytes{};
  // we assume little-endian
  std::memcpy(&bytes[0], &val, sizeof(T));
  output_stream.write(bytes.data(), bytes.size());
}

}  // namespace

std::array<char, 4> constexpr extra_blocklen_field_header{{66, 67, 2, 0}};

header read_header(std::istream& input_stream)
{
  std::array<char, 12> buffer{};
  input_stream.read(buffer.data(), sizeof(buffer));
  std::array<uint8_t, 4> constexpr expected_header{{31, 139, 8, 4}};
  CUDF_EXPECTS(
    std::equal(
      expected_header.begin(), expected_header.end(), reinterpret_cast<uint8_t*>(buffer.data())),
    "malformed BGZIP header");
  // we ignore the remaining bytes of the fixed header, since they don't matter to us
  auto const extra_length = read_int<uint16_t>(&buffer[10]);
  uint16_t extra_offset{};
  // read all the extra subfields
  while (extra_offset < extra_length) {
    auto const remaining_size = extra_length - extra_offset;
    CUDF_EXPECTS(remaining_size >= 4, "invalid extra field length");
    // a subfield consists of 2 identifier bytes and a uint16 length
    // 66/67 identifies a BGZIP block size field, we skip all other fields
    input_stream.read(buffer.data(), 4);
    extra_offset += 4;
    auto const subfield_size = read_int<uint16_t>(&buffer[2]);
    if (buffer[0] == extra_blocklen_field_header[0] &&
        buffer[1] == extra_blocklen_field_header[1]) {
      // the block size subfield contains a single uint16 value, which is block_size - 1
      CUDF_EXPECTS(
        buffer[2] == extra_blocklen_field_header[2] && buffer[3] == extra_blocklen_field_header[3],
        "malformed BGZIP extra subfield");
      input_stream.read(buffer.data(), sizeof(uint16_t));
      input_stream.seekg(remaining_size - 6, std::ios_base::cur);
      auto const block_size_minus_one = read_int<uint16_t>(&buffer[0]);
      return {block_size_minus_one + 1, extra_length};
    } else {
      input_stream.seekg(subfield_size, std::ios_base::cur);
      extra_offset += subfield_size;
    }
  }
  CUDF_FAIL("missing BGZIP size extra subfield");
}

footer read_footer(std::istream& input_stream)
{
  std::array<char, 8> buffer{};
  input_stream.read(buffer.data(), sizeof(buffer));
  return {read_int<uint32_t>(&buffer[0]), read_int<uint32_t>(&buffer[4])};
}

void write_footer(std::ostream& output_stream, host_span<char const> data)
{
  // compute crc32 with zlib, this allows checking the generated files with external tools
  write_int<uint32_t>(output_stream, crc32(0, (unsigned char*)data.data(), data.size()));
  write_int<uint32_t>(output_stream, data.size());
}

void write_header(std::ostream& output_stream,
                  uint16_t compressed_size,
                  host_span<char const> pre_size_subfield,
                  host_span<char const> post_size_subfield)
{
  std::array<uint8_t, 10> constexpr header_data{{
    31,   // magic number
    139,  // magic number
    8,    // compression type: deflate
    4,    // flags: extra header
    0,    // mtime
    0,    // mtime
    0,    // mtime
    0,    // mtime: irrelevant
    4,    // xfl: irrelevant
    3     // OS: irrelevant
  }};
  output_stream.write(reinterpret_cast<char const*>(header_data.data()), header_data.size());
  auto const extra_size = pre_size_subfield.size() + extra_blocklen_field_header.size() +
                          sizeof(uint16_t) + post_size_subfield.size();
  auto const block_size =
    header_data.size() + sizeof(uint16_t) + extra_size + compressed_size + 2 * sizeof(uint32_t);
  write_int<uint16_t>(output_stream, extra_size);
  output_stream.write(pre_size_subfield.data(), pre_size_subfield.size());
  output_stream.write(extra_blocklen_field_header.data(), extra_blocklen_field_header.size());
  CUDF_EXPECTS(block_size - 1 <= std::numeric_limits<uint16_t>::max(), "block size overflow");
  write_int<uint16_t>(output_stream, block_size - 1);
  output_stream.write(post_size_subfield.data(), post_size_subfield.size());
}

void write_uncompressed_block(std::ostream& output_stream,
                              host_span<char const> data,
                              host_span<char const> pre_size_subfields,
                              host_span<char const> post_size_subfields)
{
  CUDF_EXPECTS(data.size() <= std::numeric_limits<uint16_t>::max(), "data size overflow");
  write_header(output_stream, data.size() + 5, pre_size_subfields, post_size_subfields);
  write_int<uint8_t>(output_stream, 1);
  write_int<uint16_t>(output_stream, data.size());
  write_int<uint16_t>(output_stream, ~static_cast<uint16_t>(data.size()));
  output_stream.write(data.data(), data.size());
  write_footer(output_stream, data);
}

void write_compressed_block(std::ostream& output_stream,
                            host_span<char const> data,
                            host_span<char const> pre_size_subfields,
                            host_span<char const> post_size_subfields)
{
  CUDF_EXPECTS(data.size() <= std::numeric_limits<uint16_t>::max(), "data size overflow");
  z_stream deflate_stream{};
  // let's make sure we have enough space to store the data
  std::vector<char> compressed_out(data.size() * 2 + 256);
  deflate_stream.next_in   = reinterpret_cast<unsigned char*>(const_cast<char*>(data.data()));
  deflate_stream.avail_in  = data.size();
  deflate_stream.next_out  = reinterpret_cast<unsigned char*>(compressed_out.data());
  deflate_stream.avail_out = compressed_out.size();
  CUDF_EXPECTS(
    deflateInit2(&deflate_stream,        // stream
                 Z_DEFAULT_COMPRESSION,  // compression level
                 Z_DEFLATED,             // method
                 -15,  // log2 of window size (negative value means no ZLIB header/footer)
                 9,    // mem level: best performance/most memory usage for compression
                 Z_DEFAULT_STRATEGY  // strategy
                 ) == Z_OK,
    "deflateInit failed");
  CUDF_EXPECTS(deflate(&deflate_stream, Z_FINISH) == Z_STREAM_END, "deflate failed");
  CUDF_EXPECTS(deflateEnd(&deflate_stream) == Z_OK, "deflateEnd failed");
  write_header(output_stream, deflate_stream.total_out, pre_size_subfields, post_size_subfields);
  output_stream.write(compressed_out.data(), deflate_stream.total_out);
  write_footer(output_stream, data);
}

}  // namespace cudf::io::text::detail::bgzip
