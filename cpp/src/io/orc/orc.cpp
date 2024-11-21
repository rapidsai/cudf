/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "orc.hpp"

#include "orc_field_reader.hpp"
#include "orc_field_writer.hpp"

#include <cudf/lists/lists_column_view.hpp>

#include <thrust/tabulate.h>

#include <string>

namespace cudf::io::orc {

namespace {
[[nodiscard]] constexpr uint32_t varint_size(uint64_t val)
{
  auto len = 1u;
  while (val > 0x7f) {
    val >>= 7;
    ++len;
  }
  return len;
}
}  // namespace

uint32_t ProtobufReader::read_field_size(uint8_t const* end)
{
  auto const size = get<uint32_t>();
  CUDF_EXPECTS(size <= static_cast<uint32_t>(end - m_cur), "Protobuf parsing out of bounds");
  return size;
}

void ProtobufReader::skip_struct_field(int t)
{
  switch (t) {
    case ProtofType::VARINT: get<uint32_t>(); break;
    case ProtofType::FIXED64: skip_bytes(8); break;
    case ProtofType::FIXEDLEN: skip_bytes(get<uint32_t>()); break;
    case ProtofType::FIXED32: skip_bytes(4); break;
    default: break;
  }
}

void ProtobufReader::read(PostScript& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.footerLength),
                       field_reader(2, s.compression),
                       field_reader(3, s.compressionBlockSize),
                       packed_field_reader(4, s.version),
                       field_reader(5, s.metadataLength),
                       field_reader(6, s.writerVersion),
                       field_reader(8000, s.magic));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(Footer& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.headerLength),
                       field_reader(2, s.contentLength),
                       field_reader(3, s.stripes),
                       field_reader(4, s.types),
                       field_reader(5, s.metadata),
                       field_reader(6, s.numberOfRows),
                       raw_field_reader(7, s.statistics),
                       field_reader(8, s.rowIndexStride),
                       field_reader(9, s.writer));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeInformation& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.offset),
                       field_reader(2, s.indexLength),
                       field_reader(3, s.dataLength),
                       field_reader(4, s.footerLength),
                       field_reader(5, s.numberOfRows));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(SchemaType& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.kind),
                       packed_field_reader(2, s.subtypes),
                       field_reader(3, s.fieldNames),
                       field_reader(4, s.maximumLength),
                       field_reader(5, s.precision),
                       field_reader(6, s.scale));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(UserMetadataItem& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.name), field_reader(2, s.value));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeFooter& s, size_t maxlen)
{
  auto op = std::tuple(
    field_reader(1, s.streams), field_reader(2, s.columns), field_reader(3, s.writerTimezone));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(Stream& s, size_t maxlen)
{
  auto op =
    std::tuple(field_reader(1, s.kind), field_reader(2, s.column_id), field_reader(3, s.length));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(ColumnEncoding& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.kind), field_reader(2, s.dictionarySize));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(integer_statistics& s, size_t maxlen)
{
  auto op =
    std::tuple(field_reader(1, s.minimum), field_reader(2, s.maximum), field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(double_statistics& s, size_t maxlen)
{
  auto op =
    std::tuple(field_reader(1, s.minimum), field_reader(2, s.maximum), field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(string_statistics& s, size_t maxlen)
{
  auto op =
    std::tuple(field_reader(1, s.minimum), field_reader(2, s.maximum), field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(bucket_statistics& s, size_t maxlen)
{
  auto op = std::tuple(packed_field_reader(1, s.count));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(decimal_statistics& s, size_t maxlen)
{
  auto op =
    std::tuple(field_reader(1, s.minimum), field_reader(2, s.maximum), field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(date_statistics& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.minimum), field_reader(2, s.maximum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(binary_statistics& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(timestamp_statistics& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.minimum),
                       field_reader(2, s.maximum),
                       field_reader(3, s.minimum_utc),
                       field_reader(4, s.maximum_utc),
                       field_reader(5, s.minimum_nanos),
                       field_reader(6, s.maximum_nanos));
  function_builder(s, maxlen, op);

  // Adjust nanoseconds because they are encoded as (value + 1)
  // Range [1, 1000'000] is translated here to [0, 999'999]
  if (s.minimum_nanos.has_value()) {
    auto& min_nanos = s.minimum_nanos.value();
    CUDF_EXPECTS(min_nanos >= 1 and min_nanos <= 1000'000, "Invalid minimum nanoseconds");
    --min_nanos;
  }
  if (s.maximum_nanos.has_value()) {
    auto& max_nanos = s.maximum_nanos.value();
    CUDF_EXPECTS(max_nanos >= 1 and max_nanos <= 1000'000, "Invalid maximum nanoseconds");
    --max_nanos;
  }
}

void ProtobufReader::read(column_statistics& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.number_of_values),
                       field_reader(2, s.int_stats),
                       field_reader(3, s.double_stats),
                       field_reader(4, s.string_stats),
                       field_reader(5, s.bucket_stats),
                       field_reader(6, s.decimal_stats),
                       field_reader(7, s.date_stats),
                       field_reader(8, s.binary_stats),
                       field_reader(9, s.timestamp_stats),
                       field_reader(10, s.has_null));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeStatistics& s, size_t maxlen)
{
  auto op = std::tuple(raw_field_reader(1, s.colStats));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(Metadata& s, size_t maxlen)
{
  auto op = std::tuple(field_reader(1, s.stripeStats));
  function_builder(s, maxlen, op);
}

/**
 * @brief Add a single rowIndexEntry, negative input values treated as not present
 */
void ProtobufWriter::put_row_index_entry(int32_t present_blk,
                                         int32_t present_ofs,
                                         int32_t data_blk,
                                         int32_t data_ofs,
                                         int32_t data2_blk,
                                         int32_t data2_ofs,
                                         TypeKind kind,
                                         ColStatsBlob const* stats)
{
  ProtobufWriter position_writer;
  auto const positions_size_offset = position_writer.put_uint(
    encode_field_number(1, ProtofType::FIXEDLEN));  // 1:positions[packed=true]
  position_writer.put_byte(0xcd);                   // positions size placeholder
  uint32_t positions_size = 0;
  if (present_blk >= 0) positions_size += position_writer.put_uint(present_blk);
  if (present_ofs >= 0) {
    positions_size += position_writer.put_uint(present_ofs);
    positions_size += position_writer.put_byte(0);  // run pos = 0
    positions_size += position_writer.put_byte(0);  // bit pos = 0
  }
  if (data_blk >= 0) { positions_size += position_writer.put_uint(data_blk); }
  if (data_ofs >= 0) {
    positions_size += position_writer.put_uint(data_ofs);
    if (kind != STRING && kind != FLOAT && kind != DOUBLE && kind != DECIMAL) {
      // RLE run pos always zero (assumes RLE aligned with row index boundaries)
      positions_size += position_writer.put_byte(0);
      if (kind == BOOLEAN) {
        // bit position in byte, always zero
        positions_size += position_writer.put_byte(0);
      }
    }
  }
  // INT kind can be passed in to bypass 2nd stream index (dictionary length streams)
  if (kind != INT) {
    if (data2_blk >= 0) { positions_size += position_writer.put_uint(data2_blk); }
    if (data2_ofs >= 0) {
      positions_size += position_writer.put_uint(data2_ofs);
      // RLE run pos always zero (assumes RLE aligned with row index boundaries)
      positions_size += position_writer.put_byte(0);
    }
  }

  // size of the field 1
  position_writer.buffer()[positions_size_offset] = static_cast<uint8_t>(positions_size);

  auto const stats_size = (stats == nullptr)
                            ? 0
                            : varint_size(encode_field_number<decltype(*stats)>(2)) +
                                varint_size(stats->size()) + stats->size();
  auto const entry_size = position_writer.size() + stats_size;

  // 1:RowIndex.entry
  put_uint(encode_field_number(1, ProtofType::FIXEDLEN));
  put_uint(entry_size);
  put_bytes<uint8_t>(position_writer.buffer());

  if (stats != nullptr) {
    put_uint(encode_field_number<decltype(*stats)>(2));  // 2: statistics
    // Statistics field contains its length as varint and dtype specific data (encoded on the GPU)
    put_uint(stats->size());
    put_bytes<typename ColStatsBlob::value_type>(*stats);
  }
}

size_t ProtobufWriter::write(PostScript const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.footerLength);
  w.field_uint(2, s.compression);
  if (s.compression != NONE) { w.field_uint(3, s.compressionBlockSize); }
  w.field_packed_uint(4, s.version);
  w.field_uint(5, s.metadataLength);
  if (s.writerVersion) w.field_uint(6, *s.writerVersion);
  w.field_blob(8000, s.magic);
  return w.value();
}

size_t ProtobufWriter::write(Footer const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.headerLength);
  w.field_uint(2, s.contentLength);
  w.field_repeated_struct(3, s.stripes);
  w.field_repeated_struct(4, s.types);
  w.field_repeated_struct(5, s.metadata);
  w.field_uint(6, s.numberOfRows);
  w.field_repeated_struct_blob(7, s.statistics);
  w.field_uint(8, s.rowIndexStride);
  if (s.writer) w.field_uint(9, *s.writer);
  return w.value();
}

size_t ProtobufWriter::write(StripeInformation const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.offset);
  w.field_uint(2, s.indexLength);
  w.field_uint(3, s.dataLength);
  w.field_uint(4, s.footerLength);
  w.field_uint(5, s.numberOfRows);
  return w.value();
}

size_t ProtobufWriter::write(SchemaType const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.kind);
  w.field_packed_uint(2, s.subtypes);
  w.field_repeated_string(3, s.fieldNames);
  // w.field_uint(4, s.maximumLength);
  if (s.precision) w.field_uint(5, *s.precision);
  if (s.scale) w.field_uint(6, *s.scale);
  return w.value();
}

size_t ProtobufWriter::write(UserMetadataItem const& s)
{
  ProtobufFieldWriter w(this);
  w.field_blob(1, s.name);
  w.field_blob(2, s.value);
  return w.value();
}

size_t ProtobufWriter::write(StripeFooter const& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct(1, s.streams);
  w.field_repeated_struct(2, s.columns);
  if (s.writerTimezone != "") { w.field_blob(3, s.writerTimezone); }
  return w.value();
}

size_t ProtobufWriter::write(Stream const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.kind);
  if (s.column_id) w.field_uint(2, *s.column_id);
  w.field_uint(3, s.length);
  return w.value();
}

size_t ProtobufWriter::write(ColumnEncoding const& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.kind);
  if (s.kind == DICTIONARY || s.kind == DICTIONARY_V2) { w.field_uint(2, s.dictionarySize); }
  return w.value();
}

size_t ProtobufWriter::write(StripeStatistics const& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct_blob(1, s.colStats);
  return w.value();
}

size_t ProtobufWriter::write(Metadata const& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct(1, s.stripeStats);
  return w.value();
}

OrcDecompressor::OrcDecompressor(CompressionKind kind, uint64_t block_size)
  : m_blockSize(block_size)
{
  switch (kind) {
    case NONE:
      _compression   = compression_type::NONE;
      m_log2MaxRatio = 0;
      break;
    case ZLIB:
      _compression   = compression_type::ZLIB;
      m_log2MaxRatio = 11;  // < 2048:1
      break;
    case SNAPPY:
      _compression   = compression_type::SNAPPY;
      m_log2MaxRatio = 5;  // < 32:1
      break;
    case LZO: _compression = compression_type::LZO; break;
    case LZ4: _compression = compression_type::LZ4; break;
    case ZSTD:
      m_log2MaxRatio = 15;
      _compression   = compression_type::ZSTD;
      break;
    default: CUDF_FAIL("Invalid compression type");
  }
}

host_span<uint8_t const> OrcDecompressor::decompress_blocks(host_span<uint8_t const> src,
                                                            rmm::cuda_stream_view stream)
{
  // If uncompressed, just pass-through the input
  if (src.empty() or _compression == compression_type::NONE) { return src; }

  constexpr size_t header_size = 3;
  CUDF_EXPECTS(src.size() >= header_size, "Total size is less than the 3-byte header");

  // First, scan the input for the number of blocks and worst-case output size
  size_t max_dst_length = 0;
  for (size_t i = 0; i + header_size < src.size();) {
    uint32_t block_len         = src[i] | (src[i + 1] << 8) | (src[i + 2] << 16);
    auto const is_uncompressed = static_cast<bool>(block_len & 1);
    i += header_size;
    block_len >>= 1;
    if (is_uncompressed) {
      // Uncompressed block
      max_dst_length += block_len;
    } else {
      max_dst_length += m_blockSize;
    }
    i += block_len;
    CUDF_EXPECTS(i <= src.size() and block_len <= m_blockSize, "Error in decompression");
  }
  // Check if we have a single uncompressed block, or no blocks
  if (max_dst_length < m_blockSize) { return src.subspan(header_size, src.size() - header_size); }

  m_buf.resize(max_dst_length);
  size_t dst_length = 0;
  for (size_t i = 0; i + header_size < src.size();) {
    uint32_t block_len         = src[i] | (src[i + 1] << 8) | (src[i + 2] << 16);
    auto const is_uncompressed = static_cast<bool>(block_len & 1);
    i += header_size;
    block_len >>= 1;
    if (is_uncompressed) {
      // Uncompressed block
      memcpy(m_buf.data() + dst_length, src.data() + i, block_len);
      dst_length += block_len;
    } else {
      // Compressed block
      dst_length += cudf::io::detail::decompress(
        _compression, src.subspan(i, block_len), {m_buf.data() + dst_length, m_blockSize}, stream);
    }
    i += block_len;
  }

  m_buf.resize(dst_length);
  return m_buf;
}

metadata::metadata(datasource* const src, rmm::cuda_stream_view stream) : source(src)
{
  auto const len         = source->size();
  auto const max_ps_size = std::min(len, static_cast<size_t>(256));

  // Read uncompressed postscript section (max 255 bytes + 1 byte for length)
  auto buffer            = source->host_read(len - max_ps_size, max_ps_size);
  size_t const ps_length = buffer->data()[max_ps_size - 1];
  uint8_t const* ps_data = &buffer->data()[max_ps_size - ps_length - 1];
  ProtobufReader(ps_data, ps_length).read(ps);
  CUDF_EXPECTS(ps.footerLength + ps_length < len, "Invalid footer length");

  // If compression is used, the rest of the metadata is compressed
  // If no compressed is used, the decompressor is simply a pass-through
  decompressor = std::make_unique<OrcDecompressor>(ps.compression, ps.compressionBlockSize);

  // Read compressed filefooter section
  buffer             = source->host_read(len - ps_length - 1 - ps.footerLength, ps.footerLength);
  auto const ff_data = decompressor->decompress_blocks({buffer->data(), buffer->size()}, stream);
  ProtobufReader(ff_data.data(), ff_data.size()).read(ff);
  CUDF_EXPECTS(get_num_columns() > 0, "No columns found");

  // Read compressed metadata section
  buffer =
    source->host_read(len - ps_length - 1 - ps.footerLength - ps.metadataLength, ps.metadataLength);
  auto const md_data = decompressor->decompress_blocks({buffer->data(), buffer->size()}, stream);
  orc::ProtobufReader(md_data.data(), md_data.size()).read(md);

  init_parent_descriptors();
  init_column_names();
}

void metadata::init_column_names()
{
  column_names.resize(get_num_columns());
  thrust::tabulate(column_names.begin(), column_names.end(), [&](auto col_id) {
    if (not column_has_parent(col_id)) return std::string{};
    auto const& parent_field_names = ff.types[parent_id(col_id)].fieldNames;
    if (field_index(col_id) < static_cast<size_type>(parent_field_names.size())) {
      return parent_field_names[field_index(col_id)];
    }

    // Generate names for list and map child columns
    if (ff.types[parent_id(col_id)].subtypes.size() == 1) {
      return std::to_string(lists_column_view::child_column_index);
    } else {
      return std::to_string(field_index(col_id));
    }
  });

  column_paths.resize(get_num_columns());
  thrust::tabulate(column_paths.begin(), column_paths.end(), [&](auto col_id) {
    if (not column_has_parent(col_id)) return std::string{};
    // Don't include ORC root column name in path
    return (parent_id(col_id) == 0 ? "" : column_paths[parent_id(col_id)] + ".") +
           column_names[col_id];
  });
}

void metadata::init_parent_descriptors()
{
  auto const num_columns = static_cast<size_type>(ff.types.size());
  parents.resize(num_columns);

  for (size_type col_id = 0; col_id < num_columns; ++col_id) {
    auto const& subtypes    = ff.types[col_id].subtypes;
    auto const num_children = static_cast<size_type>(subtypes.size());
    for (size_type field_idx = 0; field_idx < num_children; ++field_idx) {
      auto const child_id = static_cast<size_type>(subtypes[field_idx]);
      CUDF_EXPECTS(child_id > col_id && child_id < num_columns, "Invalid column id");
      CUDF_EXPECTS(not column_has_parent(child_id), "Same node referenced twice");
      parents[child_id] = {col_id, field_idx};
    }
  }
}

}  // namespace cudf::io::orc
