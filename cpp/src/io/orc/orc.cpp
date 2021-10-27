/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "orc.h"
#include "orc_field_reader.hpp"
#include "orc_field_writer.hpp"

#include <thrust/tabulate.h>

#include <string>

namespace cudf {
namespace io {
namespace orc {

uint32_t ProtobufReader::read_field_size(const uint8_t* end)
{
  auto const size = get<uint32_t>();
  CUDF_EXPECTS(size <= static_cast<uint32_t>(end - m_cur), "Protobuf parsing out of bounds");
  return size;
}

void ProtobufReader::skip_struct_field(int t)
{
  switch (t) {
    case PB_TYPE_VARINT: get<uint32_t>(); break;
    case PB_TYPE_FIXED64: skip_bytes(8); break;
    case PB_TYPE_FIXEDLEN: skip_bytes(get<uint32_t>()); break;
    case PB_TYPE_FIXED32: skip_bytes(4); break;
    default: break;
  }
}

void ProtobufReader::read(PostScript& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.footerLength),
                            make_field_reader(2, s.compression),
                            make_field_reader(3, s.compressionBlockSize),
                            make_packed_field_reader(4, s.version),
                            make_field_reader(5, s.metadataLength),
                            make_field_reader(8000, s.magic));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(FileFooter& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.headerLength),
                            make_field_reader(2, s.contentLength),
                            make_field_reader(3, s.stripes),
                            make_field_reader(4, s.types),
                            make_field_reader(5, s.metadata),
                            make_field_reader(6, s.numberOfRows),
                            make_raw_field_reader(7, s.statistics),
                            make_field_reader(8, s.rowIndexStride));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeInformation& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.offset),
                            make_field_reader(2, s.indexLength),
                            make_field_reader(3, s.dataLength),
                            make_field_reader(4, s.footerLength),
                            make_field_reader(5, s.numberOfRows));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(SchemaType& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.kind),
                            make_packed_field_reader(2, s.subtypes),
                            make_field_reader(3, s.fieldNames),
                            make_field_reader(4, s.maximumLength),
                            make_field_reader(5, s.precision),
                            make_field_reader(6, s.scale));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(UserMetadataItem& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.name), make_field_reader(2, s.value));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeFooter& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.streams),
                            make_field_reader(2, s.columns),
                            make_field_reader(3, s.writerTimezone));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(Stream& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.kind),
                            make_field_reader(2, s.column_id),
                            make_field_reader(3, s.length));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(ColumnEncoding& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.kind), make_field_reader(2, s.dictionarySize));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(integer_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(
    make_field_reader(1, s.minimum), make_field_reader(2, s.maximum), make_field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(double_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(
    make_field_reader(1, s.minimum), make_field_reader(2, s.maximum), make_field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(string_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(
    make_field_reader(1, s.minimum), make_field_reader(2, s.maximum), make_field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(bucket_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_packed_field_reader(1, s.count));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(decimal_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(
    make_field_reader(1, s.minimum), make_field_reader(2, s.maximum), make_field_reader(3, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(date_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.minimum), make_field_reader(2, s.maximum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(binary_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.sum));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(timestamp_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.minimum),
                            make_field_reader(2, s.maximum),
                            make_field_reader(3, s.minimum_utc),
                            make_field_reader(4, s.maximum_utc));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(column_statistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.number_of_values),
                            make_field_reader(2, s.int_stats),
                            make_field_reader(3, s.double_stats),
                            make_field_reader(4, s.string_stats),
                            make_field_reader(5, s.bucket_stats),
                            make_field_reader(6, s.decimal_stats),
                            make_field_reader(7, s.date_stats),
                            make_field_reader(8, s.binary_stats),
                            make_field_reader(9, s.timestamp_stats));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(StripeStatistics& s, size_t maxlen)
{
  auto op = std::make_tuple(make_raw_field_reader(1, s.colStats));
  function_builder(s, maxlen, op);
}

void ProtobufReader::read(Metadata& s, size_t maxlen)
{
  auto op = std::make_tuple(make_field_reader(1, s.stripeStats));
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
                                         TypeKind kind)
{
  size_t sz = 0, lpos;
  putb(1 * 8 + PB_TYPE_FIXEDLEN);  // 1:RowIndex.entry
  lpos = m_buf->size();
  putb(0xcd);                      // sz+2
  putb(1 * 8 + PB_TYPE_FIXEDLEN);  // 1:positions[packed=true]
  putb(0xcd);                      // sz
  if (present_blk >= 0) sz += put_uint(present_blk);
  if (present_ofs >= 0) {
    sz += put_uint(present_ofs) + 2;
    putb(0);  // run pos = 0
    putb(0);  // bit pos = 0
  }
  if (data_blk >= 0) { sz += put_uint(data_blk); }
  if (data_ofs >= 0) {
    sz += put_uint(data_ofs);
    if (kind != STRING && kind != FLOAT && kind != DOUBLE && kind != DECIMAL) {
      putb(0);  // RLE run pos always zero (assumes RLE aligned with row index boundaries)
      sz++;
      if (kind == BOOLEAN) {
        putb(0);  // bit position in byte, always zero
        sz++;
      }
    }
  }
  if (kind !=
      INT)  // INT kind can be passed in to bypass 2nd stream index (dictionary length streams)
  {
    if (data2_blk >= 0) { sz += put_uint(data2_blk); }
    if (data2_ofs >= 0) {
      sz += put_uint(data2_ofs) + 1;
      putb(0);  // RLE run pos always zero (assumes RLE aligned with row index boundaries)
    }
  }
  m_buf->data()[lpos]     = (uint8_t)(sz + 2);
  m_buf->data()[lpos + 2] = (uint8_t)(sz);
}

size_t ProtobufWriter::write(const PostScript& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.footerLength);
  w.field_uint(2, s.compression);
  if (s.compression != NONE) { w.field_uint(3, s.compressionBlockSize); }
  w.field_packed_uint(4, s.version);
  w.field_uint(5, s.metadataLength);
  w.field_string(8000, s.magic);
  return w.value();
}

size_t ProtobufWriter::write(const FileFooter& s)
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
  return w.value();
}

size_t ProtobufWriter::write(const StripeInformation& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.offset);
  w.field_uint(2, s.indexLength);
  w.field_uint(3, s.dataLength);
  w.field_uint(4, s.footerLength);
  w.field_uint(5, s.numberOfRows);
  return w.value();
}

size_t ProtobufWriter::write(const SchemaType& s)
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

size_t ProtobufWriter::write(const UserMetadataItem& s)
{
  ProtobufFieldWriter w(this);
  w.field_string(1, s.name);
  w.field_string(2, s.value);
  return w.value();
}

size_t ProtobufWriter::write(const StripeFooter& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct(1, s.streams);
  w.field_repeated_struct(2, s.columns);
  if (s.writerTimezone != "") { w.field_string(3, s.writerTimezone); }
  return w.value();
}

size_t ProtobufWriter::write(const Stream& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.kind);
  if (s.column_id) w.field_uint(2, *s.column_id);
  w.field_uint(3, s.length);
  return w.value();
}

size_t ProtobufWriter::write(const ColumnEncoding& s)
{
  ProtobufFieldWriter w(this);
  w.field_uint(1, s.kind);
  if (s.kind == DICTIONARY || s.kind == DICTIONARY_V2) { w.field_uint(2, s.dictionarySize); }
  return w.value();
}

size_t ProtobufWriter::write(const StripeStatistics& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct_blob(1, s.colStats);
  return w.value();
}

size_t ProtobufWriter::write(const Metadata& s)
{
  ProtobufFieldWriter w(this);
  w.field_repeated_struct(1, s.stripeStats);
  return w.value();
}

OrcDecompressor::OrcDecompressor(CompressionKind kind, uint32_t blockSize)
  : m_kind(kind), m_blockSize(blockSize)
{
  if (kind != NONE) {
    int stream_type = IO_UNCOMP_STREAM_TYPE_INFER;  // Will be treated as invalid
    switch (kind) {
      case NONE: break;
      case ZLIB:
        stream_type    = IO_UNCOMP_STREAM_TYPE_INFLATE;
        m_log2MaxRatio = 11;  // < 2048:1
        break;
      case SNAPPY:
        stream_type    = IO_UNCOMP_STREAM_TYPE_SNAPPY;
        m_log2MaxRatio = 5;  // < 32:1
        break;
      case LZO: stream_type = IO_UNCOMP_STREAM_TYPE_LZO; break;
      case LZ4: stream_type = IO_UNCOMP_STREAM_TYPE_LZ4; break;
      case ZSTD: stream_type = IO_UNCOMP_STREAM_TYPE_ZSTD; break;
    }
    m_decompressor = HostDecompressor::Create(stream_type);
  } else {
    m_log2MaxRatio = 0;
  }
}

/**
 * @brief ORC block decompression
 *
 * @param[in] srcBytes compressed data
 * @param[in] srcLen length of compressed data
 * @param[out] dstLen length of uncompressed data
 *
 * @returns pointer to uncompressed data, nullptr if error
 */
const uint8_t* OrcDecompressor::Decompress(const uint8_t* srcBytes, size_t srcLen, size_t* dstLen)
{
  // If uncompressed, just pass-through the input
  if (m_kind == NONE) {
    *dstLen = srcLen;
    return srcBytes;
  }
  // First, scan the input for the number of blocks and worst-case output size
  size_t max_dst_length = 0;
  for (size_t i = 0; i + 3 < srcLen;) {
    uint32_t block_len       = srcBytes[i] | (srcBytes[i + 1] << 8) | (srcBytes[i + 2] << 16);
    uint32_t is_uncompressed = block_len & 1;
    i += 3;
    block_len >>= 1;
    if (is_uncompressed) {
      // Uncompressed block
      max_dst_length += block_len;
    } else {
      max_dst_length += m_blockSize;
    }
    i += block_len;
    if (i > srcLen || block_len > m_blockSize) { return nullptr; }
  }
  // Check if we have a single uncompressed block, or no blocks
  if (max_dst_length < m_blockSize) {
    if (srcLen < 3) {
      // Total size is less than the 3-byte header
      return nullptr;
    }
    *dstLen = srcLen - 3;
    return srcBytes + 3;
  }
  m_buf.resize(max_dst_length);
  auto dst          = m_buf.data();
  size_t dst_length = 0;
  for (size_t i = 0; i + 3 < srcLen;) {
    uint32_t block_len       = srcBytes[i] | (srcBytes[i + 1] << 8) | (srcBytes[i + 2] << 16);
    uint32_t is_uncompressed = block_len & 1;
    i += 3;
    block_len >>= 1;
    if (is_uncompressed) {
      // Uncompressed block
      memcpy(dst + dst_length, srcBytes + i, block_len);
      dst_length += block_len;
    } else {
      // Compressed block
      dst_length +=
        m_decompressor->Decompress(dst + dst_length, m_blockSize, srcBytes + i, block_len);
    }
    i += block_len;
  }
  *dstLen = dst_length;
  return m_buf.data();
}

metadata::metadata(datasource* const src) : source(src)
{
  const auto len         = source->size();
  const auto max_ps_size = std::min(len, static_cast<size_t>(256));

  // Read uncompressed postscript section (max 255 bytes + 1 byte for length)
  auto buffer            = source->host_read(len - max_ps_size, max_ps_size);
  const size_t ps_length = buffer->data()[max_ps_size - 1];
  const uint8_t* ps_data = &buffer->data()[max_ps_size - ps_length - 1];
  ProtobufReader(ps_data, ps_length).read(ps);
  CUDF_EXPECTS(ps.footerLength + ps_length < len, "Invalid footer length");

  // If compression is used, the rest of the metadata is compressed
  // If no compressed is used, the decompressor is simply a pass-through
  decompressor = std::make_unique<OrcDecompressor>(ps.compression, ps.compressionBlockSize);

  // Read compressed filefooter section
  buffer           = source->host_read(len - ps_length - 1 - ps.footerLength, ps.footerLength);
  size_t ff_length = 0;
  auto ff_data     = decompressor->Decompress(buffer->data(), ps.footerLength, &ff_length);
  ProtobufReader(ff_data, ff_length).read(ff);
  CUDF_EXPECTS(get_num_columns() > 0, "No columns found");

  // Read compressed metadata section
  buffer =
    source->host_read(len - ps_length - 1 - ps.footerLength - ps.metadataLength, ps.metadataLength);
  size_t md_length = 0;
  auto md_data     = decompressor->Decompress(buffer->data(), ps.metadataLength, &md_length);
  orc::ProtobufReader(md_data, md_length).read(md);

  init_parent_descriptors();
  init_column_names();
}

void metadata::init_column_names()
{
  column_names.resize(get_num_columns());
  thrust::tabulate(column_names.begin(), column_names.end(), [&](auto col_id) {
    if (not column_has_parent(col_id)) return std::string{};
    auto const& parent_field_names = ff.types[parent_id(col_id)].fieldNames;
    // Child columns of lists don't have a name in ORC files, generate placeholder in that case
    return field_index(col_id) < static_cast<size_type>(parent_field_names.size())
             ? parent_field_names[field_index(col_id)]
             : std::to_string(col_id);
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

}  // namespace orc
}  // namespace io
}  // namespace cudf
