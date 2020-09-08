/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <string.h>
#include "orc_builder.hpp"

namespace cudf {
namespace io {
namespace orc {
void ProtobufReader::skip_struct_field(int t)
{
  switch (t) {
    case PB_TYPE_VARINT: get_u32(); break;
    case PB_TYPE_FIXED64: skip_bytes(8); break;
    case PB_TYPE_FIXEDLEN: skip_bytes(get_u32()); break;
    case PB_TYPE_FIXED32: skip_bytes(4); break;
    default:
      // printf("invalid type (%d)\n", t);
      break;
  }
}

bool ProtobufReader::read(PostScript *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldUInt64(1, s->footerLength),
      FieldEnum<CompressionKind>(2, s->compression),
      FieldUInt32(3, s->compressionBlockSize),
      FieldPackedUInt32(4, s->version),
      FieldUInt64(5, s->metadataLength),
      FieldString(8000, s->magic));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(FileFooter *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldUInt64(1, s->headerLength),
      FieldUInt64(2, s->contentLength),
      FieldRepeatedStruct(3, s->stripes),
      FieldRepeatedStruct(4, s->types),
      FieldRepeatedStruct(5, s->metadata),
      FieldUInt64(6, s->numberOfRows),
      FieldRepeatedStructBlob(7, s->statistics),
      FieldUInt32(8, s->rowIndexStride));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(StripeInformation *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldUInt64(1, s->offset),
      FieldUInt64(2, s->indexLength),
      FieldUInt64(3, s->dataLength),
      FieldUInt32(4, s->footerLength),
      FieldUInt32(5, s->numberOfRows));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(SchemaType *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldEnum<TypeKind>(1, s->kind),
      FieldPackedUInt32(2, s->subtypes),
      FieldRepeatedString(3, s->fieldNames),
      FieldUInt32(4, s->maximumLength),
      FieldUInt32(5, s->precision),
      FieldUInt32(6, s->scale));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(UserMetadataItem *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldString(1, s->name),
      FieldString(2, s->value));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(StripeFooter *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldRepeatedStruct(1, s->streams),
      FieldRepeatedStruct(2, s->columns),
      FieldString(3, s->writerTimezone));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(Stream *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldEnum<StreamKind>(1, s->kind),
      FieldUInt32(2, s->column),
      FieldUInt64(3, s->length));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(ColumnEncoding *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldEnum<ColumnEncodingKind>(1, s->kind),
      FieldUInt32(2, s->dictionarySize));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(StripeStatistics *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldRepeatedStructBlob(1, s->colStats));
  return function_builder(s, maxlen, op);
}

bool ProtobufReader::read(Metadata *s, size_t maxlen)
{
  auto op = std::make_tuple(
      FieldRepeatedStruct(1, s->stripeStats));
  return function_builder(s, maxlen, op);
}


// return the column name
std::string FileFooter::GetColumnName(uint32_t column_id)
{
  std::string s       = "";
  uint32_t parent_idx = column_id, idx, field_idx;
  do {
    idx        = parent_idx;
    parent_idx = (idx < types.size()) ? (uint32_t)types[idx].parent_idx : ~0;
    field_idx  = (parent_idx < types.size()) ? (uint32_t)types[idx].field_idx : ~0;
    if (parent_idx >= types.size()) break;
    if (field_idx < types[parent_idx].fieldNames.size()) {
      if (s.length() > 0)
        s = types[parent_idx].fieldNames[field_idx] + "." + s;
      else
        s = types[parent_idx].fieldNames[field_idx];
    }
  } while (parent_idx != idx);
  // If we have no name (root column), generate a name
  if (s.length() == 0) { s = "col" + std::to_string(column_id); }
  return s;
}

// Initializes the parent_idx field in the schema
bool ProtobufReader::InitSchema(FileFooter *ff)
{
  int32_t schema_size = (int32_t)ff->types.size();
  for (int32_t i = 0; i < schema_size; i++) {
    int32_t num_children = (int32_t)ff->types[i].subtypes.size();
    if (ff->types[i].parent_idx == -1)  // Not initialized
    {
      ff->types[i].parent_idx = i;  // set root node as its own parent
    }
    for (int32_t j = 0; j < num_children; j++) {
      uint32_t column_id = ff->types[i].subtypes[j];
      if (column_id <= (uint32_t)i || column_id >= (uint32_t)schema_size) {
        // Invalid column id (or at least not a schema index)
        return false;
      }
      if (ff->types[column_id].parent_idx != -1) {
        // Same node referenced twice
        return false;
      }
      ff->types[column_id].parent_idx = i;
      ff->types[column_id].field_idx  = j;
    }
  }
  return true;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Brief ORC Protobuf Writer class
 *
 */
/* ----------------------------------------------------------------------------*/

#define PBW_BEGIN_STRUCT(st)                \
  size_t ProtobufWriter::write(const st *s) \
  {                                         \
    size_t struct_size = 0;

#define PBW_FLD_UINT(id, m)                         \
  struct_size += put_uint((id)*8 + PB_TYPE_VARINT); \
  struct_size += put_uint(static_cast<uint64_t>(s->m));

#define PBW_FLD_PACKED_UINT(id, m)                                                        \
  {                                                                                       \
    size_t cnt = s->m.size(), sz = 0, lpos;                                               \
    struct_size += put_uint((id)*8 + PB_TYPE_FIXEDLEN);                                   \
    lpos = m_buf->size();                                                                 \
    putb(0);                                                                              \
    for (size_t i = 0; i < cnt; i++) sz += put_uint(s->m[i]);                             \
    struct_size += sz + 1;                                                                \
    for (; sz > 0x7f; sz >>= 7, struct_size++)                                            \
      m_buf->insert(m_buf->begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80)); \
    (*m_buf)[lpos] = static_cast<uint8_t>(sz);                                            \
  }

#define PBW_FLD_STRING(id, m)                           \
  {                                                     \
    size_t len = s->m.length();                         \
    struct_size += put_uint((id)*8 + PB_TYPE_FIXEDLEN); \
    struct_size += put_uint(len) + len;                 \
    for (size_t i = 0; i < len; i++) putb(s->m[i]);     \
  }

#define PBW_FLD_BLOB(id, m)                             \
  {                                                     \
    size_t len = s->m.size();                           \
    struct_size += put_uint((id)*8 + PB_TYPE_FIXEDLEN); \
    struct_size += put_uint(len) + len;                 \
    for (size_t i = 0; i < len; i++) putb(s->m[i]);     \
  }

#define PBW_FLD_STRUCT(id, m)                                                             \
  {                                                                                       \
    size_t sz, lpos;                                                                      \
    struct_size += put_uint((id)*8 + PB_TYPE_FIXEDLEN);                                   \
    lpos = m_buf->size();                                                                 \
    putb(0);                                                                              \
    sz = write(&s->m);                                                                    \
    struct_size += sz + 1;                                                                \
    for (; sz > 0x7f; sz >>= 7, struct_size++)                                            \
      m_buf->insert(m_buf->begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80)); \
    (*m_buf)[lpos] = static_cast<uint8_t>(sz);                                            \
  }

#define PBW_FLD_REPEATED_STRING(id, m)                                 \
  {                                                                    \
    for (size_t k = 0; k < s->m.size(); k++) PBW_FLD_STRING(id, m[k]); \
  }

#define PBW_FLD_REPEATED_STRUCT(id, m)                                 \
  {                                                                    \
    for (size_t k = 0; k < s->m.size(); k++) PBW_FLD_STRUCT(id, m[k]); \
  }

#define PBW_FLD_REPEATED_STRUCT_BLOB(id, m)                          \
  {                                                                  \
    for (size_t k = 0; k < s->m.size(); k++) PBW_FLD_BLOB(id, m[k]); \
  }

#define PBW_END_STRUCT() \
  return struct_size;    \
  }

/**
 * @Brief Add a single rowIndexEntry, negative input values treated as not present
 *
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
    if (kind != STRING && kind != FLOAT && kind != DOUBLE) {
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

PBW_BEGIN_STRUCT(PostScript)
PBW_FLD_UINT(1, footerLength)
PBW_FLD_UINT(2, compression)
if (s->compression != NONE) { PBW_FLD_UINT(3, compressionBlockSize) }
PBW_FLD_PACKED_UINT(4, version)
PBW_FLD_UINT(5, metadataLength)
PBW_FLD_STRING(8000, magic)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(FileFooter)
PBW_FLD_UINT(1, headerLength)
PBW_FLD_UINT(2, contentLength)
PBW_FLD_REPEATED_STRUCT(3, stripes)
PBW_FLD_REPEATED_STRUCT(4, types)
PBW_FLD_REPEATED_STRUCT(5, metadata)
PBW_FLD_UINT(6, numberOfRows)
PBW_FLD_REPEATED_STRUCT_BLOB(7, statistics)
PBW_FLD_UINT(8, rowIndexStride)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(StripeInformation)
PBW_FLD_UINT(1, offset)
PBW_FLD_UINT(2, indexLength)
PBW_FLD_UINT(3, dataLength)
PBW_FLD_UINT(4, footerLength)
PBW_FLD_UINT(5, numberOfRows)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(SchemaType)
PBW_FLD_UINT(1, kind)
PBW_FLD_PACKED_UINT(2, subtypes)
PBW_FLD_REPEATED_STRING(3, fieldNames)
// PBW_FLD_UINT(4, maximumLength)
// PBW_FLD_UINT(5, precision)
// PBW_FLD_UINT(6, scale)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(UserMetadataItem)
PBW_FLD_STRING(1, name)
PBW_FLD_STRING(2, value)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(StripeFooter)
PBW_FLD_REPEATED_STRUCT(1, streams)
PBW_FLD_REPEATED_STRUCT(2, columns)
if (s->writerTimezone != "") { PBW_FLD_STRING(3, writerTimezone) }
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(Stream)
PBW_FLD_UINT(1, kind)
PBW_FLD_UINT(2, column)
PBW_FLD_UINT(3, length)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(ColumnEncoding)
PBW_FLD_UINT(1, kind)
if (s->kind == DICTIONARY || s->kind == DICTIONARY_V2) { PBW_FLD_UINT(2, dictionarySize) }
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(StripeStatistics)
PBW_FLD_REPEATED_STRUCT_BLOB(1, colStats)
PBW_END_STRUCT()

PBW_BEGIN_STRUCT(Metadata)
PBW_FLD_REPEATED_STRUCT(1, stripeStats)
PBW_END_STRUCT()

/* ----------------------------------------------------------------------------*/
/**
 * @Brief ORC decompression class
 *
 */
/* ----------------------------------------------------------------------------*/

OrcDecompressor::OrcDecompressor(CompressionKind kind, uint32_t blockSize)
  : m_kind(kind), m_blockSize(blockSize)
{
  if (kind != NONE) {
    int stream_type = IO_UNCOMP_STREAM_TYPE_INFER;  // Will be treated as invalid
    switch (kind) {
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

/* --------------------------------------------------------------------------*/
/**
 * @Brief ORC block decompression
 *
 * @param srcBytes[in] compressed data
 * @param srcLen[in] length of compressed data
 * @param dstLen[out] length of uncompressed data
 *
 * @returns pointer to uncompressed data, nullptr if error
 */
/* ----------------------------------------------------------------------------*/

const uint8_t *OrcDecompressor::Decompress(const uint8_t *srcBytes, size_t srcLen, size_t *dstLen)
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

}  // namespace orc
}  // namespace io
}  // namespace cudf
