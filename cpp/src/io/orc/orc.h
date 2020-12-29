/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <cudf/utilities/error.hpp>

#include "io/comp/io_uncomp.h"
#include "orc_common.h"

namespace cudf {
namespace io {
namespace orc {
struct PostScript {
  uint64_t footerLength         = 0;           // the length of the footer section in bytes
  CompressionKind compression   = NONE;        // the kind of generic compression used
  uint32_t compressionBlockSize = 256 * 1024;  // the maximum size of each compression chunk
  std::vector<uint32_t> version;               // the version of the writer [major, minor]
  uint64_t metadataLength = 0;                 // the length of the metadata section in bytes
  std::string magic       = "";                // the fixed string "ORC"
};

struct StripeInformation {
  uint64_t offset       = 0;  // the start of the stripe within the file
  uint64_t indexLength  = 0;  // the length of the indexes in bytes
  uint64_t dataLength   = 0;  // the length of the data in bytes
  uint32_t footerLength = 0;  // the length of the footer in bytes
  uint32_t numberOfRows = 0;  // the number of rows in the stripe
};

struct SchemaType {
  TypeKind kind = INVALID_TYPE_KIND;  // the kind of this type
  std::vector<uint32_t> subtypes;  // the type ids of any subcolumns for list, map, struct, or union
  std::vector<std::string> fieldNames;  // the list of field names for struct
  uint32_t maximumLength =
    0;  // optional: the maximum length of the type for varchar or char in UTF-8 characters
  uint32_t precision = 0;  // optional: the precision and scale for decimal
  uint32_t scale     = 0;
  // Inferred fields
  int32_t parent_idx = -1;  // parent node (equal to current node for root nodes)
  int32_t field_idx  = -1;  // field index in parent's subtype vector
};

struct UserMetadataItem {
  std::string name;   // the user defined key
  std::string value;  // the user defined binary value as string
};

struct IntegerStatistics {
  int64_t minimum = 0;
  int64_t maximum = 0;
  int64_t sum     = 0;
};

struct ColumnStatistics {
  uint64_t numberOfValues = 0;
  IntegerStatistics intStatistics;
};

typedef std::vector<uint8_t> ColStatsBlob;  // Column statistics blob

struct FileFooter {
  uint64_t headerLength  = 0;              // the length of the file header in bytes (always 3)
  uint64_t contentLength = 0;              // the length of the file header and body in bytes
  std::vector<StripeInformation> stripes;  // the information about the stripes
  std::vector<SchemaType> types;           // the schema information
  std::vector<UserMetadataItem> metadata;  // the user metadata that was added
  uint64_t numberOfRows = 0;               // the total number of rows in the file
  std::vector<ColStatsBlob> statistics;    // Column statistics blobs
  uint32_t rowIndexStride = 0;             // the maximum number of rows in each index entry
  // Helper methods
  std::string GetColumnName(uint32_t column_id);  // return the column name
};

struct Stream {
  StreamKind kind = INVALID_STREAM_KIND;
  uint32_t column = ~0;  // the column id
  uint64_t length = 0;   // the number of bytes in the file
};

struct ColumnEncoding {
  ColumnEncodingKind kind = INVALID_ENCODING_KIND;
  uint32_t dictionarySize = 0;  // for dictionary encodings, record the size of the dictionary
};

struct StripeFooter {
  std::vector<Stream> streams;          // the location of each stream
  std::vector<ColumnEncoding> columns;  // the encoding of each column
  std::string writerTimezone = "";      // time zone of the writer
};

struct StripeStatistics {
  std::vector<ColStatsBlob> colStats;  // Column statistics blobs
};

struct Metadata {
  std::vector<StripeStatistics> stripeStats;
};

// Minimal protobuf reader for orc metadata

/**
 * @brief Class for parsing Orc's Protocol Buffers encoded metadata
 */

class ProtobufReader {
 public:
  ProtobufReader(const uint8_t *base, size_t len) : m_base(base), m_cur(base), m_end(base + len) {}

  ptrdiff_t bytecount() const { return m_cur - m_base; }
  void skip_bytes(size_t bytecnt)
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }
  template <typename T>
  T get()
  {
    CUDF_FAIL("Unsupported return type");
  };

  void skip_struct_field(int t);

 public:
  bool read(PostScript &, size_t maxlen);
  bool read(FileFooter &, size_t maxlen);
  bool read(StripeInformation &, size_t maxlen);
  bool read(SchemaType &, size_t maxlen);
  bool read(UserMetadataItem &, size_t maxlen);
  bool read(StripeFooter &, size_t maxlen);
  bool read(Stream &, size_t maxlen);
  bool read(ColumnEncoding &, size_t maxlen);
  bool read(StripeStatistics &, size_t maxlen);
  bool read(ColumnStatistics &, size_t maxlen);
  bool read(IntegerStatistics &, size_t maxlen);
  bool read(Metadata &, size_t maxlen);

 protected:
  bool InitSchema(FileFooter &);

  template <typename T, typename... Operator>
  bool function_builder(T &s, size_t maxlen, std::tuple<Operator...> &op);
  template <typename T>
  bool function_builder_return(T &s, const uint8_t *end);
  struct FieldInt32;
  struct FieldUInt32;
  struct FieldInt64;
  struct FieldUInt64;
  template <typename Enum>
  struct FieldEnum;
  struct FieldPackedUInt32;
  struct FieldString;
  struct FieldRepeatedString;
  template <typename Enum>
  struct FieldRepeatedStructFunctor;
  template <typename Enum>
  struct FieldRepeatedStructBlobFunctor;
  template <typename Enum>
  struct FieldStructFunctor;
  template <typename Enum>
  FieldRepeatedStructFunctor<Enum> FieldRepeatedStruct(int f, std::vector<Enum> &v)
  {
    return FieldRepeatedStructFunctor<Enum>(f, v);
  }
  template <typename Enum>
  FieldRepeatedStructBlobFunctor<Enum> FieldRepeatedStructBlob(int f, std::vector<Enum> &v)
  {
    return FieldRepeatedStructBlobFunctor<Enum>(f, v);
  }
  template <typename Enum>
  FieldStructFunctor<Enum> FieldStruct(int f, Enum &v)
  {
    return FieldStructFunctor<Enum>(f, v);
  }

 protected:
  const uint8_t *const m_base;
  const uint8_t *m_cur;
  const uint8_t *const m_end;
};

template <>
inline uint8_t ProtobufReader::get<uint8_t>()
{
  return (m_cur < m_end) ? *m_cur++ : 0;
};

template <>
inline uint32_t ProtobufReader::get<uint32_t>()
{
  uint32_t v = 0;
  for (uint32_t l = 0;; l += 7) {
    uint32_t c = get<uint8_t>();
    v |= (c & 0x7f) << l;
    if (c < 0x80) return v;
  }
}

template <>
inline uint64_t ProtobufReader::get<uint64_t>()
{
  uint64_t v = 0;
  for (uint64_t l = 0;; l += 7) {
    uint64_t c = get<uint8_t>();
    v |= (c & 0x7f) << l;
    if (c < 0x80) return v;
  }
}

template <>
inline int32_t ProtobufReader::get<int32_t>()
{
  auto const u = get<uint32_t>();
  return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

template <>
inline int64_t ProtobufReader::get<int64_t>()
{
  auto const u = get<uint64_t>();
  return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
}

/**
 * @brief Class for encoding Orc's metadata with Protocol Buffers
 */
class ProtobufWriter {
 public:
  ProtobufWriter() { m_buf = nullptr; }
  ProtobufWriter(std::vector<uint8_t> *output) { m_buf = output; }
  void putb(uint8_t v) { m_buf->push_back(v); }
  uint32_t put_uint(uint64_t v)
  {
    int l = 1;
    while (v > 0x7f) {
      putb(static_cast<uint8_t>(v | 0x80));
      v >>= 7;
      l++;
    }
    putb(static_cast<uint8_t>(v));
    return l;
  }
  uint32_t put_int(int64_t v)
  {
    int64_t s = (v < 0);
    return put_uint(((v ^ -s) << 1) + s);
  }
  void put_row_index_entry(int32_t present_blk,
                           int32_t present_ofs,
                           int32_t data_blk,
                           int32_t data_ofs,
                           int32_t data2_blk,
                           int32_t data2_ofs,
                           TypeKind kind);

 public:
  size_t write(const PostScript &);
  size_t write(const FileFooter &);
  size_t write(const StripeInformation &);
  size_t write(const SchemaType &);
  size_t write(const UserMetadataItem &);
  size_t write(const StripeFooter &);
  size_t write(const Stream &);
  size_t write(const ColumnEncoding &);
  size_t write(const StripeStatistics &);
  size_t write(const Metadata &);

 protected:
  std::vector<uint8_t> *m_buf;
  struct ProtobufFieldWriter;
};

/**
 * @brief Class for decompressing Orc data blocks using the CPU
 */

class OrcDecompressor {
 public:
  OrcDecompressor(CompressionKind kind, uint32_t blockSize);
  const uint8_t *Decompress(const uint8_t *srcBytes, size_t srcLen, size_t *dstLen);
  uint32_t GetLog2MaxCompressionRatio() const { return m_log2MaxRatio; }
  uint32_t GetMaxUncompressedBlockSize(uint32_t block_len) const
  {
    return (block_len < (m_blockSize >> m_log2MaxRatio)) ? block_len << m_log2MaxRatio
                                                         : m_blockSize;
  }
  CompressionKind GetKind() const { return m_kind; }
  uint32_t GetBlockSize() const { return m_blockSize; }

 protected:
  CompressionKind const m_kind;
  uint32_t m_log2MaxRatio = 24;  // log2 of maximum compression ratio
  uint32_t const m_blockSize;
  std::unique_ptr<HostDecompressor> m_decompressor;
  std::vector<uint8_t> m_buf;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
