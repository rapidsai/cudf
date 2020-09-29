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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

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

typedef std::vector<uint8_t> ColumnStatistics;  // Column statistics blob

struct FileFooter {
  uint64_t headerLength  = 0;                // the length of the file header in bytes (always 3)
  uint64_t contentLength = 0;                // the length of the file header and body in bytes
  std::vector<StripeInformation> stripes;    // the information about the stripes
  std::vector<SchemaType> types;             // the schema information
  std::vector<UserMetadataItem> metadata;    // the user metadata that was added
  uint64_t numberOfRows = 0;                 // the total number of rows in the file
  std::vector<ColumnStatistics> statistics;  // Column statistics blobs
  uint32_t rowIndexStride = 0;               // the maximum number of rows in each index entry
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
  std::vector<ColumnStatistics> colStats;  // Column statistics blobs
};

struct Metadata {
  std::vector<StripeStatistics> stripeStats;
};

// Minimal protobuf reader for orc metadata

/**
 * @brief Class for parsing Orc's Protocol Buffers encoded metadata
 *
 **/

class ProtobufReader {
 public:
  ProtobufReader() { m_base = m_cur = m_end = nullptr; }
  ProtobufReader(const uint8_t *base, size_t len) { init(base, len); }
  void init(const uint8_t *base, size_t len)
  {
    m_base = m_cur = base;
    m_end          = base + len;
  }
  ptrdiff_t bytecount() const { return m_cur - m_base; }
  unsigned int getb() { return (m_cur < m_end) ? *m_cur++ : 0; }
  void skip_bytes(size_t bytecnt)
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }
  uint32_t get_u32()
  {
    uint32_t v = 0;
    for (uint32_t l = 0;; l += 7) {
      uint32_t c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) return v;
    }
  }
  uint64_t get_u64()
  {
    uint64_t v = 0;
    for (uint64_t l = 0;; l += 7) {
      uint64_t c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) return v;
    }
  }
  int32_t get_i32()
  {
    uint32_t u = get_u32();
    return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
  }
  int64_t get_i64()
  {
    uint64_t u = get_u64();
    return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
  }
  void skip_struct_field(int t);

 public:
#define DECL_ORC_STRUCT(st) bool read(st *, size_t maxlen)
  DECL_ORC_STRUCT(PostScript);
  DECL_ORC_STRUCT(FileFooter);
  DECL_ORC_STRUCT(StripeInformation);
  DECL_ORC_STRUCT(SchemaType);
  DECL_ORC_STRUCT(UserMetadataItem);
  DECL_ORC_STRUCT(StripeFooter);
  DECL_ORC_STRUCT(Stream);
  DECL_ORC_STRUCT(ColumnEncoding);
  DECL_ORC_STRUCT(StripeStatistics);
  DECL_ORC_STRUCT(Metadata);
#undef DECL_ORC_STRUCT
 protected:
  bool InitSchema(FileFooter *);

 protected:
  const uint8_t *m_base;
  const uint8_t *m_cur;
  const uint8_t *m_end;
};

/**
 * @brief Class for encoding Orc's metadata with Protocol Buffers
 *
 **/
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
#define DECL_PBW_STRUCT(st) size_t write(const st *)
  DECL_PBW_STRUCT(PostScript);
  DECL_PBW_STRUCT(FileFooter);
  DECL_PBW_STRUCT(StripeInformation);
  DECL_PBW_STRUCT(SchemaType);
  DECL_PBW_STRUCT(UserMetadataItem);
  DECL_PBW_STRUCT(StripeFooter);
  DECL_PBW_STRUCT(Stream);
  DECL_PBW_STRUCT(ColumnEncoding);
  DECL_PBW_STRUCT(StripeStatistics);
  DECL_PBW_STRUCT(Metadata);
#undef DECL_PBW_STRUCT
 protected:
  std::vector<uint8_t> *m_buf;
};

/**
 * @brief Class for decompressing Orc data blocks using the CPU
 *
 **/

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
