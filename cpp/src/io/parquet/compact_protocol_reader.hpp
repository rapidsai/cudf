/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "parquet.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Class for parsing Parquet's Thrift Compact Protocol encoded metadata
 *
 * This class takes in the starting location of the Parquet metadata, and fills
 * out Thrift-derived structs and a schema tree.
 *
 * In a Parquet, the metadata is separated from the data, both conceptually and
 * physically. There may be multiple data files sharing a common metadata file.
 *
 * The parser handles both V1 and V2 Parquet datasets, although not all
 * compression codecs are supported yet.
 */
class CompactProtocolReader {
 public:
  explicit CompactProtocolReader(uint8_t const* base = nullptr, size_t len = 0) { init(base, len); }
  void init(uint8_t const* base, size_t len)
  {
    m_base = m_cur = base;
    m_end          = base + len;
  }
  [[nodiscard]] ptrdiff_t bytecount() const noexcept { return m_cur - m_base; }
  unsigned int getb() noexcept { return (m_cur < m_end) ? *m_cur++ : 0; }
  void skip_bytes(size_t bytecnt) noexcept
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }

  // returns a varint encoded integer
  template <typename T>
  T get_varint() noexcept
  {
    T v = 0;
    for (uint32_t l = 0;; l += 7) {
      T c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) { break; }
    }
    return v;
  }

  // returns a zigzag encoded signed integer
  template <typename T>
  T get_zigzag() noexcept
  {
    using U   = std::make_unsigned_t<T>;
    U const u = get_varint<U>();
    return static_cast<T>((u >> 1u) ^ -static_cast<T>(u & 1));
  }

  // thrift spec says to use zigzag i32 for i16 types
  int32_t get_i16() noexcept { return get_zigzag<int32_t>(); }
  int32_t get_i32() noexcept { return get_zigzag<int32_t>(); }
  int64_t get_i64() noexcept { return get_zigzag<int64_t>(); }

  uint32_t get_u32() noexcept { return get_varint<uint32_t>(); }
  uint64_t get_u64() noexcept { return get_varint<uint64_t>(); }

  [[nodiscard]] std::pair<uint8_t, uint32_t> get_listh() noexcept
  {
    uint32_t const c = getb();
    uint32_t sz      = c >> 4;
    uint8_t t        = c & 0xf;
    if (sz == 0xf) { sz = get_u32(); }
    return {t, sz};
  }

  bool skip_struct_field(int t, int depth = 0);

 public:
  // Generate Thrift structure parsing routines
  bool read(FileMetaData* f);
  bool read(SchemaElement* s);
  bool read(LogicalType* l);
  bool read(DecimalType* d);
  bool read(TimeType* t);
  bool read(TimeUnit* u);
  bool read(TimestampType* t);
  bool read(IntType* t);
  bool read(RowGroup* r);
  bool read(ColumnChunk* c);
  bool read(ColumnChunkMetaData* c);
  bool read(PageHeader* p);
  bool read(DataPageHeader* d);
  bool read(DictionaryPageHeader* d);
  bool read(DataPageHeaderV2* d);
  bool read(KeyValue* k);
  bool read(PageLocation* p);
  bool read(OffsetIndex* o);
  bool read(ColumnIndex* c);
  bool read(Statistics* s);
  bool read(ColumnOrder* c);

 public:
  static int NumRequiredBits(uint32_t max_level) noexcept
  {
    return 32 - CountLeadingZeros32(max_level);
  }
  bool InitSchema(FileMetaData* md);

 protected:
  int WalkSchema(FileMetaData* md,
                 int idx           = 0,
                 int parent_idx    = 0,
                 int max_def_level = 0,
                 int max_rep_level = 0);

 protected:
  uint8_t const* m_base = nullptr;
  uint8_t const* m_cur  = nullptr;
  uint8_t const* m_end  = nullptr;

  friend class parquet_field_string;
  friend class parquet_field_string_list;
  friend class parquet_field_binary;
  friend class parquet_field_binary_list;
  friend class parquet_field_struct_blob;
};

}  // namespace cudf::io::parquet::detail
