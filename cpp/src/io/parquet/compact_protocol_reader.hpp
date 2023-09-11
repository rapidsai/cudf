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
#include <vector>

namespace cudf {
namespace io {
namespace parquet {
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
 protected:
  static const uint8_t g_list2struct[16];

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
      if (c < 0x80) break;
    }
    return v;
  }

  // returns a zigzag encoded signed integer
  template <typename T>
  T get_zigzag() noexcept
  {
    using U = std::make_unsigned_t<T>;
    U u     = get_varint<U>();
    return static_cast<T>((u >> 1u) ^ -static_cast<T>(u & 1));
  }

  // thrift spec says to use zigzag i32 for i16 types
  int32_t get_i16() noexcept { return get_zigzag<int32_t>(); }
  int32_t get_i32() noexcept { return get_zigzag<int32_t>(); }
  int64_t get_i64() noexcept { return get_zigzag<int64_t>(); }

  uint32_t get_u32() noexcept { return get_varint<int32_t>(); }
  uint64_t get_u64() noexcept { return get_varint<int64_t>(); }

  uint32_t get_listh(uint8_t* el_type) noexcept
  {
    uint32_t c  = getb();
    uint32_t sz = c >> 4;
    *el_type    = c & 0xf;
    if (sz == 0xf) sz = get_u32();
    return sz;
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
  bool read(SizeStatistics* s);
  bool read(ColumnIndex* c);
  bool read(Statistics* s);

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

  friend class ParquetFieldString;
  friend class ParquetFieldStringList;
  friend class ParquetFieldBinary;
  friend class ParquetFieldBinaryList;
  friend class ParquetFieldStructBlob;
};

class ParquetField {
 protected:
  int field_val;

  ParquetField(int f) : field_val(f) {}

 public:
  int field() const { return field_val; }
};

/**
 * @brief Functor to set value to bool read from CompactProtocolReader
 *
 * @return True if field type is not bool
 */
class ParquetFieldBool : public ParquetField {
  bool& val;

 public:
  ParquetFieldBool(int f, bool& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    return (field_type != ST_FLD_TRUE && field_type != ST_FLD_FALSE) ||
           !(val = (field_type == ST_FLD_TRUE), true);
  }
};

/**
 * @brief Functor to read a vector of booleans from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * bool fails
 */
class ParquetFieldBoolList : public ParquetField {
  std::vector<bool>& val;

 public:
  ParquetFieldBoolList(int f, std::vector<bool>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_TRUE) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      unsigned int current_byte = cpr->getb();
      if (current_byte != ST_FLD_TRUE && current_byte != ST_FLD_FALSE) return true;
      val[i] = current_byte == ST_FLD_TRUE;
    }
    return false;
  }
};

/**
 * @brief Base type for a functor that reads an integer from CompactProtocolReader
 *
 * Assuming signed ints since the parquet spec does not use unsigned ints anywhere.
 *
 * @return True if there is a type mismatch
 */
template <typename T, int EXPECTED_TYPE>
class ParquetFieldInt : public ParquetField {
  static constexpr bool is_byte = std::is_same_v<T, int8_t>;

  T& val;

 public:
  ParquetFieldInt(int f, T& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if constexpr (is_byte) {
      val = cpr->getb();
    } else {
      val = cpr->get_zigzag<T>();
    }
    return (field_type != EXPECTED_TYPE);
  }

  int field() { return field_val; }
};

using ParquetFieldInt8  = ParquetFieldInt<int8_t, ST_FLD_BYTE>;
using ParquetFieldInt32 = ParquetFieldInt<int32_t, ST_FLD_I32>;
using ParquetFieldInt64 = ParquetFieldInt<int64_t, ST_FLD_I64>;

/**
 * @brief Functor to read a vector of integers from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * integer fails
 */
template <typename T, int EXPECTED_TYPE>
class ParquetFieldIntList : public ParquetField {
  std::vector<int64_t>& val;

 public:
  ParquetFieldIntList(int f, std::vector<int64_t>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != EXPECTED_TYPE) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      val[i] = cpr->get_zigzag<T>();
    }
    return false;
  }
};

using ParquetFieldInt64List = ParquetFieldIntList<int64_t, ST_FLD_I64>;

/**
 * @brief Functor to read a string from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of string exceeds bounds
 * of the CompactProtocolReader
 */
class ParquetFieldString : public ParquetField {
  std::string& val;

 public:
  ParquetFieldString(int f, std::string& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_BINARY) return true;
    uint32_t n = cpr->get_u32();
    if (n < static_cast<size_t>(cpr->m_end - cpr->m_cur)) {
      val.assign(reinterpret_cast<char const*>(cpr->m_cur), n);
      cpr->m_cur += n;
      return false;
    } else {
      return true;
    }
  }
};

/**
 * @brief Functor to read a vector of strings from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * string fails
 */
class ParquetFieldStringList : public ParquetField {
  std::vector<std::string>& val;

 public:
  ParquetFieldStringList(int f, std::vector<std::string>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_BINARY) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t l = cpr->get_u32();
      if (l < static_cast<size_t>(cpr->m_end - cpr->m_cur)) {
        val[i].assign(reinterpret_cast<char const*>(cpr->m_cur), l);
        cpr->m_cur += l;
      } else
        return true;
    }
    return false;
  }
};

/**
 * @brief Functor to set value to enum read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
template <typename Enum>
class ParquetFieldEnum : public ParquetField {
  Enum& val;

 public:
  ParquetFieldEnum(int f, Enum& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = static_cast<Enum>(cpr->get_i32());
    return (field_type != ST_FLD_I32);
  }
};

/**
 * @brief Functor to read a vector of enums from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * enum fails
 */
template <typename Enum>
class ParquetFieldEnumListFunctor : public ParquetField {
  std::vector<Enum>& val;

 public:
  ParquetFieldEnumListFunctor(int f, std::vector<Enum>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_I32) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      val[i] = static_cast<Enum>(cpr->get_i32());
    }
    return false;
  }
};

template <typename T>
ParquetFieldEnumListFunctor<T> ParquetFieldEnumList(int field, std::vector<T>& v)
{
  return ParquetFieldEnumListFunctor<T>(field, v);
}

/**
 * @brief Functor to read a structure from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class ParquetFieldStruct : public ParquetField {
  T& val;

 public:
  ParquetFieldStruct(int f, T& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    return (field_type != ST_FLD_STRUCT || !(cpr->read(&val)));
  }
};

/**
 * @brief Functor to read a vector of structures from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class ParquetFieldStructList : public ParquetField {
  std::vector<T>& val;

 public:
  ParquetFieldStructList(int f, std::vector<T>& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;

    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_STRUCT) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      if (!(cpr->read(&val[i]))) { return true; }
    }

    return false;
  }
};

/**
 * @brief Functor to read a union member from CompactProtocolReader
 *
 * @tparam is_empty True if tparam `T` type is empty type, else false.
 *
 * @return True if field types mismatch or if the process of reading a
 * union member fails
 */
template <typename T, bool is_empty = false>
class ParquetFieldUnionFunctor : public ParquetField {
  bool& is_set;
  T& val;

 public:
  ParquetFieldUnionFunctor(int f, bool& b, T& v) : ParquetField(f), is_set(b), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) {
      return true;
    } else {
      is_set = true;
      return !cpr->read(&val);
    }
  }

  int field() { return field_val; }
};

template <typename T>
class ParquetFieldUnionFunctor<T, true> : public ParquetField {
  bool& is_set;
  T& val;

 public:
  ParquetFieldUnionFunctor(int f, bool& b, T& v) : ParquetField(f), is_set(b), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) {
      return true;
    } else {
      is_set = true;
      cpr->skip_struct_field(field_type);
      return false;
    }
  }
};

template <typename T>
ParquetFieldUnionFunctor<T, std::is_empty_v<T>> ParquetFieldUnion(int f, bool& b, T& v)
{
  return ParquetFieldUnionFunctor<T, std::is_empty_v<T>>(f, b, v);
}

/**
 * @brief Functor to read a binary from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of binary exceeds bounds
 * of the CompactProtocolReader
 */
class ParquetFieldBinary : public ParquetField {
  std::vector<uint8_t>& val;

 public:
  ParquetFieldBinary(int f, std::vector<uint8_t>& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_BINARY) return true;
    uint32_t n = cpr->get_u32();
    if (n <= static_cast<size_t>(cpr->m_end - cpr->m_cur)) {
      val.resize(n);
      val.assign(cpr->m_cur, cpr->m_cur + n);
      cpr->m_cur += n;
      return false;
    } else {
      return true;
    }
  }
};

/**
 * @brief Functor to read a vector of binaries from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * binary fails
 */
class ParquetFieldBinaryList : public ParquetField {
  std::vector<std::vector<uint8_t>>& val;

 public:
  ParquetFieldBinaryList(int f, std::vector<std::vector<uint8_t>>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    uint32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_BINARY) return true;
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t l = cpr->get_u32();
      if (l <= (size_t)(cpr->m_end - cpr->m_cur)) {
        val[i].resize(l);
        val[i].assign(cpr->m_cur, cpr->m_cur + l);
        cpr->m_cur += l;
      } else
        return true;
    }
    return false;
  }
};

/**
 * @brief Functor to read a struct from CompactProtocolReader
 *
 * @return True if field type mismatches
 */
class ParquetFieldStructBlob : public ParquetField {
  std::vector<uint8_t>& val;

 public:
  ParquetFieldStructBlob(int f, std::vector<uint8_t>& v) : ParquetField(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) return true;
    uint8_t const* start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
    return false;
  }
};

// functor to wrap functors for optional fields
template <typename T, typename FieldFunctor>
class ParquetFieldOptional : public ParquetField {
  std::optional<T>& val;

 public:
  ParquetFieldOptional(int f, std::optional<T>& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    bool res = FieldFunctor(field_val, v).operator()(cpr, field_type);
    if (!res) { val = v; }
    return res;
  }
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
