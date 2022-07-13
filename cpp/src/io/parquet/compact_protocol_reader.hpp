/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <thrust/optional.h>

#include <algorithm>
#include <cstddef>
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
  explicit CompactProtocolReader(const uint8_t* base = nullptr, size_t len = 0) { init(base, len); }
  void init(const uint8_t* base, size_t len)
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
  uint32_t get_u32() noexcept
  {
    uint32_t v = 0;
    for (uint32_t l = 0;; l += 7) {
      uint32_t c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) break;
    }
    return v;
  }
  uint64_t get_u64() noexcept
  {
    uint64_t v = 0;
    for (uint64_t l = 0;; l += 7) {
      uint64_t c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) break;
    }
    return v;
  }
  int32_t get_i16() noexcept { return get_i32(); }
  int32_t get_i32() noexcept
  {
    uint32_t u = get_u32();
    return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
  }
  int64_t get_i64() noexcept
  {
    uint64_t u = get_u64();
    return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
  }
  int32_t get_listh(uint8_t* el_type) noexcept
  {
    uint32_t c = getb();
    int32_t sz = c >> 4;
    *el_type   = c & 0xf;
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
  bool read(KeyValue* k);
  bool read(PageLocation* p);
  bool read(OffsetIndex* o);
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
  const uint8_t* m_base = nullptr;
  const uint8_t* m_cur  = nullptr;
  const uint8_t* m_end  = nullptr;

  friend class ParquetFieldBool;
  friend class ParquetFieldBoolList;
  friend class ParquetFieldInt8;
  friend class ParquetFieldInt32;
  friend class ParquetFieldOptionalInt32;
  friend class ParquetFieldInt64;
  friend class ParquetFieldInt64List;
  template <typename T>
  friend class ParquetFieldStructListFunctor;
  friend class ParquetFieldString;
  template <typename T>
  friend class ParquetFieldStructFunctor;
  template <typename T, bool>
  friend class ParquetFieldUnionFunctor;
  template <typename T>
  friend class ParquetFieldEnum;
  template <typename T>
  friend class ParquetFieldEnumListFunctor;
  friend class ParquetFieldStringList;
  friend class ParquetFieldBinary;
  friend class ParquetFieldBinaryList;
  friend class ParquetFieldStructBlob;
};

/**
 * @brief Functor to set value to bool read from CompactProtocolReader
 *
 * @return True if field type is not bool
 */
class ParquetFieldBool {
  int field_val;
  bool& val;

 public:
  ParquetFieldBool(int f, bool& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    return (field_type != ST_FLD_TRUE && field_type != ST_FLD_FALSE) ||
           !(val = (field_type == ST_FLD_TRUE), true);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a vector of booleans from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * bool fails
 */
class ParquetFieldBoolList {
  int field_val;
  std::vector<bool>& val;

 public:
  ParquetFieldBoolList(int f, std::vector<bool>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    int32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_TRUE) return true;
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      unsigned int current_byte = cpr->getb();
      if (current_byte != ST_FLD_TRUE && current_byte != ST_FLD_FALSE) return true;
      val[i] = current_byte == ST_FLD_TRUE;
    }
    return false;
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to set value to 8 bit integer read from CompactProtocolReader
 *
 * @return True if field type is not int8
 */
class ParquetFieldInt8 {
  int field_val;
  int8_t& val;

 public:
  ParquetFieldInt8(int f, int8_t& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = cpr->getb();
    return (field_type != ST_FLD_BYTE);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to set value to 32 bit integer read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
class ParquetFieldInt32 {
  int field_val;
  int32_t& val;

 public:
  ParquetFieldInt32(int f, int32_t& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = cpr->get_i32();
    return (field_type != ST_FLD_I32);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to set value to optional 32 bit integer read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
class ParquetFieldOptionalInt32 {
  int field_val;
  thrust::optional<int32_t>& val;

 public:
  ParquetFieldOptionalInt32(int f, thrust::optional<int32_t>& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = cpr->get_i32();
    return (field_type != ST_FLD_I32);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to set value to 64 bit integer read from CompactProtocolReader
 *
 * @return True if field type is not int32 or int64
 */
class ParquetFieldInt64 {
  int field_val;
  int64_t& val;

 public:
  ParquetFieldInt64(int f, int64_t& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = cpr->get_i64();
    return (field_type < ST_FLD_I16 || field_type > ST_FLD_I64);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a vector of 64-bit integers from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * int64 fails
 */
class ParquetFieldInt64List {
  int field_val;
  std::vector<int64_t>& val;

 public:
  ParquetFieldInt64List(int f, std::vector<int64_t>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    int32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_I64) return true;
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      val[i] = cpr->get_i64();
    }
    return false;
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a vector of structures from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class ParquetFieldStructListFunctor {
  int field_val;
  std::vector<T>& val;

 public:
  ParquetFieldStructListFunctor(int f, std::vector<T>& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;

    int current_byte = cpr->getb();
    if ((current_byte & 0xf) != ST_FLD_STRUCT) return true;
    int n = current_byte >> 4;
    if (n == 0xf) n = cpr->get_u32();
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      if (!(cpr->read(&val[i]))) { return true; }
    }

    return false;
  }

  int field() { return field_val; }
};

template <typename T>
ParquetFieldStructListFunctor<T> ParquetFieldStructList(int f, std::vector<T>& v)
{
  return ParquetFieldStructListFunctor<T>(f, v);
}

/**
 * @brief Functor to read a string from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of string exceeds bounds
 * of the CompactProtocolReader
 */
class ParquetFieldString {
  int field_val;
  std::string& val;

 public:
  ParquetFieldString(int f, std::string& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_BINARY) return true;
    uint32_t n = cpr->get_u32();
    if (n < (size_t)(cpr->m_end - cpr->m_cur)) {
      val.assign((const char*)cpr->m_cur, n);
      cpr->m_cur += n;
      return false;
    } else {
      return true;
    }
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a structure from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class ParquetFieldStructFunctor {
  int field_val;
  T& val;

 public:
  ParquetFieldStructFunctor(int f, T& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    return (field_type != ST_FLD_STRUCT || !(cpr->read(&val)));
  }

  int field() { return field_val; }
};

template <typename T>
ParquetFieldStructFunctor<T> ParquetFieldStruct(int f, T& v)
{
  return ParquetFieldStructFunctor<T>(f, v);
}

/**
 * @brief Functor to read a union member from CompactProtocolReader
 *
 * @tparam is_empty True if tparam `T` type is empty type, else false.
 *
 * @return True if field types mismatch or if the process of reading a
 * union member fails
 */
template <typename T, bool is_empty = false>
class ParquetFieldUnionFunctor {
  int field_val;
  bool& is_set;
  T& val;

 public:
  ParquetFieldUnionFunctor(int f, bool& b, T& v) : field_val(f), is_set(b), val(v) {}

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
struct ParquetFieldUnionFunctor<T, true> {
  int field_val;
  bool& is_set;
  T& val;

 public:
  ParquetFieldUnionFunctor(int f, bool& b, T& v) : field_val(f), is_set(b), val(v) {}

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

  int field() { return field_val; }
};

template <typename T>
ParquetFieldUnionFunctor<T, std::is_empty_v<T>> ParquetFieldUnion(int f, bool& b, T& v)
{
  return ParquetFieldUnionFunctor<T, std::is_empty_v<T>>(f, b, v);
}

/**
 * @brief Functor to set value to enum read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
template <typename Enum>
class ParquetFieldEnum {
  int field_val;
  Enum& val;

 public:
  ParquetFieldEnum(int f, Enum& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    val = static_cast<Enum>(cpr->get_i32());
    return (field_type != ST_FLD_I32);
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a vector of enums from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * enum fails
 */
template <typename Enum>
class ParquetFieldEnumListFunctor {
  int field_val;
  std::vector<Enum>& val;

 public:
  ParquetFieldEnumListFunctor(int f, std::vector<Enum>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    int current_byte = cpr->getb();
    if ((current_byte & 0xf) != ST_FLD_I32) return true;
    int n = current_byte >> 4;
    if (n == 0xf) n = cpr->get_u32();
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      val[i] = static_cast<Enum>(cpr->get_i32());
    }
    return false;
  }

  int field() { return field_val; }
};

template <typename T>
ParquetFieldEnumListFunctor<T> ParquetFieldEnumList(int field, std::vector<T>& v)
{
  return ParquetFieldEnumListFunctor<T>(field, v);
}

/**
 * @brief Functor to read a vector of strings from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * string fails
 */
class ParquetFieldStringList {
  int field_val;
  std::vector<std::string>& val;

 public:
  ParquetFieldStringList(int f, std::vector<std::string>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    int32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_BINARY) return true;
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      uint32_t l = cpr->get_u32();
      if (l < (size_t)(cpr->m_end - cpr->m_cur)) {
        val[i].assign((const char*)cpr->m_cur, l);
        cpr->m_cur += l;
      } else
        return true;
    }
    return false;
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a binary from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of binary exceeds bounds
 * of the CompactProtocolReader
 */
class ParquetFieldBinary {
  int field_val;
  std::vector<uint8_t>& val;

 public:
  ParquetFieldBinary(int f, std::vector<uint8_t>& v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_BINARY) return true;
    uint32_t n = cpr->get_u32();
    if (n <= (size_t)(cpr->m_end - cpr->m_cur)) {
      val.resize(n);
      val.assign(cpr->m_cur, cpr->m_cur + n);
      cpr->m_cur += n;
      return false;
    } else {
      return true;
    }
  }

  int field() { return field_val; }
};

/**
 * @brief Functor to read a vector of binaries from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * binary fails
 */
class ParquetFieldBinaryList {
  int field_val;
  std::vector<std::vector<uint8_t>>& val;

 public:
  ParquetFieldBinaryList(int f, std::vector<std::vector<uint8_t>>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    uint8_t t;
    int32_t n = cpr->get_listh(&t);
    if (t != ST_FLD_BINARY) return true;
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
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

  int field() { return field_val; }
};

/**
 * @brief Functor to read a struct from CompactProtocolReader
 *
 * @return True if field type mismatches
 */
class ParquetFieldStructBlob {
  int field_val;
  std::vector<uint8_t>& val;

 public:
  ParquetFieldStructBlob(int f, std::vector<uint8_t>& v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) return true;
    const uint8_t* start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
    return false;
  }

  int field() { return field_val; }
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
