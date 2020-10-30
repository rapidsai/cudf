/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <io/parquet/parquet_common.hpp>

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace cudf {
namespace io {
namespace parquet {
constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));

/**
 * @brief Struct that describes the Parquet file data header
 **/
struct file_header_s {
  uint32_t magic;
};

/**
 * @brief Struct that describes the Parquet file data postscript
 **/
struct file_ender_s {
  uint32_t footer_len;
  uint32_t magic;
};

/**
 * @brief Struct for describing an element/field in the Parquet format schema
 *
 * Parquet is a strongly-typed format so the file layout can be interpreted as
 * as a schema tree.
 */
struct SchemaElement {
  Type type                    = UNDEFINED_TYPE;
  ConvertedType converted_type = UNKNOWN;
  int32_t type_length =
    0;  // Byte length of FIXED_LENGTH_BYTE_ARRAY elements, or maximum bit length for other types
  FieldRepetitionType repetition_type = REQUIRED;
  std::string name                    = "";
  int32_t num_children                = 0;
  int32_t decimal_scale               = 0;
  int32_t decimal_precision           = 0;

  // The following fields are filled in later during schema initialization
  int max_definition_level = 0;
  int max_repetition_level = 0;
  int parent_idx           = 0;

  bool operator==(SchemaElement const &other) const
  {
    return type == other.type && converted_type == other.converted_type &&
           type_length == other.type_length && repetition_type == other.repetition_type &&
           name == other.name && num_children == other.num_children &&
           decimal_scale == other.decimal_scale && decimal_precision == other.decimal_precision;
  }

  // the parquet format is a little squishy when it comes to interpreting
  // repeated fields. sometimes repeated fields act as "stubs" in the schema
  // that don't represent a true nesting level.
  //
  // this is the case with plain lists:
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- not part of the output hierarchy
  //     required binary str (UTF8);
  //   };
  // }
  //
  // However, for backwards compatibility reasons, there are a few special cases, namely
  // List<Struct<>> (which also corresponds to how the map type is specified), where
  // this does not hold true
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- part of the hierarchy because it represents a struct
  //     required binary str (UTF8);
  //     required int32 num;
  //  };
  // }
  bool is_stub() const { return repetition_type == REPEATED && num_children == 1; }
  // in parquet terms, a group is a level of nesting in the schema. a group
  // can be a struct or a list
  bool is_struct() const
  {
    return type == UNDEFINED_TYPE &&
           // this assumption might be a little weak.
           ((repetition_type != REPEATED) || (repetition_type == REPEATED && num_children == 2));
  }
};

/**
 * @brief Thrift-derived struct describing a column chunk
 **/
struct ColumnChunkMetaData {
  Type type = BOOLEAN;
  std::vector<Encoding> encodings;
  std::vector<std::string> path_in_schema;
  Compression codec  = UNCOMPRESSED;
  int64_t num_values = 0;
  int64_t total_uncompressed_size =
    0;  // total byte size of all uncompressed pages in this column chunk (including the headers)
  int64_t total_compressed_size =
    0;  // total byte size of all compressed pages in this column chunk (including the headers)
  int64_t data_page_offset  = 0;  // Byte offset from beginning of file to first data page
  int64_t index_page_offset = 0;  // Byte offset from beginning of file to root index page
  int64_t dictionary_page_offset =
    0;  // Byte offset from the beginning of file to first (only) dictionary page
  std::vector<uint8_t> statistics_blob;  // Encoded chunk-level statistics as binary blob
};

/**
 * @brief Thrift-derived struct describing a chunk of data for a particular
 * column
 *
 * Each column chunk lives in a particular row group and are guaranteed to be
 * contiguous in the file. Any mssing or corrupted chunks can be skipped during
 * reading.
 **/
struct ColumnChunk {
  std::string file_path = "";
  int64_t file_offset   = 0;
  ColumnChunkMetaData meta_data;
  int64_t offset_index_offset = 0;  // File offset of ColumnChunk's OffsetIndex
  int32_t offset_index_length = 0;  // Size of ColumnChunk's OffsetIndex, in bytes
  int64_t column_index_offset = 0;  // File offset of ColumnChunk's ColumnIndex
  int32_t column_index_length = 0;  // Size of ColumnChunk's ColumnIndex, in bytes

  // Following fields are derived from other fields
  int schema_idx = -1;  // Index in flattened schema (derived from path_in_schema)
};

/**
 * @brief Thrift-derived struct describing a group of row data
 *
 * There may be one or more row groups within a dataset, with each row group
 * consisting of a column chunk for each column.
 **/
struct RowGroup {
  int64_t total_byte_size = 0;
  std::vector<ColumnChunk> columns;
  int64_t num_rows = 0;
};

/**
 * @brief Thrift-derived struct describing a key-value pair, for user metadata
 **/
struct KeyValue {
  std::string key;
  std::string value;
};

/**
 * @brief Thrift-derived struct describing file-level metadata
 *
 * The additional information stored in the key_value_metadata can be used
 * during reading to reconstruct the output data to the exact original dataset
 * prior to conversion to Parquet.
 **/
struct FileMetaData {
  int32_t version = 0;
  std::vector<SchemaElement> schema;
  int64_t num_rows = 0;
  std::vector<RowGroup> row_groups;
  std::vector<KeyValue> key_value_metadata;
  std::string created_by         = "";
  uint32_t column_order_listsize = 0;
};

/**
 * @brief Thrift-derived struct describing the header for a data page
 **/
struct DataPageHeader {
  int32_t num_values                 = 0;  // Number of values, including NULLs, in this data page.
  Encoding encoding                  = Encoding::PLAIN;  // Encoding used for this data page
  Encoding definition_level_encoding = Encoding::PLAIN;  // Encoding used for definition levels
  Encoding repetition_level_encoding = Encoding::PLAIN;  // Encoding used for repetition levels
};

/**
 * @brief Thrift-derived struct describing the header for a dictionary page
 **/
struct DictionaryPageHeader {
  int32_t num_values = 0;                // Number of values in the dictionary
  Encoding encoding  = Encoding::PLAIN;  // Encoding using this dictionary page
};

/**
 * @brief Thrift-derived struct describing the page header
 *
 * Column data are divided into individual chunks, which are subdivided into
 * pages. Each page has an associated header, describing the page type. There
 * can be multiple page types interleaved in a column chunk, and each page is
 * individually compressed and encoded. Any missing or corrupted pages can be
 * skipped during reading.
 **/
struct PageHeader {
  PageType type =
    PageType::DATA_PAGE;  // the type of the page: indicates which of the *_header fields is set
  int32_t uncompressed_page_size = 0;  // Uncompressed page size in bytes (not including the header)
  int32_t compressed_page_size   = 0;  // Compressed page size in bytes (not including the header)
  DataPageHeader data_page_header;
  DictionaryPageHeader dictionary_page_header;
};

/**
 * @brief Count the number of leading zeros in an unsigned integer
 **/
static inline int CountLeadingZeros32(uint32_t value)
{
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 32;
  return static_cast<int>(__builtin_clz(value));
#else
  int bitpos = 0;
  while (value != 0) {
    value >>= 1;
    ++bitpos;
  }
  return 32 - bitpos;
#endif
}

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
 **/
class CompactProtocolReader {
 protected:
  static const uint8_t g_list2struct[16];

 public:
  explicit CompactProtocolReader(const uint8_t *base = nullptr, size_t len = 0) { init(base, len); }
  void init(const uint8_t *base, size_t len)
  {
    m_base = m_cur = base;
    m_end          = base + len;
  }
  ptrdiff_t bytecount() const noexcept { return m_cur - m_base; }
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
  int32_t get_listh(uint8_t *el_type) noexcept
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
  bool read(FileMetaData *f);
  bool read(SchemaElement *s);
  bool read(RowGroup *r);
  bool read(ColumnChunk *c);
  bool read(ColumnChunkMetaData *c);
  bool read(PageHeader *p);
  bool read(DataPageHeader *d);
  bool read(DictionaryPageHeader *d);
  bool read(KeyValue *k);

 public:
  static int NumRequiredBits(uint32_t max_level) noexcept
  {
    return 32 - CountLeadingZeros32(max_level);
  }
  bool InitSchema(FileMetaData *md);

 protected:
  int WalkSchema(FileMetaData *md,
                 int idx           = 0,
                 int parent_idx    = 0,
                 int max_def_level = 0,
                 int max_rep_level = 0);

 protected:
  const uint8_t *m_base = nullptr;
  const uint8_t *m_cur  = nullptr;
  const uint8_t *m_end  = nullptr;

  friend class ParquetFieldInt32;
  friend class ParquetFieldInt64;
  template <typename T>
  friend class ParquetFieldStructListFunctor;
  friend class ParquetFieldString;
  template <typename T>
  friend class ParquetFieldStructFunctor;
  template <typename T>
  friend class ParquetFieldEnum;
  template <typename T>
  friend class ParquetFieldEnumListFunctor;
  friend class ParquetFieldStringList;
  friend class ParquetFieldStructBlob;
};

/**
 * @brief Functor to set value to 32 bit integer read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
class ParquetFieldInt32 {
  int field_val;
  int32_t &val;

 public:
  ParquetFieldInt32(int f, int32_t &v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader *cpr, int field_type)
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
  int64_t &val;

 public:
  ParquetFieldInt64(int f, int64_t &v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    val = cpr->get_i64();
    return (field_type < ST_FLD_I16 || field_type > ST_FLD_I64);
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
  std::vector<T> &val;

 public:
  ParquetFieldStructListFunctor(int f, std::vector<T> &v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader *cpr, int field_type)
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
ParquetFieldStructListFunctor<T> ParquetFieldStructList(int f, std::vector<T> &v)
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
  std::string &val;

 public:
  ParquetFieldString(int f, std::string &v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    if (field_type != ST_FLD_BINARY) return true;
    uint32_t n = cpr->get_u32();
    if (n < (size_t)(cpr->m_end - cpr->m_cur)) {
      val.assign((const char *)cpr->m_cur, n);
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
  T &val;

 public:
  ParquetFieldStructFunctor(int f, T &v) : field_val(f), val(v) {}

  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    return (field_type != ST_FLD_STRUCT || !(cpr->read(&val)));
  }

  int field() { return field_val; }
};

template <typename T>
ParquetFieldStructFunctor<T> ParquetFieldStruct(int f, T &v)
{
  return ParquetFieldStructFunctor<T>(f, v);
}

/**
 * @brief Functor to set value to enum read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
template <typename Enum>
class ParquetFieldEnum {
  int field_val;
  Enum &val;

 public:
  ParquetFieldEnum(int f, Enum &v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader *cpr, int field_type)
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
  std::vector<Enum> &val;

 public:
  ParquetFieldEnumListFunctor(int f, std::vector<Enum> &v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    int current_byte = cpr->getb();
    if ((current_byte & 0xf) != ST_FLD_I32) return true;
    int n = current_byte >> 4;
    if (n == 0xf) n = cpr->get_u32();
    val.resize(n);
    for (int32_t i = 0; i < n; i++) { val[i] = static_cast<Enum>(cpr->get_i32()); }
    return false;
  }

  int field() { return field_val; }
};

template <typename T>
ParquetFieldEnumListFunctor<T> ParquetFieldEnumList(int field, std::vector<T> &v)
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
  std::vector<std::string> &val;

 public:
  ParquetFieldStringList(int f, std::vector<std::string> &v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) return true;
    int current_byte = cpr->getb();
    if ((current_byte & 0xf) != ST_FLD_BINARY) return true;
    int n = current_byte >> 4;
    if (n == 0xf) n = cpr->get_u32();
    val.resize(n);
    for (int32_t i = 0; i < n; i++) {
      uint32_t l = cpr->get_u32();
      if (l < (size_t)(cpr->m_end - cpr->m_cur)) {
        val[i].assign((const char *)cpr->m_cur, l);
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
  std::vector<uint8_t> &val;

 public:
  ParquetFieldStructBlob(int f, std::vector<uint8_t> &v) : field_val(f), val(v) {}
  inline bool operator()(CompactProtocolReader *cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) return true;
    const uint8_t *start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
    return false;
  }

  int field() { return field_val; }
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
