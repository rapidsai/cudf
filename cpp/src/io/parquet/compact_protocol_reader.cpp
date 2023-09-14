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

#include "compact_protocol_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <tuple>

namespace cudf {
namespace io {
namespace parquet {

/**
 * @brief Base class for parquet field functors.
 *
 * Holds the field value used by all of the specialized functors.
 */
class ParquetField {
 protected:
  int const field_val;

  ParquetField(int f) : field_val(f) {}

 public:
  virtual ~ParquetField() = default;
  int field() const { return field_val; }
};

/**
 * @brief Abstract base class for list functors.
 */
template <typename T>
class ParquetFieldList : public ParquetField {
 protected:
  using read_func_type = std::function<bool(uint32_t, CompactProtocolReader*)>;
  std::vector<T>& val;
  FieldType const expected_type;
  read_func_type read_value;

  void bind_func(read_func_type fn) { read_value = fn; }

  ParquetFieldList(int f, std::vector<T>& v, FieldType t)
    : ParquetField(f), val(v), expected_type(t)
  {
  }

 public:
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) { return true; }
    uint8_t t;
    uint32_t const n = cpr->get_listh(&t);
    if (t != expected_type) { return true; }
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      if (read_value(i, cpr)) { return true; }
    }
    return false;
  }
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
struct ParquetFieldBoolList : public ParquetFieldList<bool> {
  ParquetFieldBoolList(int f, std::vector<bool>& v) : ParquetFieldList(f, v, ST_FLD_TRUE)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      auto const current_byte = cpr->getb();
      if (current_byte != ST_FLD_TRUE && current_byte != ST_FLD_FALSE) { return true; }
      this->val[i] = current_byte == ST_FLD_TRUE;
      return false;
    };
    bind_func(read_value);
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
template <typename T, FieldType EXPECTED_TYPE>
struct ParquetFieldIntList : public ParquetFieldList<T> {
  ParquetFieldIntList(int f, std::vector<T>& v) : ParquetFieldList<T>(f, v, EXPECTED_TYPE)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      this->val[i] = cpr->get_zigzag<T>();
      return false;
    };
    this->bind_func(read_value);
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
    if (field_type != ST_FLD_BINARY) { return true; }
    auto const n = cpr->get_u32();
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
struct ParquetFieldStringList : public ParquetFieldList<std::string> {
  ParquetFieldStringList(int f, std::vector<std::string>& v) : ParquetFieldList(f, v, ST_FLD_BINARY)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      auto const l = cpr->get_u32();
      if (l < static_cast<size_t>(cpr->m_end - cpr->m_cur)) {
        this->val[i].assign(reinterpret_cast<char const*>(cpr->m_cur), l);
        cpr->m_cur += l;
      } else {
        return true;
      }
      return false;
    };
    bind_func(read_value);
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
struct ParquetFieldEnumList : public ParquetFieldList<Enum> {
  ParquetFieldEnumList(int f, std::vector<Enum>& v) : ParquetFieldList<Enum>(f, v, ST_FLD_I32)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      this->val[i] = static_cast<Enum>(cpr->get_i32());
      return false;
    };
    this->bind_func(read_value);
  }
};

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
 * @brief Functor to read empty structures in unions
 *
 * @return True if field types mismatch
 */
template <typename T>
class ParquetFieldEmptyStruct : public ParquetField {
  T& val;

 public:
  ParquetFieldEmptyStruct(int f, T& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) { return true; }
    cpr->skip_struct_field(field_type);
    return false;
  }
};

/**
 * @brief Functor to read a vector of structures from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
struct ParquetFieldStructList : public ParquetFieldList<T> {
  ParquetFieldStructList(int f, std::vector<T>& v) : ParquetFieldList<T>(f, v, ST_FLD_STRUCT)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      if (not cpr->read(&this->val[i])) { return true; }
      return false;
    };
    this->bind_func(read_value);
  }
};

// TODO(ets): replace current union handling (which mirrors thrift) to use std::optional fields
// in a struct
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
    if (field_type != ST_FLD_BINARY) { return true; }
    auto const n = cpr->get_u32();
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
struct ParquetFieldBinaryList : public ParquetFieldList<std::vector<uint8_t>> {
  ParquetFieldBinaryList(int f, std::vector<std::vector<uint8_t>>& v)
    : ParquetFieldList(f, v, ST_FLD_BINARY)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      auto const l = cpr->get_u32();
      if (l <= static_cast<size_t>(cpr->m_end - cpr->m_cur)) {
        val[i].resize(l);
        val[i].assign(cpr->m_cur, cpr->m_cur + l);
        cpr->m_cur += l;
      } else {
        return true;
      }
      return false;
    };
    bind_func(read_value);
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
    if (field_type != ST_FLD_STRUCT) { return true; }
    uint8_t const* const start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
    return false;
  }
};

/**
 * @brief functor to wrap functors for optional fields
 */
template <typename T, typename FieldFunctor>
class ParquetFieldOptional : public ParquetField {
  std::optional<T>& val;

 public:
  ParquetFieldOptional(int f, std::optional<T>& v) : ParquetField(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    bool const res = FieldFunctor(field_val, v).operator()(cpr, field_type);
    if (!res) { val = v; }
    return res;
  }
};

// mapping of binary protocol field types to compact protocol field types
uint8_t const CompactProtocolReader::g_list2struct[16] = {0,
                                                          1,
                                                          2,
                                                          ST_FLD_BYTE,
                                                          ST_FLD_DOUBLE,
                                                          5,
                                                          ST_FLD_I16,
                                                          7,
                                                          ST_FLD_I32,
                                                          9,
                                                          ST_FLD_I64,
                                                          ST_FLD_BINARY,
                                                          ST_FLD_STRUCT,
                                                          ST_FLD_MAP,
                                                          ST_FLD_SET,
                                                          ST_FLD_LIST};

/**
 * @brief Skips the number of bytes according to the specified struct type
 *
 * @param[in] t Struct type enumeration
 * @param[in] depth Level of struct nesting
 *
 * @return True if the struct type is recognized, false otherwise
 */
bool CompactProtocolReader::skip_struct_field(int t, int depth)
{
  switch (t) {
    case ST_FLD_TRUE:
    case ST_FLD_FALSE: break;
    case ST_FLD_I16:
    case ST_FLD_I32:
    case ST_FLD_I64: get_u64(); break;
    case ST_FLD_BYTE: skip_bytes(1); break;
    case ST_FLD_DOUBLE: skip_bytes(8); break;
    case ST_FLD_BINARY: skip_bytes(get_u32()); break;
    // FIXME: this likely won't work, it's using the binary protocol not the compact. n should
    // be get_u32() (varint) and t should not be translated.
    case ST_FLD_LIST: [[fallthrough]];
    case ST_FLD_SET: {
      int const c = getb();
      int n       = c >> 4;
      if (n == 0xf) { n = get_i32(); }
      t = g_list2struct[c & 0xf];
      if (depth > 10) { return false; }
      for (int32_t i = 0; i < n; i++) {
        skip_struct_field(t, depth + 1);
      }
    } break;
    case ST_FLD_STRUCT:
      for (;;) {
        int const c = getb();
        t           = c & 0xf;
        if (!c) { break; }
        if (depth > 10) { return false; }
        skip_struct_field(t, depth + 1);
      }
      break;
    default:
      // printf("unsupported skip for type %d\n", t);
      break;
  }
  return true;
}

template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline bool run(CompactProtocolReader* cpr,
                         int field_type,
                         int const& field,
                         std::tuple<Operator...>& ops)
  {
    if (field == std::get<index>(ops).field()) {
      return std::get<index>(ops)(cpr, field_type);
    } else {
      return FunctionSwitchImpl<index - 1>::run(cpr, field_type, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline bool run(CompactProtocolReader* cpr,
                         int field_type,
                         int const& field,
                         std::tuple<Operator...>& ops)
  {
    if (field == std::get<0>(ops).field()) {
      return std::get<0>(ops)(cpr, field_type);
    } else {
      cpr->skip_struct_field(field_type);
      return false;
    }
  }
};

template <typename... Operator>
inline bool function_builder(CompactProtocolReader* cpr, std::tuple<Operator...>& op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  int field           = 0;
  while (true) {
    int const current_byte = cpr->getb();
    if (!current_byte) { break; }
    int const field_delta    = current_byte >> 4;
    int const field_type     = current_byte & 0xf;
    field                    = field_delta ? field + field_delta : cpr->get_i16();
    bool const exit_function = FunctionSwitchImpl<index>::run(cpr, field_type, field, op);
    if (exit_function) { return false; }
  }
  return true;
}

bool CompactProtocolReader::read(FileMetaData* f)
{
  using OptionalListColumnOrder =
    ParquetFieldOptional<std::vector<ColumnOrder>, ParquetFieldStructList<ColumnOrder>>;
  auto op = std::make_tuple(ParquetFieldInt32(1, f->version),
                            ParquetFieldStructList(2, f->schema),
                            ParquetFieldInt64(3, f->num_rows),
                            ParquetFieldStructList(4, f->row_groups),
                            ParquetFieldStructList(5, f->key_value_metadata),
                            ParquetFieldString(6, f->created_by),
                            OptionalListColumnOrder(7, f->column_orders));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(SchemaElement* s)
{
  auto op = std::make_tuple(ParquetFieldEnum<Type>(1, s->type),
                            ParquetFieldInt32(2, s->type_length),
                            ParquetFieldEnum<FieldRepetitionType>(3, s->repetition_type),
                            ParquetFieldString(4, s->name),
                            ParquetFieldInt32(5, s->num_children),
                            ParquetFieldEnum<ConvertedType>(6, s->converted_type),
                            ParquetFieldInt32(7, s->decimal_scale),
                            ParquetFieldInt32(8, s->decimal_precision),
                            ParquetFieldOptional<int32_t, ParquetFieldInt32>(9, s->field_id),
                            ParquetFieldStruct(10, s->logical_type));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(LogicalType* l)
{
  auto op =
    std::make_tuple(ParquetFieldUnion(1, l->isset.STRING, l->STRING),
                    ParquetFieldUnion(2, l->isset.MAP, l->MAP),
                    ParquetFieldUnion(3, l->isset.LIST, l->LIST),
                    ParquetFieldUnion(4, l->isset.ENUM, l->ENUM),
                    ParquetFieldUnion(5, l->isset.DECIMAL, l->DECIMAL),      // read the struct
                    ParquetFieldUnion(6, l->isset.DATE, l->DATE),
                    ParquetFieldUnion(7, l->isset.TIME, l->TIME),            //  read the struct
                    ParquetFieldUnion(8, l->isset.TIMESTAMP, l->TIMESTAMP),  //  read the struct
                    ParquetFieldUnion(10, l->isset.INTEGER, l->INTEGER),     //  read the struct
                    ParquetFieldUnion(11, l->isset.UNKNOWN, l->UNKNOWN),
                    ParquetFieldUnion(12, l->isset.JSON, l->JSON),
                    ParquetFieldUnion(13, l->isset.BSON, l->BSON));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DecimalType* d)
{
  auto op = std::make_tuple(ParquetFieldInt32(1, d->scale), ParquetFieldInt32(2, d->precision));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimeType* t)
{
  auto op =
    std::make_tuple(ParquetFieldBool(1, t->isAdjustedToUTC), ParquetFieldStruct(2, t->unit));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimestampType* t)
{
  auto op =
    std::make_tuple(ParquetFieldBool(1, t->isAdjustedToUTC), ParquetFieldStruct(2, t->unit));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimeUnit* u)
{
  auto op = std::make_tuple(ParquetFieldUnion(1, u->isset.MILLIS, u->MILLIS),
                            ParquetFieldUnion(2, u->isset.MICROS, u->MICROS),
                            ParquetFieldUnion(3, u->isset.NANOS, u->NANOS));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(IntType* i)
{
  auto op = std::make_tuple(ParquetFieldInt8(1, i->bitWidth), ParquetFieldBool(2, i->isSigned));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(RowGroup* r)
{
  auto op = std::make_tuple(ParquetFieldStructList(1, r->columns),
                            ParquetFieldInt64(2, r->total_byte_size),
                            ParquetFieldInt64(3, r->num_rows));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnChunk* c)
{
  auto op = std::make_tuple(ParquetFieldString(1, c->file_path),
                            ParquetFieldInt64(2, c->file_offset),
                            ParquetFieldStruct(3, c->meta_data),
                            ParquetFieldInt64(4, c->offset_index_offset),
                            ParquetFieldInt32(5, c->offset_index_length),
                            ParquetFieldInt64(6, c->column_index_offset),
                            ParquetFieldInt32(7, c->column_index_length));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnChunkMetaData* c)
{
  auto op = std::make_tuple(ParquetFieldEnum<Type>(1, c->type),
                            ParquetFieldEnumList(2, c->encodings),
                            ParquetFieldStringList(3, c->path_in_schema),
                            ParquetFieldEnum<Compression>(4, c->codec),
                            ParquetFieldInt64(5, c->num_values),
                            ParquetFieldInt64(6, c->total_uncompressed_size),
                            ParquetFieldInt64(7, c->total_compressed_size),
                            ParquetFieldInt64(9, c->data_page_offset),
                            ParquetFieldInt64(10, c->index_page_offset),
                            ParquetFieldInt64(11, c->dictionary_page_offset),
                            ParquetFieldStruct(12, c->statistics));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(PageHeader* p)
{
  auto op = std::make_tuple(ParquetFieldEnum<PageType>(1, p->type),
                            ParquetFieldInt32(2, p->uncompressed_page_size),
                            ParquetFieldInt32(3, p->compressed_page_size),
                            ParquetFieldStruct(5, p->data_page_header),
                            ParquetFieldStruct(7, p->dictionary_page_header),
                            ParquetFieldStruct(8, p->data_page_header_v2));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DataPageHeader* d)
{
  auto op = std::make_tuple(ParquetFieldInt32(1, d->num_values),
                            ParquetFieldEnum<Encoding>(2, d->encoding),
                            ParquetFieldEnum<Encoding>(3, d->definition_level_encoding),
                            ParquetFieldEnum<Encoding>(4, d->repetition_level_encoding));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DictionaryPageHeader* d)
{
  auto op = std::make_tuple(ParquetFieldInt32(1, d->num_values),
                            ParquetFieldEnum<Encoding>(2, d->encoding));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DataPageHeaderV2* d)
{
  auto op = std::make_tuple(ParquetFieldInt32(1, d->num_values),
                            ParquetFieldInt32(2, d->num_nulls),
                            ParquetFieldInt32(3, d->num_rows),
                            ParquetFieldEnum<Encoding>(4, d->encoding),
                            ParquetFieldInt32(5, d->definition_levels_byte_length),
                            ParquetFieldInt32(6, d->repetition_levels_byte_length),
                            ParquetFieldBool(7, d->is_compressed));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(KeyValue* k)
{
  auto op = std::make_tuple(ParquetFieldString(1, k->key), ParquetFieldString(2, k->value));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(PageLocation* p)
{
  auto op = std::make_tuple(ParquetFieldInt64(1, p->offset),
                            ParquetFieldInt32(2, p->compressed_page_size),
                            ParquetFieldInt64(3, p->first_row_index));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(OffsetIndex* o)
{
  auto op = std::make_tuple(ParquetFieldStructList(1, o->page_locations));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnIndex* c)
{
  auto op = std::make_tuple(ParquetFieldBoolList(1, c->null_pages),
                            ParquetFieldBinaryList(2, c->min_values),
                            ParquetFieldBinaryList(3, c->max_values),
                            ParquetFieldEnum<BoundaryOrder>(4, c->boundary_order),
                            ParquetFieldInt64List(5, c->null_counts));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(Statistics* s)
{
  auto op = std::make_tuple(ParquetFieldBinary(1, s->max),
                            ParquetFieldBinary(2, s->min),
                            ParquetFieldInt64(3, s->null_count),
                            ParquetFieldInt64(4, s->distinct_count),
                            ParquetFieldBinary(5, s->max_value),
                            ParquetFieldBinary(6, s->min_value));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnOrder* c)
{
  using OptionalTypeDefined =
    ParquetFieldOptional<TypeDefinedOrder, ParquetFieldEmptyStruct<TypeDefinedOrder>>;
  auto op = std::make_tuple(OptionalTypeDefined(1, c->TYPE_ORDER));
  return function_builder(this, op);
}

/**
 * @brief Constructs the schema from the file-level metadata
 *
 * @param[in] md File metadata that was previously parsed
 *
 * @return True if schema constructed completely, false otherwise
 */
bool CompactProtocolReader::InitSchema(FileMetaData* md)
{
  if (static_cast<std::size_t>(WalkSchema(md)) != md->schema.size()) { return false; }

  /* Inside FileMetaData, there is a std::vector of RowGroups and each RowGroup contains a
   * a std::vector of ColumnChunks. Each ColumnChunk has a member ColumnMetaData, which contains
   * a std::vector of std::strings representing paths. The purpose of the code below is to set the
   * schema_idx of each column of each row to it corresponding row_group. This is effectively
   * mapping the columns to the schema.
   */
  for (auto& row_group : md->row_groups) {
    int current_schema_index = 0;
    for (auto& column : row_group.columns) {
      int parent = 0;  // root of schema
      for (auto const& path : column.meta_data.path_in_schema) {
        auto const it = [&] {
          // find_if starting at (current_schema_index + 1) and then wrapping
          auto const schema = [&](auto const& e) {
            return e.parent_idx == parent && e.name == path;
          };
          auto const mid = md->schema.cbegin() + current_schema_index + 1;
          auto const it  = std::find_if(mid, md->schema.cend(), schema);
          if (it != md->schema.cend()) { return it; }
          return std::find_if(md->schema.cbegin(), mid, schema);
        }();
        if (it == md->schema.cend()) { return false; }
        current_schema_index = std::distance(md->schema.cbegin(), it);
        column.schema_idx    = current_schema_index;
        parent               = current_schema_index;
      }
    }
  }

  return true;
}

/**
 * @brief Populates each node in the schema tree
 *
 * @param[out] md File metadata
 * @param[in] idx Current node index
 * @param[in] parent_idx Parent node index
 * @param[in] max_def_level Max definition level
 * @param[in] max_rep_level Max repetition level
 *
 * @return The node index that was populated
 */
int CompactProtocolReader::WalkSchema(
  FileMetaData* md, int idx, int parent_idx, int max_def_level, int max_rep_level)
{
  if (idx >= 0 && (size_t)idx < md->schema.size()) {
    SchemaElement* e = &md->schema[idx];
    if (e->repetition_type == OPTIONAL) {
      ++max_def_level;
    } else if (e->repetition_type == REPEATED) {
      ++max_def_level;
      ++max_rep_level;
    }
    e->max_definition_level = max_def_level;
    e->max_repetition_level = max_rep_level;
    e->parent_idx           = parent_idx;

    parent_idx = idx;
    ++idx;
    if (e->num_children > 0) {
      for (int i = 0; i < e->num_children; i++) {
        e->children_idx.push_back(idx);
        int const idx_old = idx;
        idx               = WalkSchema(md, idx, parent_idx, max_def_level, max_rep_level);
        if (idx <= idx_old) { break; }  // Error
      }
    }
    return idx;
  } else {
    // Error
    return -1;
  }
}

}  // namespace parquet
}  // namespace io
}  // namespace cudf
