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

namespace cudf::io::parquet::detail {

/**
 * @brief Base class for parquet field functors.
 *
 * Holds the field value used by all of the specialized functors.
 */
class parquet_field {
 private:
  int _field_val;

 protected:
  parquet_field(int f) : _field_val(f) {}

 public:
  virtual ~parquet_field() = default;
  int field() const { return _field_val; }
};

/**
 * @brief Abstract base class for list functors.
 */
template <typename T>
class parquet_field_list : public parquet_field {
 private:
  using read_func_type = std::function<bool(uint32_t, CompactProtocolReader*)>;
  FieldType _expected_type;
  read_func_type _read_value;

 protected:
  std::vector<T>& val;

  void bind_read_func(read_func_type fn) { _read_value = fn; }

  parquet_field_list(int f, std::vector<T>& v, FieldType t)
    : parquet_field(f), _expected_type(t), val(v)
  {
  }

 public:
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_LIST) { return true; }
    auto const [t, n] = cpr->get_listh();
    if (t != _expected_type) { return true; }
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      if (_read_value(i, cpr)) { return true; }
    }
    return false;
  }
};

/**
 * @brief Functor to set value to bool read from CompactProtocolReader
 *
 * bool doesn't actually encode a value, we just use the field type to indicate true/false
 *
 * @return True if field type is not bool
 */
class parquet_field_bool : public parquet_field {
  bool& val;

 public:
  parquet_field_bool(int f, bool& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_TRUE && field_type != ST_FLD_FALSE) { return true; }
    val = field_type == ST_FLD_TRUE;
    return false;
  }
};

/**
 * @brief Functor to read a vector of booleans from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * bool fails
 */
struct parquet_field_bool_list : public parquet_field_list<bool> {
  parquet_field_bool_list(int f, std::vector<bool>& v) : parquet_field_list(f, v, ST_FLD_TRUE)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      auto const current_byte = cpr->getb();
      if (current_byte != ST_FLD_TRUE && current_byte != ST_FLD_FALSE) { return true; }
      this->val[i] = current_byte == ST_FLD_TRUE;
      return false;
    };
    bind_read_func(read_value);
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
class parquet_field_int : public parquet_field {
  static constexpr bool is_byte = std::is_same_v<T, int8_t>;

  T& val;

 public:
  parquet_field_int(int f, T& v) : parquet_field(f), val(v) {}

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

using parquet_field_int8  = parquet_field_int<int8_t, ST_FLD_BYTE>;
using parquet_field_int32 = parquet_field_int<int32_t, ST_FLD_I32>;
using parquet_field_int64 = parquet_field_int<int64_t, ST_FLD_I64>;

/**
 * @brief Functor to read a vector of integers from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * integer fails
 */
template <typename T, FieldType EXPECTED_TYPE>
struct parquet_field_int_list : public parquet_field_list<T> {
  parquet_field_int_list(int f, std::vector<T>& v) : parquet_field_list<T>(f, v, EXPECTED_TYPE)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      this->val[i] = cpr->get_zigzag<T>();
      return false;
    };
    this->bind_read_func(read_value);
  }
};

using parquet_field_int64_list = parquet_field_int_list<int64_t, ST_FLD_I64>;

/**
 * @brief Functor to read a string from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of string exceeds bounds
 * of the CompactProtocolReader
 */
class parquet_field_string : public parquet_field {
  std::string& val;

 public:
  parquet_field_string(int f, std::string& v) : parquet_field(f), val(v) {}

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
struct parquet_field_string_list : public parquet_field_list<std::string> {
  parquet_field_string_list(int f, std::vector<std::string>& v)
    : parquet_field_list(f, v, ST_FLD_BINARY)
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
    bind_read_func(read_value);
  }
};

/**
 * @brief Functor to set value to enum read from CompactProtocolReader
 *
 * @return True if field type is not int32
 */
template <typename Enum>
class parquet_field_enum : public parquet_field {
  Enum& val;

 public:
  parquet_field_enum(int f, Enum& v) : parquet_field(f), val(v) {}
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
struct parquet_field_enum_list : public parquet_field_list<Enum> {
  parquet_field_enum_list(int f, std::vector<Enum>& v) : parquet_field_list<Enum>(f, v, ST_FLD_I32)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      this->val[i] = static_cast<Enum>(cpr->get_i32());
      return false;
    };
    this->bind_read_func(read_value);
  }
};

/**
 * @brief Functor to read a structure from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class parquet_field_struct : public parquet_field {
  T& val;

 public:
  parquet_field_struct(int f, T& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    return (field_type != ST_FLD_STRUCT || !(cpr->read(&val)));
  }
};

/**
 * @brief Functor to read optional structures in unions
 *
 * @return True if field types mismatch
 */
template <typename E, typename T>
class parquet_field_union_struct : public parquet_field {
  E& enum_val;
  thrust::optional<T>& val;  // union structs are always wrapped in std::optional

 public:
  parquet_field_union_struct(int f, E& ev, thrust::optional<T>& v)
    : parquet_field(f), enum_val(ev), val(v)
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    bool const res = parquet_field_struct<T>(field(), v).operator()(cpr, field_type);
    if (!res) {
      val      = v;
      enum_val = static_cast<E>(field());
    }
    return res;
  }
};

/**
 * @brief Functor to read empty structures in unions
 *
 * Added to avoid having to define read() functions for empty structs contained in unions.
 *
 * @return True if field types mismatch
 */
template <typename E>
class parquet_field_union_enumerator : public parquet_field {
  E& val;

 public:
  parquet_field_union_enumerator(int f, E& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (field_type != ST_FLD_STRUCT) { return true; }
    cpr->skip_struct_field(field_type);
    val = static_cast<E>(field());
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
struct parquet_field_struct_list : public parquet_field_list<T> {
  parquet_field_struct_list(int f, std::vector<T>& v) : parquet_field_list<T>(f, v, ST_FLD_STRUCT)
  {
    auto const read_value = [this](uint32_t i, CompactProtocolReader* cpr) {
      if (not cpr->read(&this->val[i])) { return true; }
      return false;
    };
    this->bind_read_func(read_value);
  }
};

/**
 * @brief Functor to read a binary from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of binary exceeds bounds
 * of the CompactProtocolReader
 */
class parquet_field_binary : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_binary(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}

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
struct parquet_field_binary_list : public parquet_field_list<std::vector<uint8_t>> {
  parquet_field_binary_list(int f, std::vector<std::vector<uint8_t>>& v)
    : parquet_field_list(f, v, ST_FLD_BINARY)
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
    bind_read_func(read_value);
  }
};

/**
 * @brief Functor to read a struct from CompactProtocolReader
 *
 * @return True if field type mismatches
 */
class parquet_field_struct_blob : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_struct_blob(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}
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
class parquet_field_optional : public parquet_field {
  thrust::optional<T>& val;

 public:
  parquet_field_optional(int f, thrust::optional<T>& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    bool const res = FieldFunctor(field(), v).operator()(cpr, field_type);
    if (!res) { val = v; }
    return res;
  }
};

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
    case ST_FLD_LIST: [[fallthrough]];
    case ST_FLD_SET: {
      auto const [t, n] = get_listh();
      if (depth > 10) { return false; }
      for (uint32_t i = 0; i < n; i++) {
        skip_struct_field(t, depth + 1);
      }
    } break;
    case ST_FLD_STRUCT:
      for (;;) {
        int const c = getb();
        t           = c & 0xf;
        if (c == 0) { break; }               // end of struct
        if ((c & 0xf0) == 0) { get_i16(); }  // field id is not a delta
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
  using optional_list_column_order =
    parquet_field_optional<std::vector<ColumnOrder>, parquet_field_struct_list<ColumnOrder>>;
  auto op = std::make_tuple(parquet_field_int32(1, f->version),
                            parquet_field_struct_list(2, f->schema),
                            parquet_field_int64(3, f->num_rows),
                            parquet_field_struct_list(4, f->row_groups),
                            parquet_field_struct_list(5, f->key_value_metadata),
                            parquet_field_string(6, f->created_by),
                            optional_list_column_order(7, f->column_orders));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(SchemaElement* s)
{
  using optional_converted_type =
    parquet_field_optional<ConvertedType, parquet_field_enum<ConvertedType>>;
  using optional_logical_type =
    parquet_field_optional<LogicalType, parquet_field_struct<LogicalType>>;
  auto op = std::make_tuple(parquet_field_enum<Type>(1, s->type),
                            parquet_field_int32(2, s->type_length),
                            parquet_field_enum<FieldRepetitionType>(3, s->repetition_type),
                            parquet_field_string(4, s->name),
                            parquet_field_int32(5, s->num_children),
                            optional_converted_type(6, s->converted_type),
                            parquet_field_int32(7, s->decimal_scale),
                            parquet_field_int32(8, s->decimal_precision),
                            parquet_field_optional<int32_t, parquet_field_int32>(9, s->field_id),
                            optional_logical_type(10, s->logical_type));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(LogicalType* l)
{
  auto op = std::make_tuple(
    parquet_field_union_enumerator(1, l->type),
    parquet_field_union_enumerator(2, l->type),
    parquet_field_union_enumerator(3, l->type),
    parquet_field_union_enumerator(4, l->type),
    parquet_field_union_struct<LogicalType::Type, DecimalType>(5, l->type, l->decimal_type),
    parquet_field_union_enumerator(6, l->type),
    parquet_field_union_struct<LogicalType::Type, TimeType>(7, l->type, l->time_type),
    parquet_field_union_struct<LogicalType::Type, TimestampType>(8, l->type, l->timestamp_type),
    parquet_field_union_struct<LogicalType::Type, IntType>(10, l->type, l->int_type),
    parquet_field_union_enumerator(11, l->type),
    parquet_field_union_enumerator(12, l->type),
    parquet_field_union_enumerator(13, l->type));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DecimalType* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->scale), parquet_field_int32(2, d->precision));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimeType* t)
{
  auto op =
    std::make_tuple(parquet_field_bool(1, t->isAdjustedToUTC), parquet_field_struct(2, t->unit));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimestampType* t)
{
  auto op =
    std::make_tuple(parquet_field_bool(1, t->isAdjustedToUTC), parquet_field_struct(2, t->unit));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(TimeUnit* u)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, u->type),
                            parquet_field_union_enumerator(2, u->type),
                            parquet_field_union_enumerator(3, u->type));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(IntType* i)
{
  auto op = std::make_tuple(parquet_field_int8(1, i->bitWidth), parquet_field_bool(2, i->isSigned));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(RowGroup* r)
{
  auto op = std::make_tuple(parquet_field_struct_list(1, r->columns),
                            parquet_field_int64(2, r->total_byte_size),
                            parquet_field_int64(3, r->num_rows));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnChunk* c)
{
  auto op = std::make_tuple(parquet_field_string(1, c->file_path),
                            parquet_field_int64(2, c->file_offset),
                            parquet_field_struct(3, c->meta_data),
                            parquet_field_int64(4, c->offset_index_offset),
                            parquet_field_int32(5, c->offset_index_length),
                            parquet_field_int64(6, c->column_index_offset),
                            parquet_field_int32(7, c->column_index_length));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnChunkMetaData* c)
{
  auto op = std::make_tuple(parquet_field_enum<Type>(1, c->type),
                            parquet_field_enum_list(2, c->encodings),
                            parquet_field_string_list(3, c->path_in_schema),
                            parquet_field_enum<Compression>(4, c->codec),
                            parquet_field_int64(5, c->num_values),
                            parquet_field_int64(6, c->total_uncompressed_size),
                            parquet_field_int64(7, c->total_compressed_size),
                            parquet_field_int64(9, c->data_page_offset),
                            parquet_field_int64(10, c->index_page_offset),
                            parquet_field_int64(11, c->dictionary_page_offset),
                            parquet_field_struct(12, c->statistics));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(PageHeader* p)
{
  auto op = std::make_tuple(parquet_field_enum<PageType>(1, p->type),
                            parquet_field_int32(2, p->uncompressed_page_size),
                            parquet_field_int32(3, p->compressed_page_size),
                            parquet_field_struct(5, p->data_page_header),
                            parquet_field_struct(7, p->dictionary_page_header),
                            parquet_field_struct(8, p->data_page_header_v2));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DataPageHeader* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_enum<Encoding>(2, d->encoding),
                            parquet_field_enum<Encoding>(3, d->definition_level_encoding),
                            parquet_field_enum<Encoding>(4, d->repetition_level_encoding));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DictionaryPageHeader* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_enum<Encoding>(2, d->encoding));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(DataPageHeaderV2* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_int32(2, d->num_nulls),
                            parquet_field_int32(3, d->num_rows),
                            parquet_field_enum<Encoding>(4, d->encoding),
                            parquet_field_int32(5, d->definition_levels_byte_length),
                            parquet_field_int32(6, d->repetition_levels_byte_length),
                            parquet_field_bool(7, d->is_compressed));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(KeyValue* k)
{
  auto op = std::make_tuple(parquet_field_string(1, k->key), parquet_field_string(2, k->value));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(PageLocation* p)
{
  auto op = std::make_tuple(parquet_field_int64(1, p->offset),
                            parquet_field_int32(2, p->compressed_page_size),
                            parquet_field_int64(3, p->first_row_index));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(OffsetIndex* o)
{
  auto op = std::make_tuple(parquet_field_struct_list(1, o->page_locations));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnIndex* c)
{
  auto op = std::make_tuple(parquet_field_bool_list(1, c->null_pages),
                            parquet_field_binary_list(2, c->min_values),
                            parquet_field_binary_list(3, c->max_values),
                            parquet_field_enum<BoundaryOrder>(4, c->boundary_order),
                            parquet_field_int64_list(5, c->null_counts));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(Statistics* s)
{
  using optional_binary = parquet_field_optional<std::vector<uint8_t>, parquet_field_binary>;
  using optional_int64  = parquet_field_optional<int64_t, parquet_field_int64>;

  auto op = std::make_tuple(optional_binary(1, s->max),
                            optional_binary(2, s->min),
                            optional_int64(3, s->null_count),
                            optional_int64(4, s->distinct_count),
                            optional_binary(5, s->max_value),
                            optional_binary(6, s->min_value));
  return function_builder(this, op);
}

bool CompactProtocolReader::read(ColumnOrder* c)
{
  auto op = std::make_tuple(parquet_field_union_enumerator<ColumnOrder::Type>(1, c->type));
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

}  // namespace cudf::io::parquet::detail
