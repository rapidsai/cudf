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

#include "compact_protocol_writer.hpp"

namespace cudf {
namespace io {
namespace parquet {

/**
 * @brief Parquet CompactProtocolWriter class
 */

size_t CompactProtocolWriter::write(const FileMetaData& f)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, f.version);
  c.field_struct_list(2, f.schema);
  c.field_int(3, f.num_rows);
  c.field_struct_list(4, f.row_groups);
  if (f.key_value_metadata.size() != 0) { c.field_struct_list(5, f.key_value_metadata); }
  if (f.created_by.size() != 0) { c.field_string(6, f.created_by); }
  if (f.column_order_listsize != 0) {
    // Dummy list of struct containing an empty field1 struct
    c.put_field_header(7, c.current_field(), ST_FLD_LIST);
    c.put_byte((uint8_t)((std::min(f.column_order_listsize, 0xfu) << 4) | ST_FLD_STRUCT));
    if (f.column_order_listsize >= 0xf) c.put_uint(f.column_order_listsize);
    for (uint32_t i = 0; i < f.column_order_listsize; i++) {
      c.put_field_header(1, 0, ST_FLD_STRUCT);
      c.put_byte(0);  // ColumnOrder.field1 struct end
      c.put_byte(0);  // ColumnOrder struct end
    }
    c.set_current_field(7);
  }
  return c.value();
}

size_t CompactProtocolWriter::write(const DecimalType& decimal)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, decimal.scale);
  c.field_int(2, decimal.precision);
  return c.value();
}

size_t CompactProtocolWriter::write(const TimeUnit& time_unit)
{
  CompactProtocolFieldWriter c(*this);
  auto const isset = time_unit.isset;
  if (isset.MILLIS) {
    c.field_struct(1, time_unit.MILLIS);
  } else if (isset.MICROS) {
    c.field_struct(2, time_unit.MICROS);
  } else if (isset.NANOS) {
    c.field_struct(3, time_unit.NANOS);
  }
  return c.value();
}

size_t CompactProtocolWriter::write(const TimeType& time)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, time.isAdjustedToUTC);
  c.field_struct(2, time.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(const TimestampType& timestamp)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, timestamp.isAdjustedToUTC);
  c.field_struct(2, timestamp.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(const IntType& integer)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int8(1, integer.bitWidth);
  c.field_bool(2, integer.isSigned);
  return c.value();
}

size_t CompactProtocolWriter::write(const LogicalType& logical_type)
{
  CompactProtocolFieldWriter c(*this);
  auto const isset = logical_type.isset;
  if (isset.STRING) {
    c.field_struct(1, logical_type.STRING);
  } else if (isset.MAP) {
    c.field_struct(2, logical_type.MAP);
  } else if (isset.LIST) {
    c.field_struct(3, logical_type.LIST);
  } else if (isset.ENUM) {
    c.field_struct(4, logical_type.ENUM);
  } else if (isset.DECIMAL) {
    c.field_struct(5, logical_type.DECIMAL);
  } else if (isset.DATE) {
    c.field_struct(6, logical_type.DATE);
  } else if (isset.TIME) {
    c.field_struct(7, logical_type.TIME);
  } else if (isset.TIMESTAMP) {
    c.field_struct(8, logical_type.TIMESTAMP);
  } else if (isset.INTEGER) {
    c.field_struct(10, logical_type.INTEGER);
  } else if (isset.UNKNOWN) {
    c.field_struct(11, logical_type.UNKNOWN);
  } else if (isset.JSON) {
    c.field_struct(12, logical_type.JSON);
  } else if (isset.BSON) {
    c.field_struct(13, logical_type.BSON);
  }
  return c.value();
}

size_t CompactProtocolWriter::write(const SchemaElement& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.type != UNDEFINED_TYPE) {
    c.field_int(1, s.type);
    if (s.type_length != 0) { c.field_int(2, s.type_length); }
  }
  if (s.repetition_type != NO_REPETITION_TYPE) { c.field_int(3, s.repetition_type); }
  c.field_string(4, s.name);

  if (s.type == UNDEFINED_TYPE) { c.field_int(5, s.num_children); }
  if (s.converted_type != UNKNOWN) {
    c.field_int(6, s.converted_type);
    if (s.converted_type == DECIMAL) {
      c.field_int(7, s.decimal_scale);
      c.field_int(8, s.decimal_precision);
    }
  }
  if (s.field_id) { c.field_int(9, s.field_id.value()); }
  auto const isset = s.logical_type.isset;
  // TODO: add handling for all logical types
  // if (isset.STRING or isset.MAP or isset.LIST or isset.ENUM or isset.DECIMAL or isset.DATE or
  //    isset.TIME or isset.TIMESTAMP or isset.INTEGER or isset.UNKNOWN or isset.JSON or isset.BSON)
  //    {
  if (isset.TIMESTAMP or isset.TIME) { c.field_struct(10, s.logical_type); }
  return c.value();
}

size_t CompactProtocolWriter::write(const RowGroup& r)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, r.columns);
  c.field_int(2, r.total_byte_size);
  c.field_int(3, r.num_rows);
  return c.value();
}

size_t CompactProtocolWriter::write(const KeyValue& k)
{
  CompactProtocolFieldWriter c(*this);
  c.field_string(1, k.key);
  if (k.value.size() != 0) { c.field_string(2, k.value); }
  return c.value();
}

size_t CompactProtocolWriter::write(const ColumnChunk& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.file_path.size() != 0) { c.field_string(1, s.file_path); }
  c.field_int(2, s.file_offset);
  c.field_struct(3, s.meta_data);
  if (s.offset_index_length != 0) {
    c.field_int(4, s.offset_index_offset);
    c.field_int(5, s.offset_index_length);
  }
  if (s.column_index_length != 0) {
    c.field_int(6, s.column_index_offset);
    c.field_int(7, s.column_index_length);
  }
  return c.value();
}

size_t CompactProtocolWriter::write(const ColumnChunkMetaData& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, s.type);
  c.field_int_list(2, s.encodings);
  c.field_string_list(3, s.path_in_schema);
  c.field_int(4, s.codec);
  c.field_int(5, s.num_values);
  c.field_int(6, s.total_uncompressed_size);
  c.field_int(7, s.total_compressed_size);
  c.field_int(9, s.data_page_offset);
  if (s.index_page_offset != 0) { c.field_int(10, s.index_page_offset); }
  if (s.dictionary_page_offset != 0) { c.field_int(11, s.dictionary_page_offset); }
  if (s.statistics_blob.size() != 0) { c.field_struct_blob(12, s.statistics_blob); }
  return c.value();
}

size_t CompactProtocolWriter::write(const PageLocation& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, s.offset);
  c.field_int(2, s.compressed_page_size);
  c.field_int(3, s.first_row_index);
  return c.value();
}

size_t CompactProtocolWriter::write(const OffsetIndex& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, s.page_locations);
  return c.value();
}

void CompactProtocolFieldWriter::put_byte(uint8_t v) { writer.m_buf.push_back(v); }

void CompactProtocolFieldWriter::put_byte(const uint8_t* raw, uint32_t len)
{
  for (uint32_t i = 0; i < len; i++)
    writer.m_buf.push_back(raw[i]);
}

uint32_t CompactProtocolFieldWriter::put_uint(uint64_t v)
{
  int l = 1;
  while (v > 0x7f) {
    put_byte(static_cast<uint8_t>(v | 0x80));
    v >>= 7;
    l++;
  }
  put_byte(static_cast<uint8_t>(v));
  return l;
}

uint32_t CompactProtocolFieldWriter::put_int(int64_t v)
{
  int64_t s = (v < 0);
  return put_uint(((v ^ -s) << 1) + s);
}

void CompactProtocolFieldWriter::put_field_header(int f, int cur, int t)
{
  if (f > cur && f <= cur + 15)
    put_byte(((f - cur) << 4) | t);
  else {
    put_byte(t);
    put_int(f);
  }
}

inline void CompactProtocolFieldWriter::field_bool(int field, bool b)
{
  put_field_header(field, current_field_value, b ? ST_FLD_TRUE : ST_FLD_FALSE);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int8(int field, int8_t val)
{
  put_field_header(field, current_field_value, ST_FLD_BYTE);
  put_byte(val);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int(int field, int32_t val)
{
  put_field_header(field, current_field_value, ST_FLD_I32);
  put_int(val);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int(int field, int64_t val)
{
  put_field_header(field, current_field_value, ST_FLD_I64);
  put_int(val);
  current_field_value = field;
}

template <typename Enum>
inline void CompactProtocolFieldWriter::field_int_list(int field, const std::vector<Enum>& val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_I32));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    put_int(static_cast<int32_t>(v));
  }
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct(int field, const T& val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  if constexpr (not std::is_empty_v<T>) {
    writer.write(val);  // write the struct if it's not empty
  } else {
    put_byte(0);  // otherwise, add a stop field
  }
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct_list(int field, const std::vector<T>& val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_STRUCT));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    writer.write(v);
  }
  current_field_value = field;
}

inline size_t CompactProtocolFieldWriter::value()
{
  put_byte(0);
  return writer.m_buf.size() - struct_start_pos;
}

inline void CompactProtocolFieldWriter::field_struct_blob(int field,
                                                          const std::vector<uint8_t>& val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  put_byte(val.data(), (uint32_t)val.size());
  put_byte(0);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string(int field, const std::string& val)
{
  put_field_header(field, current_field_value, ST_FLD_BINARY);
  put_uint(val.size());
  // FIXME : replace reinterpret_cast
  put_byte(reinterpret_cast<const uint8_t*>(val.data()), (uint32_t)val.size());
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string_list(int field,
                                                          const std::vector<std::string>& val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_BINARY));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    put_uint(v.size());
    // FIXME : replace reinterpret_cast
    put_byte(reinterpret_cast<const uint8_t*>(v.data()), (uint32_t)v.size());
  }
  current_field_value = field;
}

inline int CompactProtocolFieldWriter::current_field() { return current_field_value; }

inline void CompactProtocolFieldWriter::set_current_field(const int& field)
{
  current_field_value = field;
}

}  // namespace parquet
}  // namespace io
}  // namespace cudf
