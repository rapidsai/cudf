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

#include "compact_protocol_writer.hpp"

#include <cudf/utilities/error.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Parquet CompactProtocolWriter class
 */

size_t CompactProtocolWriter::write(FileMetaData const& f)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, f.version);
  c.field_struct_list(2, f.schema);
  c.field_int(3, f.num_rows);
  c.field_struct_list(4, f.row_groups);
  if (not f.key_value_metadata.empty()) { c.field_struct_list(5, f.key_value_metadata); }
  if (not f.created_by.empty()) { c.field_string(6, f.created_by); }
  if (f.column_orders.has_value()) { c.field_struct_list(7, f.column_orders.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(DecimalType const& decimal)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, decimal.scale);
  c.field_int(2, decimal.precision);
  return c.value();
}

size_t CompactProtocolWriter::write(TimeUnit const& time_unit)
{
  CompactProtocolFieldWriter c(*this);
  switch (time_unit.type) {
    case TimeUnit::MILLIS:
    case TimeUnit::MICROS:
    case TimeUnit::NANOS: c.field_empty_struct(time_unit.type); break;
    default: CUDF_FAIL("Trying to write an invalid TimeUnit " + std::to_string(time_unit.type));
  }
  return c.value();
}

size_t CompactProtocolWriter::write(TimeType const& time)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, time.isAdjustedToUTC);
  c.field_struct(2, time.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(TimestampType const& timestamp)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, timestamp.isAdjustedToUTC);
  c.field_struct(2, timestamp.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(IntType const& integer)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int8(1, integer.bitWidth);
  c.field_bool(2, integer.isSigned);
  return c.value();
}

size_t CompactProtocolWriter::write(LogicalType const& logical_type)
{
  CompactProtocolFieldWriter c(*this);
  switch (logical_type.type) {
    case LogicalType::STRING:
    case LogicalType::MAP:
    case LogicalType::LIST:
    case LogicalType::ENUM:
    case LogicalType::DATE:
    case LogicalType::UNKNOWN:
    case LogicalType::JSON:
    case LogicalType::BSON: c.field_empty_struct(logical_type.type); break;
    case LogicalType::DECIMAL:
      c.field_struct(LogicalType::DECIMAL, logical_type.decimal_type.value());
      break;
    case LogicalType::TIME:
      c.field_struct(LogicalType::TIME, logical_type.time_type.value());
      break;
    case LogicalType::TIMESTAMP:
      c.field_struct(LogicalType::TIMESTAMP, logical_type.timestamp_type.value());
      break;
    case LogicalType::INTEGER:
      c.field_struct(LogicalType::INTEGER, logical_type.int_type.value());
      break;
    default:
      CUDF_FAIL("Trying to write an invalid LogicalType " + std::to_string(logical_type.type));
  }
  return c.value();
}

size_t CompactProtocolWriter::write(SchemaElement const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.type != UNDEFINED_TYPE) {
    c.field_int(1, s.type);
    if (s.type_length != 0) { c.field_int(2, s.type_length); }
  }
  if (s.repetition_type != NO_REPETITION_TYPE) { c.field_int(3, s.repetition_type); }
  c.field_string(4, s.name);

  if (s.type == UNDEFINED_TYPE) { c.field_int(5, s.num_children); }
  if (s.converted_type.has_value()) {
    c.field_int(6, s.converted_type.value());
    if (s.converted_type == DECIMAL) {
      c.field_int(7, s.decimal_scale);
      c.field_int(8, s.decimal_precision);
    }
  }
  if (s.field_id.has_value()) { c.field_int(9, s.field_id.value()); }
  if (s.logical_type.has_value()) { c.field_struct(10, s.logical_type.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(RowGroup const& r)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, r.columns);
  c.field_int(2, r.total_byte_size);
  c.field_int(3, r.num_rows);
  return c.value();
}

size_t CompactProtocolWriter::write(KeyValue const& k)
{
  CompactProtocolFieldWriter c(*this);
  c.field_string(1, k.key);
  if (not k.value.empty()) { c.field_string(2, k.value); }
  return c.value();
}

size_t CompactProtocolWriter::write(ColumnChunk const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (not s.file_path.empty()) { c.field_string(1, s.file_path); }
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

size_t CompactProtocolWriter::write(ColumnChunkMetaData const& s)
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
  c.field_struct(12, s.statistics);
  return c.value();
}

size_t CompactProtocolWriter::write(Statistics const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.max.has_value()) { c.field_binary(1, s.max.value()); }
  if (s.min.has_value()) { c.field_binary(2, s.min.value()); }
  if (s.null_count.has_value()) { c.field_int(3, s.null_count.value()); }
  if (s.distinct_count.has_value()) { c.field_int(4, s.distinct_count.value()); }
  if (s.max_value.has_value()) { c.field_binary(5, s.max_value.value()); }
  if (s.min_value.has_value()) { c.field_binary(6, s.min_value.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(PageLocation const& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, s.offset);
  c.field_int(2, s.compressed_page_size);
  c.field_int(3, s.first_row_index);
  return c.value();
}

size_t CompactProtocolWriter::write(OffsetIndex const& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, s.page_locations);
  return c.value();
}

size_t CompactProtocolWriter::write(ColumnOrder const& co)
{
  CompactProtocolFieldWriter c(*this);
  switch (co.type) {
    case ColumnOrder::TYPE_ORDER: c.field_empty_struct(co.type); break;
    default: CUDF_FAIL("Trying to write an invalid ColumnOrder " + std::to_string(co.type));
  }
  return c.value();
}

void CompactProtocolFieldWriter::put_byte(uint8_t v) { writer.m_buf.push_back(v); }

void CompactProtocolFieldWriter::put_byte(uint8_t const* raw, uint32_t len)
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
inline void CompactProtocolFieldWriter::field_int_list(int field, std::vector<Enum> const& val)
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
inline void CompactProtocolFieldWriter::field_struct(int field, T const& val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  if constexpr (not std::is_empty_v<T>) {
    writer.write(val);  // write the struct if it's not empty
  } else {
    put_byte(0);  // otherwise, add a stop field
  }
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_empty_struct(int field)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  put_byte(0);  // add a stop field
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct_list(int field, std::vector<T> const& val)
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
                                                          std::vector<uint8_t> const& val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  put_byte(val.data(), static_cast<uint32_t>(val.size()));
  put_byte(0);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_binary(int field, std::vector<uint8_t> const& val)
{
  put_field_header(field, current_field_value, ST_FLD_BINARY);
  put_uint(val.size());
  put_byte(val.data(), static_cast<uint32_t>(val.size()));
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string(int field, std::string const& val)
{
  put_field_header(field, current_field_value, ST_FLD_BINARY);
  put_uint(val.size());
  // FIXME : replace reinterpret_cast
  put_byte(reinterpret_cast<uint8_t const*>(val.data()), static_cast<uint32_t>(val.size()));
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string_list(int field,
                                                          std::vector<std::string> const& val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_BINARY));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    put_uint(v.size());
    // FIXME : replace reinterpret_cast
    put_byte(reinterpret_cast<uint8_t const*>(v.data()), static_cast<uint32_t>(v.size()));
  }
  current_field_value = field;
}

inline int CompactProtocolFieldWriter::current_field() { return current_field_value; }

inline void CompactProtocolFieldWriter::set_current_field(int const& field)
{
  current_field_value = field;
}

}  // namespace cudf::io::parquet::detail
