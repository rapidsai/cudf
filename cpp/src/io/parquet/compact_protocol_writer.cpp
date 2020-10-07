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

#include <io/parquet/compact_protocol_writer.hpp>

namespace cudf {
namespace io {
namespace parquet {

/**
 * @Brief Parquet CompactProtocolWriter class
 */

size_t CompactProtocolWriter::write(const FileMetaData &f)
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

size_t CompactProtocolWriter::write(const SchemaElement &s)
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
  return c.value();
}

size_t CompactProtocolWriter::write(const RowGroup &r)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, r.columns);
  c.field_int(2, r.total_byte_size);
  c.field_int(3, r.num_rows);
  return c.value();
}

size_t CompactProtocolWriter::write(const KeyValue &k)
{
  CompactProtocolFieldWriter c(*this);
  c.field_string(1, k.key);
  if (k.value.size() != 0) { c.field_string(2, k.value); }
  return c.value();
}

size_t CompactProtocolWriter::write(const ColumnChunk &s)
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

size_t CompactProtocolWriter::write(const ColumnChunkMetaData &s)
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

void CompactProtocolFieldWriter::put_byte(uint8_t v) { writer.m_buf.push_back(v); }

void CompactProtocolFieldWriter::put_byte(const uint8_t *raw, uint32_t len)
{
  for (uint32_t i = 0; i < len; i++) writer.m_buf.push_back(raw[i]);
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
inline void CompactProtocolFieldWriter::field_int_list(int field, const std::vector<Enum> &val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_I32));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) { put_int(static_cast<int32_t>(v)); }
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct(int field, const T &val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  writer.write(val);
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct_list(int field, const std::vector<T> &val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_STRUCT));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) { writer.write(v); }
  current_field_value = field;
}

inline size_t CompactProtocolFieldWriter::value()
{
  put_byte(0);
  return writer.m_buf.size() - struct_start_pos;
}

inline void CompactProtocolFieldWriter::field_struct_blob(int field,
                                                          const std::vector<uint8_t> &val)
{
  put_field_header(field, current_field_value, ST_FLD_STRUCT);
  put_byte(val.data(), (uint32_t)val.size());
  put_byte(0);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string(int field, const std::string &val)
{
  put_field_header(field, current_field_value, ST_FLD_BINARY);
  put_uint(val.size());
  // FIXME : replace reinterpret_cast
  put_byte(reinterpret_cast<const uint8_t *>(val.data()), (uint32_t)val.size());
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string_list(int field,
                                                          const std::vector<std::string> &val)
{
  put_field_header(field, current_field_value, ST_FLD_LIST);
  put_byte((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_BINARY));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) {
    put_uint(v.size());
    // FIXME : replace reinterpret_cast
    put_byte(reinterpret_cast<const uint8_t *>(v.data()), (uint32_t)v.size());
  }
  current_field_value = field;
}

inline int CompactProtocolFieldWriter::current_field() { return current_field_value; }

inline void CompactProtocolFieldWriter::set_current_field(const int &field)
{
  current_field_value = field;
}

}  // namespace parquet
}  // namespace io
}  // namespace cudf
