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
  field_int(1, f.version);
  field_struct_list(2, f.schema);
  field_int(3, f.num_rows);
  field_struct_list(4, f.row_groups);
  if (f.key_value_metadata.size() != 0) { field_struct_list(5, f.key_value_metadata); }
  if (f.created_by.size() != 0) { field_string(6, f.created_by); }
  if (f.column_order_listsize != 0) {
    // Dummy list of struct containing an empty field1 struct
    put_fldh(7, current_field(), ST_FLD_LIST);
    putb((uint8_t)((std::min(f.column_order_listsize, 0xfu) << 4) | ST_FLD_STRUCT));
    if (f.column_order_listsize >= 0xf) put_uint(f.column_order_listsize);
    for (uint32_t i = 0; i < f.column_order_listsize; i++) {
      put_fldh(1, 0, ST_FLD_STRUCT);
      putb(0);  // ColumnOrder.field1 struct end
      putb(0);  // ColumnOrder struct end
    }
    set_current_field(7);
  }
  return value();
}

size_t CompactProtocolWriter::write(const SchemaElement &s)
{
  if (s.type != UNDEFINED_TYPE) {
    field_int(1, s.type);
    if (s.type_length != 0) { field_int(2, s.type_length); }
  }
  if (s.repetition_type != NO_REPETITION_TYPE) { field_int(3, s.repetition_type); }
  field_string(4, s.name);

  if (s.type == UNDEFINED_TYPE) { field_int(5, s.num_children); }
  if (s.converted_type != UNKNOWN) {
    field_int(6, s.converted_type);
    if (s.converted_type == DECIMAL) {
      field_int(7, s.decimal_scale);
      field_int(8, s.decimal_precision);
    }
  }
  return value();
}

size_t CompactProtocolWriter::write(const RowGroup &r)
{
  field_struct_list(1, r.columns);
  field_int(2, r.total_byte_size);
  field_int(3, r.num_rows);
  return value();
}

size_t CompactProtocolWriter::write(const KeyValue &k)
{
  field_string(1, k.key);
  if (k.value.size() != 0) { field_string(2, k.value); }
  return value();
}

size_t CompactProtocolWriter::write(const ColumnChunk &s)
{
  if (s.file_path.size() != 0) { field_string(1, s.file_path); }
  field_int(2, s.file_offset);
  field_struct(3, s.meta_data);
  if (s.offset_index_length != 0) {
    field_int(4, s.offset_index_offset);
    field_int(5, s.offset_index_length);
  }
  if (s.column_index_length != 0) {
    field_int(6, s.column_index_offset);
    field_int(7, s.column_index_length);
  }
  return value();
}

size_t CompactProtocolWriter::write(const ColumnChunkMetaData &s)
{
  field_int(1, s.type);
  field_int_list(2, s.encodings);
  field_string_list(3, s.path_in_schema);
  field_int(4, s.codec);
  field_int(5, s.num_values);
  field_int(6, s.total_uncompressed_size);
  field_int(7, s.total_compressed_size);
  field_int(9, s.data_page_offset);
  if (s.index_page_offset != 0) { field_int(10, s.index_page_offset); }
  if (s.dictionary_page_offset != 0) { field_int(11, s.dictionary_page_offset); }
  if (s.statistics_blob.size() != 0) { field_struct_blob(12, s.statistics_blob); }
  return value();
}

void CompactProtocolWriter::putb(uint8_t v) { m_buf->push_back(v); }

void CompactProtocolWriter::putb(const uint8_t *raw, uint32_t len)
{
  for (uint32_t i = 0; i < len; i++) m_buf->push_back(raw[i]);
}

uint32_t CompactProtocolWriter::put_uint(uint64_t v)
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

uint32_t CompactProtocolWriter::put_int(int64_t v)
{
  int64_t s = (v < 0);
  return put_uint(((v ^ -s) << 1) + s);
}

void CompactProtocolWriter::put_fldh(int f, int cur, int t)
{
  if (f > cur && f <= cur + 15)
    putb(((f - cur) << 4) | t);
  else {
    putb(t);
    put_int(f);
  }
}

inline void CompactProtocolWriter::field_int(int field, int32_t val)
{
  put_fldh(field, current_field_value, ST_FLD_I32);
  put_int(val);
  current_field_value = field;
}

inline void CompactProtocolWriter::field_int(int field, int64_t val)
{
  put_fldh(field, current_field_value, ST_FLD_I64);
  put_int(val);
  current_field_value = field;
}

template <typename Enum>
inline void CompactProtocolWriter::field_int_list(int field, const std::vector<Enum> &val)
{
  put_fldh(field, current_field_value, ST_FLD_LIST);
  putb((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_I32));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) { put_int(static_cast<int32_t>(v)); }
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolWriter::field_struct(int field, const T &val)
{
  put_fldh(field, current_field_value, ST_FLD_STRUCT);
  write(val);
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolWriter::field_struct_list(int field, const std::vector<T> &val)
{
  put_fldh(field, current_field_value, ST_FLD_LIST);
  putb((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_STRUCT));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) { write(v); }
  current_field_value = field;
}

inline size_t CompactProtocolWriter::value()
{
  putb(0);
  return m_buf->size() - struct_start_pos;
}

inline void CompactProtocolWriter::field_struct_blob(int field, const std::vector<uint8_t> &val)
{
  put_fldh(field, current_field_value, ST_FLD_STRUCT);
  putb(val.data(), (uint32_t)val.size());
  putb(0);
  current_field_value = field;
}

inline void CompactProtocolWriter::field_string(int field, const std::string &val)
{
  put_fldh(field, current_field_value, ST_FLD_BINARY);
  put_uint(val.size());
  // FIXME : replace reinterpret_cast
  putb(reinterpret_cast<const uint8_t *>(val.data()), (uint32_t)val.size());
  current_field_value = field;
}

inline void CompactProtocolWriter::field_string_list(int field, const std::vector<std::string> &val)
{
  put_fldh(field, current_field_value, ST_FLD_LIST);
  putb((uint8_t)((std::min(val.size(), (size_t)0xfu) << 4) | ST_FLD_BINARY));
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto &v : val) {
    put_uint(v.size());
    // FIXME : replace reinterpret_cast
    putb(reinterpret_cast<const uint8_t *>(v.data()), (uint32_t)v.size());
  }
  current_field_value = field;
}

inline int CompactProtocolWriter::current_field() { return current_field_value; }

inline void CompactProtocolWriter::set_current_field(const int &field)
{
  current_field_value = field;
}

}  // namespace parquet
}  // namespace io
}  // namespace cudf
