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

#include <io/parquet/parquet.hpp>
#include <io/parquet/parquet_common.hpp>

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace parquet {

/**
 * @brief Class for parsing Parquet's Thrift Compact Protocol encoded metadata
 *
 * This class takes in the Parquet structs and outputs a Thrift-encoded binary blob
 *
 **/
class CompactProtocolWriter {
 public:
  CompactProtocolWriter(std::vector<uint8_t> *output) : m_buf(*output) {}

  size_t write(const FileMetaData &);
  size_t write(const SchemaElement &);
  size_t write(const RowGroup &);
  size_t write(const KeyValue &);
  size_t write(const ColumnChunk &);
  size_t write(const ColumnChunkMetaData &);

 protected:
  std::vector<uint8_t> &m_buf;
  friend class CompactProtocolFieldWriter;
};

class CompactProtocolFieldWriter {
  CompactProtocolWriter &writer;
  size_t struct_start_pos;
  int current_field_value;

 public:
  CompactProtocolFieldWriter(CompactProtocolWriter &caller)
    : writer(caller), struct_start_pos(writer.m_buf.size()), current_field_value(0)
  {
  }

  void put_byte(uint8_t v);

  void put_byte(const uint8_t *raw, uint32_t len);

  uint32_t put_uint(uint64_t v);

  uint32_t put_int(int64_t v);

  void put_field_header(int f, int cur, int t);

  inline void field_int(int field, int32_t val);

  inline void field_int(int field, int64_t val);

  template <typename Enum>
  inline void field_int_list(int field, const std::vector<Enum> &val);

  template <typename T>
  inline void field_struct(int field, const T &val);

  template <typename T>
  inline void field_struct_list(int field, const std::vector<T> &val);

  inline size_t value();

  inline void field_struct_blob(int field, const std::vector<uint8_t> &val);

  inline void field_string(int field, const std::string &val);

  inline void field_string_list(int field, const std::vector<std::string> &val);

  inline int current_field();

  inline void set_current_field(const int &field);
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
