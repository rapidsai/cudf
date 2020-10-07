/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "avro_common.h"

namespace cudf {
namespace io {
namespace avro {
#define AVRO_MAGIC (('O' << 0) | ('b' << 8) | ('j' << 16) | (0x01 << 24))

/**
 * @Brief AVRO schema entry
 */
struct schema_entry {
  explicit schema_entry(type_kind_e kind_, int32_t parent_idx_ = -1, int32_t num_children_ = 0)
    : kind(kind_), parent_idx(parent_idx_), num_children(num_children_)
  {
  }
  int32_t parent_idx   = -1;  // index of parent entry in schema array, negative if no parent
  int32_t num_children = 0;
  type_kind_e kind     = type_not_set;
  std::string name     = "";
  std::vector<std::string> symbols;
};

/**
 * @Brief AVRO output column
 */
struct column_desc {
  int32_t schema_data_idx  = -1;  // schema index of data column
  int32_t schema_null_idx  = -1;  // schema index of corresponding null object
  int32_t parent_union_idx = -1;  // index of this column in parent union (-1 if not a union member)
  std::string name         = "";
};

/**
 * @Brief AVRO file metadata struct
 */
struct file_metadata {
  std::map<std::string, std::string> user_data;
  std::string codec       = "";
  uint64_t sync_marker[2] = {0, 0};
  size_t metadata_size    = 0;
  size_t total_data_size  = 0;
  size_t num_rows         = 0;
  uint32_t skip_rows      = 0;
  uint32_t max_block_size = 0;
  std::vector<schema_entry> schema;
  std::vector<block_desc_s> block_list;
  std::vector<column_desc> columns;
};

/**
 * @Brief Extract AVRO schema from JSON string
 */
class schema_parser {
 protected:
  enum { MAX_SCHEMA_DEPTH = 32 };

 public:
  schema_parser() {}
  bool parse(std::vector<schema_entry> &schema, const std::string &str);

 protected:
  bool more_data() const { return (m_cur < m_end); }
  std::string get_str();

 protected:
  const char *m_base;
  const char *m_cur;
  const char *m_end;
};

/**
 * @Brief AVRO file container parsing class
 */
class container {
 public:
  container() { m_base = m_cur = m_end = nullptr; }
  container(const uint8_t *base, size_t len) { init(base, len); }
  void init(const uint8_t *base, size_t len)
  {
    m_base = m_cur = base;
    m_end          = base + len;
  }
  ptrdiff_t bytecount() const { return m_cur - m_base; }
  unsigned int getb() { return (m_cur < m_end) ? *m_cur++ : 0; }
  uint64_t get_u64()
  {
    uint64_t v = 0;
    for (uint64_t l = 0;; l += 7) {
      uint64_t c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) return v;
    }
  }
  int64_t get_i64()
  {
    uint64_t u = get_u64();
    return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
  }
  std::string get_str()
  {
    const char *s;
    size_t len = get_u64();
    len        = ((len & 1) || (m_cur >= m_end)) ? 0 : std::min(len >> 1, (size_t)(m_end - m_cur));
    s          = reinterpret_cast<const char *>(m_cur);
    m_cur += len;
    return std::string(s, len);
  }

 public:
  bool parse(file_metadata *md, size_t max_num_rows = 0x7fffffff, size_t first_row = 0);

 protected:
  const uint8_t *m_base;
  const uint8_t *m_cur;
  const uint8_t *m_end;
};

}  // namespace avro
}  // namespace io
}  // namespace cudf
