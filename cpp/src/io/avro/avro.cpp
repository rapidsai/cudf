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

#include "avro.h"
#include <string.h>
#include <unordered_map>

namespace cudf {
namespace io {
namespace avro {
/**
 * @Brief AVRO file metadata parser
 *
 * @param md[out] parsed avro file metadata
 * @param max_num_rows[in] maximum number of rows
 * @param first_row[in] drop blocks below first_row
 *
 * @returns true if successful, false if error
 */
bool container::parse(file_metadata *md, size_t max_num_rows, size_t first_row)
{
  uint32_t sig4, max_block_size;
  size_t total_object_count;

  sig4 = getb();
  sig4 |= getb() << 8;
  sig4 |= getb() << 16;
  sig4 |= getb() << 24;
  if (sig4 != AVRO_MAGIC) { return false; }
  for (;;) {
    uint32_t num_md_items = static_cast<uint32_t>(get_i64());
    if (num_md_items == 0) { break; }
    for (uint32_t i = 0; i < num_md_items; i++) {
      std::string key   = get_str();
      std::string value = get_str();
      if (key == "avro.codec") {
        md->codec = value;
      } else if (key == "avro.schema") {
        schema_parser sp;
        if (!sp.parse(md->schema, value)) { return false; }
      } else {
        // printf("\"%s\" = \"%s\"\n", key.c_str(), value.c_str());
        md->user_data.emplace(key, value);
      }
    }
  }
  for (int i = 0; i < 16; i++) { (reinterpret_cast<uint8_t *>(&md->sync_marker[0]))[i] = getb(); }
  md->metadata_size  = m_cur - m_base;
  md->skip_rows      = 0;
  max_block_size     = 0;
  total_object_count = 0;
  while (m_cur + 18 < m_end && total_object_count < max_num_rows) {
    uint32_t object_count = static_cast<uint32_t>(get_i64());
    uint32_t block_size   = static_cast<uint32_t>(get_i64());
    if (block_size <= 0 || object_count <= 0 || m_cur + block_size + 16 > m_end) { break; }
    if (object_count > first_row) {
      uint32_t block_row = static_cast<uint32_t>(total_object_count);
      max_block_size     = std::max(max_block_size, block_size);
      total_object_count += object_count;
      if (!md->block_list.size()) {
        md->skip_rows = static_cast<uint32_t>(first_row);
        total_object_count -= first_row;
        first_row = 0;
      }
      md->block_list.emplace_back(m_cur - m_base, block_size, block_row, object_count);
    } else {
      first_row -= object_count;
    }
    m_cur += block_size;
    m_cur += 16;  // TODO: Validate sync marker
  }
  md->max_block_size  = max_block_size;
  md->num_rows        = total_object_count;
  md->total_data_size = m_cur - (m_base + md->metadata_size);
  // Extract columns
  for (size_t i = 0; i < md->schema.size(); i++) {
    type_kind_e kind = md->schema[i].kind;
    if (kind > type_null && kind < type_record) {
      // Primitive type column
      column_desc col;
      int parent_idx       = md->schema[i].parent_idx;
      col.schema_data_idx  = (int32_t)i;
      col.schema_null_idx  = -1;
      col.parent_union_idx = -1;
      col.name             = md->schema[i].name;
      if (parent_idx >= 0) {
        while (parent_idx >= 0) {
          if (md->schema[parent_idx].kind == type_union) {
            int pos = parent_idx + 1;
            for (int num_children = md->schema[parent_idx].num_children; num_children > 0;
                 --num_children) {
              int skip = 1;
              if (pos == i) {
                col.parent_union_idx = md->schema[parent_idx].num_children - num_children;
              } else if (md->schema[pos].kind == type_null) {
                col.schema_null_idx = pos;
                break;
              }
              do {
                skip = skip + md->schema[pos].num_children - 1;
                pos++;
              } while (skip != 0);
            }
          }
          // Ignore the root or array entries
          if ((parent_idx != 0 && md->schema[parent_idx].kind != type_array) ||
              col.name.length() == 0) {
            if (col.name.length() > 0) { col.name.insert(0, 1, '.'); }
            col.name.insert(0, md->schema[parent_idx].name);
          }
          parent_idx = md->schema[parent_idx].parent_idx;
        }
      }
      md->columns.emplace_back(std::move(col));
    }
  }
  return true;
}

/**
 * @brief Parser state
 **/
enum json_state_e {
  state_attrname = 0,
  state_attrcolon,
  state_attrvalue,
  state_attrvalue_last,
  state_nextattr,
  state_nextsymbol,
};

enum {
  attrtype_none = -1,
  attrtype_type = 0,
  attrtype_name,
  attrtype_fields,
  attrtype_symbols,
  attrtype_items,
};

/**
 * @Brief AVRO JSON schema parser
 *
 * @param schema[out] parsed avro schema
 * @param str[in] avro schema (JSON string)
 *
 * @returns true if successful, false if error
 */
bool schema_parser::parse(std::vector<schema_entry> &schema, const std::string &json_str)
{
  char depthbuf[MAX_SCHEMA_DEPTH];
  int depth = 0, parent_idx = -1, entry_idx = -1;
  json_state_e state = state_attrname;
  std::string str;
  const std::unordered_map<std::string, type_kind_e> typenames = {{"null", type_null},
                                                                  {"boolean", type_boolean},
                                                                  {"int", type_int},
                                                                  {"long", type_long},
                                                                  {"float", type_float},
                                                                  {"double", type_double},
                                                                  {"bytes", type_bytes},
                                                                  {"string", type_string},
                                                                  {"record", type_record},
                                                                  {"enum", type_enum},
                                                                  {"array", type_array}};
  const std::unordered_map<std::string, int> attrnames         = {{"type", attrtype_type},
                                                          {"name", attrtype_name},
                                                          {"fields", attrtype_fields},
                                                          {"symbols", attrtype_symbols},
                                                          {"items", attrtype_items}};
  int cur_attr                                                 = attrtype_none;
  m_base                                                       = json_str.c_str();
  m_cur                                                        = m_base;
  m_end                                                        = m_base + json_str.length();
  while (more_data()) {
    int c = *m_cur++;
    switch (c) {
      case '"':
        str = get_str();
        // printf("str: \"%s\" (cur_attr=%d, state=%d)\n", str.c_str(), cur_attr, state);
        if (state == state_attrname && cur_attr == attrtype_none &&
            typenames.find(str) != typenames.end()) {
          cur_attr = attrtype_type;
          state    = state_attrvalue_last;
        }
        if (state == state_attrname) {
          auto t   = attrnames.find(str);
          cur_attr = (t == attrnames.end()) ? attrtype_none : t->second;
          state    = state_attrcolon;
        } else if (state == state_attrvalue || state == state_attrvalue_last) {
          if (entry_idx < 0) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_not_set, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          }
          if (cur_attr == attrtype_type) {
            auto t = typenames.find(str);
            if (t == typenames.end()) return false;
            schema[entry_idx].kind = t->second;
          } else if (cur_attr == attrtype_name) {
            if (entry_idx < 0) return false;
            schema[entry_idx].name = std::move(str);
          }
          if (state == state_attrvalue_last) { entry_idx = -1; }
          state    = state_nextattr;
          cur_attr = attrtype_none;
        } else if (state == state_nextsymbol) {
          if (entry_idx < 0) return false;
          schema[entry_idx].symbols.emplace_back(std::move(str));
        }
        break;
      case ':':
        if (state != state_attrcolon) return false;
        state = state_attrvalue;
        break;
      case ',':
        if (state != state_nextsymbol) {
          if (state != state_nextattr) return false;
          state = state_attrname;
        }
        break;
      case '{':
        if (state == state_attrvalue && cur_attr == attrtype_type) {
          if (entry_idx < 0) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_record, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          }
          cur_attr = attrtype_none;
          state    = state_attrname;
        } else if (state == state_attrvalue && cur_attr == attrtype_items && entry_idx >= 0) {
          // Treat array as a one-field record
          parent_idx = entry_idx;
          entry_idx  = -1;
          cur_attr   = attrtype_none;
          state      = state_attrname;
        }
        if (depth >= MAX_SCHEMA_DEPTH || state != state_attrname) { return false; }
        depthbuf[depth++] = '{';
        break;
      case '}':
        if (depth == 0 || state != state_nextattr || depthbuf[depth - 1] != '{') return false;
        --depth;
        if (entry_idx < 0) {
          parent_idx = (parent_idx >= 0) ? schema[parent_idx].parent_idx : -1;
        } else {
          entry_idx = -1;
        }
        break;
      case '[':
        if (state == state_attrname && cur_attr == attrtype_none) {
          cur_attr = attrtype_type;
          state    = state_attrvalue;
        }
        if (depth >= MAX_SCHEMA_DEPTH || state != state_attrvalue) { return false; }
        depthbuf[depth++] = '[';
        if (cur_attr == attrtype_symbols) {
          state = state_nextsymbol;
          break;
        } else if (cur_attr == attrtype_type) {
          if (entry_idx < 0 || schema[entry_idx].kind != type_not_set) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_union, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          } else {
            schema[entry_idx].kind = type_union;
          }
          parent_idx = entry_idx;
        } else if (cur_attr != attrtype_fields || entry_idx < 0 ||
                   schema[entry_idx].kind < type_record) {
          return false;
        } else {
          parent_idx = entry_idx;
        }
        entry_idx = -1;
        cur_attr  = attrtype_none;
        state     = state_attrname;
        break;
      case ']':
        if (depth == 0 || (state != state_nextattr && state != state_nextsymbol) ||
            depthbuf[depth - 1] != '[')
          return false;
        --depth;
        if (state == state_nextsymbol) {
          state = state_nextattr;
        } else if (parent_idx >= 0) {
          entry_idx  = parent_idx;
          parent_idx = schema[parent_idx].parent_idx;
        }
        break;
      case ' ':
      case '\x09':
      case '\x0d':
      case '\x0a':
        // Ignore spaces, tabs and CRLF
        break;
      default: return false;
    }
  }
  // printf("schema (%d entries) = %s\n", (int)schema.size(), m_base);
  return true;
}

/**
 * @Brief Parse a string
 *
 * @returns parsed string, consuming the terminating quote
 */
std::string schema_parser::get_str()
{
  std::string s;
  const char *start = m_cur;
  const char *cur   = start;
  while (cur < m_end && *cur++ != '"')
    ;
  int32_t len = static_cast<int32_t>(cur - start - 1);
  m_cur       = cur;
  return s.assign(start, std::max(len, 0));
}

}  // namespace avro
}  // namespace io
}  // namespace cudf
