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

#include <algorithm>
#include <io/parquet/parquet.hpp>

namespace cudf {
namespace io {
namespace parquet {
const uint8_t CompactProtocolReader::g_list2struct[16] = {0,
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
    case ST_FLD_LIST:
    case ST_FLD_SET: {
      int c = getb();
      int n = c >> 4;
      if (n == 0xf) n = get_i32();
      t = g_list2struct[c & 0xf];
      if (depth > 10) return false;
      for (int32_t i = 0; i < n; i++) skip_struct_field(t, depth + 1);
    } break;
    case ST_FLD_STRUCT:
      for (;;) {
        int c = getb();
        int d = c >> 4;
        t     = c & 0xf;
        if (!c) break;
        if (depth > 10) return false;
        skip_struct_field(t, depth + 1);
      }
      break;
    default:
      // printf("unsupported skip for type %d\n", t);
      break;
  }
  return true;
}

#define PARQUET_BEGIN_STRUCT(st)          \
  bool CompactProtocolReader::read(st *s) \
  { /*printf(#st "\n");*/                 \
    int fld = 0;                          \
    for (;;) {                            \
      int c, t, f;                        \
      c = getb();                         \
      if (!c) break;                      \
      f   = c >> 4;                       \
      t   = c & 0xf;                      \
      fld = (f) ? fld + f : get_i16();    \
      switch (fld) {
#define PARQUET_FLD_INT16(id, m)       \
  case id:                             \
    s->m = get_i16();                  \
    if (t != ST_FLD_I16) return false; \
    break;

#define PARQUET_FLD_INT32(id, m)       \
  case id:                             \
    s->m = get_i32();                  \
    if (t != ST_FLD_I32) return false; \
    break;

#define PARQUET_FLD_ENUM(id, m, mt)    \
  case id:                             \
    s->m = (mt)get_i32();              \
    if (t != ST_FLD_I32) return false; \
    break;

#define PARQUET_FLD_INT64(id, m)                        \
  case id:                                              \
    s->m = get_i64();                                   \
    if (t < ST_FLD_I16 || t > ST_FLD_I64) return false; \
    break;

#define PARQUET_FLD_STRING(id, m)            \
  case id:                                   \
    if (t != ST_FLD_BINARY)                  \
      return false;                          \
    else {                                   \
      uint32_t n = get_u32();                \
      if (n < (size_t)(m_end - m_cur)) {     \
        s->m.assign((const char *)m_cur, n); \
        m_cur += n;                          \
      } else                                 \
        return false;                        \
    }                                        \
    break;

#define PARQUET_FLD_STRUCT_LIST(id, m)              \
  case id:                                          \
    if (t != ST_FLD_LIST) return false;             \
    {                                               \
      int n;                                        \
      c = getb();                                   \
      if ((c & 0xf) != ST_FLD_STRUCT) return false; \
      n = c >> 4;                                   \
      if (n == 0xf) n = get_u32();                  \
      s->m.resize(n);                               \
      for (int32_t i = 0; i < n; i++)               \
        if (!read(&s->m[i])) return false;          \
      break;                                        \
    }

#define PARQUET_FLD_ENUM_LIST(id, m, mt)                       \
  case id:                                                     \
    if (t != ST_FLD_LIST) return false;                        \
    {                                                          \
      int n;                                                   \
      c = getb();                                              \
      if ((c & 0xf) != ST_FLD_I32) return false;               \
      n = c >> 4;                                              \
      if (n == 0xf) n = get_u32();                             \
      s->m.resize(n);                                          \
      for (int32_t i = 0; i < n; i++) s->m[i] = (mt)get_i32(); \
      break;                                                   \
    }

#define PARQUET_FLD_STRING_LIST(id, m)              \
  case id:                                          \
    if (t != ST_FLD_LIST) return false;             \
    {                                               \
      int n;                                        \
      c = getb();                                   \
      if ((c & 0xf) != ST_FLD_BINARY) return false; \
      n = c >> 4;                                   \
      if (n == 0xf) n = get_u32();                  \
      s->m.resize(n);                               \
      for (int32_t i = 0; i < n; i++) {             \
        uint32_t l = get_u32();                     \
        if (l < (size_t)(m_end - m_cur)) {          \
          s->m[i].assign((const char *)m_cur, l);   \
          m_cur += l;                               \
        } else                                      \
          return false;                             \
      }                                             \
      break;                                        \
    }

#define PARQUET_FLD_STRUCT(id, m)                         \
  case id:                                                \
    if (t != ST_FLD_STRUCT || !read(&s->m)) return false; \
    break;

#define PARQUET_FLD_STRUCT_BLOB(id, m)                  \
  case id:                                              \
    if (t != ST_FLD_STRUCT) return false;               \
    {                                                   \
      const uint8_t *start = m_cur;                     \
      skip_struct_field(t);                             \
      if (m_cur > start) s->m.assign(start, m_cur - 1); \
      break;                                            \
    }

#define PARQUET_END_STRUCT()                                                        \
  default: /*printf("unknown fld %d of type %d\n", fld, t);*/ skip_struct_field(t); \
    }                                                                               \
    }                                                                               \
    return true;                                                                    \
    }

PARQUET_BEGIN_STRUCT(FileMetaData)
PARQUET_FLD_INT32(1, version)
PARQUET_FLD_STRUCT_LIST(2, schema)
PARQUET_FLD_INT64(3, num_rows)
PARQUET_FLD_STRUCT_LIST(4, row_groups)
PARQUET_FLD_STRUCT_LIST(5, key_value_metadata)
PARQUET_FLD_STRING(6, created_by)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(SchemaElement)
PARQUET_FLD_ENUM(1, type, Type)
PARQUET_FLD_INT32(2, type_length)
PARQUET_FLD_ENUM(3, repetition_type, FieldRepetitionType)
PARQUET_FLD_STRING(4, name)
PARQUET_FLD_INT32(5, num_children)
PARQUET_FLD_ENUM(6, converted_type, ConvertedType)
PARQUET_FLD_INT32(7, decimal_scale)
PARQUET_FLD_INT32(8, decimal_precision)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(RowGroup)
PARQUET_FLD_STRUCT_LIST(1, columns)
PARQUET_FLD_INT64(2, total_byte_size)
PARQUET_FLD_INT64(3, num_rows)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(ColumnChunk)
PARQUET_FLD_STRING(1, file_path)
PARQUET_FLD_INT64(2, file_offset)
PARQUET_FLD_STRUCT(3, meta_data)
PARQUET_FLD_INT64(4, offset_index_offset)
PARQUET_FLD_INT32(5, offset_index_length)
PARQUET_FLD_INT64(6, column_index_offset)
PARQUET_FLD_INT32(7, column_index_length)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(ColumnChunkMetaData)
PARQUET_FLD_ENUM(1, type, Type)
PARQUET_FLD_ENUM_LIST(2, encodings, Encoding)
PARQUET_FLD_STRING_LIST(3, path_in_schema)
PARQUET_FLD_ENUM(4, codec, Compression)
PARQUET_FLD_INT64(5, num_values)
PARQUET_FLD_INT64(6, total_uncompressed_size)
PARQUET_FLD_INT64(7, total_compressed_size)
PARQUET_FLD_INT64(9, data_page_offset)
PARQUET_FLD_INT64(10, index_page_offset)
PARQUET_FLD_INT64(11, dictionary_page_offset)
PARQUET_FLD_STRUCT_BLOB(12, statistics_blob)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(PageHeader)
PARQUET_FLD_ENUM(1, type, PageType)
PARQUET_FLD_INT32(2, uncompressed_page_size)
PARQUET_FLD_INT32(3, compressed_page_size)
PARQUET_FLD_STRUCT(5, data_page_header)
PARQUET_FLD_STRUCT(7, dictionary_page_header)
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(DataPageHeader)
PARQUET_FLD_INT32(1, num_values)
PARQUET_FLD_ENUM(2, encoding, Encoding);
PARQUET_FLD_ENUM(3, definition_level_encoding, Encoding);
PARQUET_FLD_ENUM(4, repetition_level_encoding, Encoding);
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(DictionaryPageHeader)
PARQUET_FLD_INT32(1, num_values)
PARQUET_FLD_ENUM(2, encoding, Encoding);
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(KeyValue)
PARQUET_FLD_STRING(1, key)
PARQUET_FLD_STRING(2, value)
PARQUET_END_STRUCT()

/**
 * @brief Constructs the schema from the file-level metadata
 *
 * @param[in] md File metadata that was previously parsed
 *
 * @return True if schema constructed completely, false otherwise
 */
bool CompactProtocolReader::InitSchema(FileMetaData *md)
{
  if (WalkSchema(md) != md->schema.size()) return false;

  /* Inside FileMetaData, there is a std::vector of RowGroups and each RowGroup contains a
   * a std::vector of ColumnChunks. Each ColumnChunk has a member ColumnMetaData, which contains
   * a std::vector of std::strings representing paths. The purpose of the code below is to set the
   * schema_idx of each column of each row to it corresonding row_group. This is effectively
   * mapping the columns to the schema.
   */
  for (auto &row_group : md->row_groups) {
    int current_schema_index = 0;
    for (auto &column : row_group.columns) {
      int parent = 0;  // root of schema
      for (auto const &path : column.meta_data.path_in_schema) {
        auto const it = [&] {
          // find_if starting at (current_schema_index + 1) and then wrapping
          auto schema = [&](auto const &e) { return e.parent_idx == parent && e.name == path; };
          auto mid    = md->schema.cbegin() + current_schema_index + 1;
          auto it     = std::find_if(mid, md->schema.cend(), schema);
          if (it != md->schema.cend()) return it;
          return std::find_if(md->schema.cbegin(), mid, schema);
        }();
        if (it == md->schema.cend()) return false;
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
  FileMetaData *md, int idx, int parent_idx, int max_def_level, int max_rep_level)
{
  if (idx >= 0 && (size_t)idx < md->schema.size()) {
    SchemaElement *e = &md->schema[idx];
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
        int idx_old = idx;
        idx         = WalkSchema(md, idx, parent_idx, max_def_level, max_rep_level);
        if (idx <= idx_old) break;  // Error
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
