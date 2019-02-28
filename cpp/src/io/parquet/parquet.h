/*
* Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef __IO_PARQUET_H__
#define __IO_PARQUET_H__

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>
#include <algorithm>

#include "parquet_common.h"

namespace parquet {

#define PARQUET_MAGIC   (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24))

struct file_header_s
{
    uint32_t magic;
};

struct file_ender_s
{
    uint32_t footer_len;
    uint32_t magic;
};


struct SchemaElement
{
    Type type = BOOLEAN;
    ConvertedType converted_type = UNKNOWN;
    int32_t type_length = 0;    // Byte length of FIXED_LENGTH_BYTE_ARRAY elements, or maximum bit length for other types
    FieldRepetitionType repetition_type = REQUIRED;
    std::string name = "";
    int32_t num_children = 0;
    // The following fields are filled in later during schema initialization
    int max_definition_level = 0;
    int max_repetition_level = 0;
    int parent_idx = 0;
};

struct ColumnMetaData
{
    Type type = BOOLEAN;
    std::vector<Encoding> encodings;
    std::vector<std::string> path_in_schema;
    Compression codec = UNCOMPRESSED;
    int64_t num_values = 0;
    int64_t total_uncompressed_size = 0;    // total byte size of all uncompressed pages in this column chunk (including the headers)
    int64_t total_compressed_size = 0;      // total byte size of all compressed pages in this column chunk (including the headers)
    int64_t data_page_offset = 0;           // Byte offset from beginning of file to first data page
    int64_t index_page_offset = 0;          // Byte offset from beginning of file to root index page
    int64_t dictionary_page_offset = 0;     // Byte offset from the beginning of file to first (only) dictionary page
};

struct ColumnChunk
{
    std::string file_path = "";
    int64_t file_offset = 0;
    ColumnMetaData meta_data;    // Column metadata for this chunk.
    int64_t offset_index_offset = 0;        // File offset of ColumnChunk's OffsetIndex
    int32_t offset_index_length = 0;        // Size of ColumnChunk's OffsetIndex, in bytes
    int64_t column_index_offset = 0;        // File offset of ColumnChunk's ColumnIndex
    int32_t column_index_length = 0;        // Size of ColumnChunk's ColumnIndex, in bytes
    // Following fields are derived from other fields
    int schema_idx = -1;        // Index in flattened schema (derived from path_in_schema)
};

struct RowGroup
{
    int64_t total_byte_size = 0;
    std::vector<ColumnChunk> columns;
    int64_t num_rows = 0;
};

struct KeyValue
{
    std::string key;
    std::string value;
};

struct FileMetaData
{
    int32_t version = 0;
    std::vector<SchemaElement> schema;
    int64_t num_rows = 0;
    std::vector<RowGroup> row_groups;
    std::vector<KeyValue> key_value_metadata;
    std::string created_by = "";
};

struct DataPageHeader
{
    int32_t num_values = 0; // Number of values, including NULLs, in this data page.
    Encoding encoding = PLAIN; // Encoding used for this data page
    Encoding definition_level_encoding = PLAIN; // Encoding used for definition levels
    Encoding repetition_level_encoding = PLAIN; // Encoding used for repetition levels
};

struct DictionaryPageHeader
{
    int32_t num_values = 0; // Number of values in the dictionary
    Encoding encoding = PLAIN;  // Encoding using this dictionary page
};

struct PageHeader
{
    PageType type = DATA_PAGE; // the type of the page: indicates which of the *_header fields is set
    int32_t uncompressed_page_size = 0; // Uncompressed page size in bytes (not including the header)
    int32_t compressed_page_size = 0;   // Compressed page size in bytes (not including the header)
    DataPageHeader data_page_header;
    DictionaryPageHeader dictionary_page_header;
};


/// \brief Count the number of leading zeros in an unsigned integer.
static inline int CountLeadingZeros32(uint32_t value) {
#if defined(__clang__) || defined(__GNUC__)
    if (value == 0) return 32;
    return static_cast<int>(__builtin_clz(value));
#elif defined(_MSC_VER)
    unsigned long index;
    return (_BitScanReverse(&index, static_cast<unsigned long>(value))) ? 31 - static_cast<int>(index) : 32;
#else
    int bitpos = 0;
    while (value != 0) {
        value >>= 1;
        ++bitpos;
    }
    return 32 - bitpos;
#endif
}


#define DECL_PARQUET_STRUCT(st)      bool read(st *)

class CPReader
{
protected:
    // Struct field types
    enum {
        ST_FLD_TRUE = 1,
        ST_FLD_FALSE = 2,
        ST_FLD_BYTE = 3,
        ST_FLD_I16 = 4,
        ST_FLD_I32 = 5,
        ST_FLD_I64 = 6,
        ST_FLD_DOUBLE = 7,
        ST_FLD_BINARY = 8,
        ST_FLD_LIST = 9,
        ST_FLD_SET = 10,
        ST_FLD_MAP = 11,
        ST_FLD_STRUCT = 12,
    };
    static const uint8_t g_list2struct[16];

public:
    CPReader() { m_base = m_cur = m_end = nullptr; }
    CPReader(const uint8_t *base, size_t len) { init(base, len); }
    void init(const uint8_t *base, size_t len) { m_base = m_cur = base; m_end = base + len; }
    ptrdiff_t bytecount() const { return m_cur - m_base; }
    unsigned int getb() { return (m_cur < m_end) ? *m_cur++ : 0; }
    void skip_bytes(size_t bytecnt) { bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur)); m_cur += bytecnt; }
    uint32_t get_u32() { uint32_t v = 0; for (uint32_t l = 0; ; l += 7) { uint32_t c = getb(); v |= (c & 0x7f) << l; if (c < 0x80) break; } return v; }
    uint64_t get_u64() { uint64_t v = 0; for (uint64_t l = 0; ; l += 7) { uint64_t c = getb(); v |= (c & 0x7f) << l; if (c < 0x80) break; } return v; }
    int32_t get_i16() { return get_i32(); }
    int32_t get_i32() { uint32_t u = get_u32(); return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1)); }
    int64_t get_i64() { uint64_t u = get_u64(); return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1)); }
    int32_t get_listh(uint8_t *el_type) { uint32_t c = getb(); int32_t sz = c >> 4; *el_type = c & 0xf; if (sz == 0xf) sz = get_u32(); return sz; }
    bool skip_struct_field(int t, int depth = 0);

public:
    // Thrift structure parsing
    DECL_PARQUET_STRUCT(FileMetaData);
    DECL_PARQUET_STRUCT(SchemaElement);
    DECL_PARQUET_STRUCT(RowGroup);
    DECL_PARQUET_STRUCT(ColumnChunk);
    DECL_PARQUET_STRUCT(ColumnMetaData);
    DECL_PARQUET_STRUCT(PageHeader);
    DECL_PARQUET_STRUCT(DataPageHeader);
    DECL_PARQUET_STRUCT(DictionaryPageHeader);
    DECL_PARQUET_STRUCT(KeyValue);

public:
    int NumRequiredBits(uint32_t max_level) { return 32 - CountLeadingZeros32(max_level); }
    bool InitSchema(FileMetaData *md);

protected:
    int WalkSchema(std::vector<SchemaElement> &schema, int idx = 0, int parent_idx = 0, int max_def_level = 0, int max_rep_level = 0);

protected:
    const uint8_t *m_base;
    const uint8_t *m_cur;
    const uint8_t *m_end;
};


}; // namespace parquet

#endif // __IO_PARQUET_H__

