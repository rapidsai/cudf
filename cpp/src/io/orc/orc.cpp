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

#include <string.h>
#include "orc.h"

namespace orc {

void ProtobufReader::skip_struct_field(int t)
{
    switch (t)
    {
    case PB_TYPE_VARINT:
        get_u32();
        break;
    case PB_TYPE_FIXED64:
        skip_bytes(8);
        break;
    case PB_TYPE_FIXEDLEN:
        skip_bytes(get_u32());
        break;
    case PB_TYPE_FIXED32:
        skip_bytes(4);
        break;
    default:
        //printf("invalid type (%d)\n", t);
        break;
    }
}


#define ORC_BEGIN_STRUCT(st)                \
    bool ProtobufReader::read(st *s, size_t maxlen)   \
    {   /*printf(#st "\n");*/                   \
        const uint8_t *end = std::min(m_cur + maxlen, m_end); \
        while (m_cur < end)                     \
        {                                       \
            int fld = get_u32();                \
            switch(fld) {                       \

#define ORC_FLD_UINT64(id, m)                   \
            case (id)*8+PB_TYPE_VARINT: s->m = get_u64(); break; \

#define ORC_FLD_INT64(id, m)                    \
            case (id)*8+PB_TYPE_VARINT: s->m = get_i64(); break; \

#define ORC_FLD_UINT32(id, m)                   \
            case (id)*8+PB_TYPE_VARINT: s->m = get_u32(); break; \

#define ORC_FLD_INT32(id, m)                    \
            case (id)*8+PB_TYPE_VARINT: s->m = get_i32(); break; \

#define ORC_FLD_ENUM(id, m, mt)                 \
            case (id)*8+PB_TYPE_VARINT: s->m = (mt)get_u32(); break; \

#define ORC_FLD_PACKED_UINT32(id, m)            \
            case (id)*8+PB_TYPE_FIXEDLEN: {     \
                uint32_t len = get_u32();       \
                const uint8_t *fld_end = std::min(m_cur+len, end); \
                while (m_cur < fld_end) s->m.push_back(get_u32()); break; \
                } \

#define ORC_FLD_STRING(id, m)                   \
            case (id)*8+PB_TYPE_FIXEDLEN: {     \
                uint32_t n = get_u32(); if (n > (size_t)(end - m_cur)) return false; \
                s->m.assign((const char *)m_cur, n); m_cur += n; \
                break; }                        \

#define ORC_FLD_REPEATED_STRING(id, m)          \
            case (id)*8+PB_TYPE_FIXEDLEN: {     \
                uint32_t n = get_u32(); if (n > (size_t)(end - m_cur)) return false; \
                s->m.resize(s->m.size() + 1);   \
                s->m.back().assign((const char *)m_cur, n); m_cur += n; \
                break; }                        \

#define ORC_FLD_REPEATED_STRUCT(id, m)          \
            case (id)*8+PB_TYPE_FIXEDLEN: {     \
                uint32_t n = get_u32(); if (n > (size_t)(end - m_cur)) return false; \
                s->m.resize(s->m.size() + 1); if (!read(&s->m.back(), n)) return false; \
                break; }                        \

#define ORC_END_STRUCT_(postproccond)           \
            default: /*printf("unknown fld %d of type %d\n", fld >> 3, fld & 7);*/ skip_struct_field(fld & 7); \
            }                                   \
        }                                       \
        return (postproccond);                  \
    }                                           \

#define ORC_END_STRUCT()                        ORC_END_STRUCT_(m_cur <= end)
#define ORC_END_STRUCT_POSTPROC(fn)             ORC_END_STRUCT_(m_cur <= end && fn(s))


ORC_BEGIN_STRUCT(PostScript)
    ORC_FLD_UINT64(1, footerLength)
    ORC_FLD_ENUM(2, compression, CompressionKind)
    ORC_FLD_UINT32(3, compressionBlockSize)
    ORC_FLD_PACKED_UINT32(4, version)
    ORC_FLD_UINT64(5, metadataLength)
    ORC_FLD_STRING(8000, magic)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(FileFooter)
    ORC_FLD_UINT64(1, headerLength)
    ORC_FLD_UINT64(2, contentLength)
    ORC_FLD_REPEATED_STRUCT(3, stripes)
    ORC_FLD_REPEATED_STRUCT(4, types)
    ORC_FLD_REPEATED_STRUCT(5, metadata)
    ORC_FLD_UINT64(6, numberOfRows)
    ORC_FLD_UINT32(8, rowIndexStride)
ORC_END_STRUCT_POSTPROC(InitSchema)

ORC_BEGIN_STRUCT(StripeInformation)
    ORC_FLD_UINT64(1, offset)
    ORC_FLD_UINT64(2, indexLength)
    ORC_FLD_UINT64(3, dataLength)
    ORC_FLD_UINT32(4, footerLength)
    ORC_FLD_UINT32(5, numberOfRows)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(SchemaType)
    ORC_FLD_ENUM(1, kind, TypeKind)
    ORC_FLD_PACKED_UINT32(2, subtypes)
    ORC_FLD_REPEATED_STRING(3, fieldNames)
    ORC_FLD_UINT32(4, maximumLength)
    ORC_FLD_UINT32(5, precision)
    ORC_FLD_UINT32(6, scale)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(UserMetadataItem)
    ORC_FLD_STRING(1, name)
    ORC_FLD_STRING(2, value)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(StripeFooter)
    ORC_FLD_REPEATED_STRUCT(1, streams)
    ORC_FLD_REPEATED_STRUCT(2, columns)
    ORC_FLD_STRING(3, writerTimezone)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(Stream)
    ORC_FLD_ENUM(1, kind, StreamKind)
    ORC_FLD_UINT32(2, column)
    ORC_FLD_UINT64(3, length)
ORC_END_STRUCT()

ORC_BEGIN_STRUCT(ColumnEncoding)
    ORC_FLD_ENUM(1, kind, ColumnEncodingKind)
    ORC_FLD_UINT32(2, dictionarySize)
ORC_END_STRUCT()


// return the column name
std::string FileFooter::GetColumnName(uint32_t column_id)
{
    std::string s = "";
    uint32_t parent_idx = column_id, idx, field_idx;
    do
    {
        idx = parent_idx;
        parent_idx = (idx < types.size()) ? (uint32_t)types[idx].parent_idx : ~0;
        field_idx = (parent_idx < types.size()) ? (uint32_t)types[idx].field_idx : ~0;
        if (parent_idx >= types.size())
            break;
        if (field_idx < types[parent_idx].fieldNames.size())
        {
            if (s.length() > 0)
                s = types[parent_idx].fieldNames[field_idx] + "." + s;
            else
                s = types[parent_idx].fieldNames[field_idx];
        }
    } while (parent_idx != idx);
    // If we have no name (root column), generate a name
    if (s.length() == 0)
    {
        s = "col" + std::to_string(column_id);
    }
    return s;
}

// Initializes the parent_idx field in the schema
bool ProtobufReader::InitSchema(FileFooter *ff)
{
    int32_t schema_size = (int32_t)ff->types.size();
    for (int32_t i = 0; i < schema_size; i++)
    {
        int32_t num_children = (int32_t)ff->types[i].subtypes.size();
        if (ff->types[i].parent_idx == -1) // Not initialized
        {
            ff->types[i].parent_idx = i; // set root node as its own parent
        }
        for (int32_t j = 0; j < num_children; j++)
        {
            uint32_t column_id = ff->types[i].subtypes[j];
            if (column_id <= (uint32_t)i || column_id >= (uint32_t)schema_size)
            {
                // Invalid column id (or at least not a schema index)
                return false;
            }
            if (ff->types[column_id].parent_idx != -1)
            {
                // Same node referenced twice
                return false;
            }
            ff->types[column_id].parent_idx = i;
            ff->types[column_id].field_idx = j;
        }
    }
    return true;
}


/* ----------------------------------------------------------------------------*/
/**
* @Brief ORC decompression class
*
*/
/* ----------------------------------------------------------------------------*/

OrcDecompressor::OrcDecompressor(CompressionKind kind, uint32_t blockSize):
    m_kind(kind), m_log2MaxRatio(24), m_blockSize(blockSize), m_decompressor(nullptr)
{
    if (kind != NONE)
    {
        int stream_type;
        switch(kind)
        {
        case ZLIB:
            stream_type = IO_UNCOMP_STREAM_TYPE_INFLATE;
            m_log2MaxRatio = 11; // < 2048:1
            break;
        case SNAPPY:
            stream_type = IO_UNCOMP_STREAM_TYPE_SNAPPY;
            m_log2MaxRatio = 5; // < 32:1
            break;
        case LZO:
            stream_type = IO_UNCOMP_STREAM_TYPE_LZO;
            break;
        case LZ4:
            stream_type = IO_UNCOMP_STREAM_TYPE_LZ4;
            break;
        case ZSTD:
            stream_type = IO_UNCOMP_STREAM_TYPE_ZSTD;
            break;
        default:
            stream_type = IO_UNCOMP_STREAM_TYPE_INFER; // Will be treated as invalid
        }
        m_decompressor = (stream_type != IO_UNCOMP_STREAM_TYPE_INFER) ? HostDecompressor::Create(stream_type) : nullptr;
    }
    else
    {
        m_log2MaxRatio = 0;
    }
}

OrcDecompressor::~OrcDecompressor()
{
    if (m_decompressor)
    {
        delete m_decompressor;
    }
}


/* --------------------------------------------------------------------------*/
/**
* @Brief ORC block decompression
*
* @param srcBytes[in] compressed data
* @param srcLen[in] length of compressed data
* @param dstLen[out] length of uncompressed data
*
* @returns pointer to uncompressed data, nullptr if error
*/
/* ----------------------------------------------------------------------------*/

const uint8_t *OrcDecompressor::Decompress(const uint8_t *srcBytes, size_t srcLen, size_t *dstLen)
{
    // If uncompressed, just pass-through the input
    if (m_kind == NONE)
    {
        *dstLen = srcLen;
        return srcBytes;
    }
    *dstLen = 0;
    if (m_decompressor)
    {
        // First, scan the input for the number of blocks and worst-case output size
        size_t max_dst_length = 0, dst_length;
        uint32_t num_blocks = 0;
        uint8_t *dst;
        for (size_t i = 0; i + 3 < srcLen; )
        {
            uint32_t block_len = srcBytes[i] | (srcBytes[i + 1] << 8) | (srcBytes[i + 2] << 16);
            uint32_t is_uncompressed = block_len & 1;
            i += 3;
            block_len >>= 1;
            if (is_uncompressed)
            {
                // Uncompressed block
                max_dst_length += block_len;
            }
            else
            {
                max_dst_length += m_blockSize;
            }
            i += block_len;
            if (i > srcLen || block_len > m_blockSize)
            {
                return nullptr;
            }
        }
        // Check if we have a single uncompressed block, or no blocks
        if (max_dst_length < m_blockSize)
        {
            if (srcLen < 3)
            {
                // Total size is less than the 3-byte header
                return nullptr;
            }
            *dstLen = srcLen - 3;
            return srcBytes + 3;
        }
        m_buf.resize(max_dst_length);
        dst = m_buf.data();
        dst_length = 0;
        for (size_t i = 0; i + 3 < srcLen; )
        {
            uint32_t block_len = srcBytes[i] | (srcBytes[i + 1] << 8) | (srcBytes[i + 2] << 16);
            uint32_t is_uncompressed = block_len & 1;
            i += 3;
            block_len >>= 1;
            if (is_uncompressed)
            {
                // Uncompressed block
                memcpy(dst + dst_length, srcBytes + i, block_len);
                dst_length += block_len;
            }
            else
            {
                // Compressed block
                dst_length += m_decompressor->Decompress(dst + dst_length, m_blockSize, srcBytes + i, block_len);
            }
            i += block_len;
        }
        *dstLen = dst_length;
        return m_buf.data();
    }
    return nullptr;
}


}; // namespace orc
