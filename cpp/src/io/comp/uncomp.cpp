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

#include <cuda_runtime.h>
#include <string.h>  // memset
#include <zlib.h>    // uncompress
#include "io_uncomp.h"
#include "unbz2.h"  // bz2 uncompress

#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {
#define GZ_FLG_FTEXT 0x01     // ASCII text hint
#define GZ_FLG_FHCRC 0x02     // Header CRC present
#define GZ_FLG_FEXTRA 0x04    // Extra fields present
#define GZ_FLG_FNAME 0x08     // Original file name present
#define GZ_FLG_FCOMMENT 0x10  // Comment present

#pragma pack(push, 1)

struct gz_file_header_s {
  uint8_t id1;        // 0x1f
  uint8_t id2;        // 0x8b
  uint8_t comp_mthd;  // compression method (0-7=reserved, 8=deflate)
  uint8_t flags;      // flags (GZ_FLG_XXX)
  uint8_t mtime[4];   // If non-zero: modification time (Unix format)
  uint8_t xflags;     // Extra compressor-specific flags
  uint8_t os;         // OS id
};

struct zip_eocd_s  // end of central directory
{
  uint32_t sig;            // 0x06054b50
  uint16_t disk_id;        // number of this disk
  uint16_t start_disk;     // number of the disk with the start of the central directory
  uint16_t num_entries;    // number of entries in the central dir on this disk
  uint16_t total_entries;  // total number of entries in the central dir
  uint32_t cdir_size;      // size of the central directory
  uint32_t cdir_offset;    // offset of start of central directory with respect to the starting disk
                         // number uint16_t comment_len;   // comment length (excluded from struct)
};

struct zip64_eocdl  // end of central dir locator
{
  uint32_t sig;         // 0x07064b50
  uint32_t disk_start;  // number of the disk with the start of the zip64 end of central directory
  uint64_t eocdr_ofs;   // relative offset of the zip64 end of central directory record
  uint32_t num_disks;   // total number of disks
};

struct zip_cdfh_s  // central directory file header
{
  uint32_t sig;          // 0x02014b50
  uint16_t ver;          // version made by
  uint16_t min_ver;      // version needed to extract
  uint16_t gp_flags;     // general purpose bit flag
  uint16_t comp_method;  // compression method
  uint16_t file_time;    // last mod file time
  uint16_t file_date;    // last mod file date
  uint32_t crc32;        // crc - 32
  uint32_t comp_size;    // compressed size
  uint32_t uncomp_size;  // uncompressed size
  uint16_t fname_len;    // filename length
  uint16_t extra_len;    // extra field length
  uint16_t comment_len;  // file comment length
  uint16_t start_disk;   // disk number start
  uint16_t int_fattr;    // internal file attributes
  uint32_t ext_fattr;    // external file attributes
  uint32_t hdr_ofs;      // relative offset of local header
};

struct zip_lfh_s {
  uint32_t sig;          // 0x04034b50
  uint16_t ver;          // version needed to extract
  uint16_t gp_flags;     // general purpose bit flag
  uint16_t comp_method;  // compression method
  uint16_t file_time;    // last mod file time
  uint16_t file_date;    // last mod file date
  uint32_t crc32;        // crc - 32
  uint32_t comp_size;    // compressed size
  uint32_t uncomp_size;  // uncompressed size
  uint16_t fname_len;    // filename length
  uint16_t extra_len;    // extra field length
};

struct bz2_file_header_s {
  uint8_t sig[3];  // "BZh"
  uint8_t blksz;   // block size 1..9 in 100kB units (post-RLE)
};

#pragma pack(pop)

struct gz_archive_s {
  const gz_file_header_s *fhdr;
  uint16_t hcrc16;  // header crc16 if present
  uint16_t xlen;
  const uint8_t *fxtra;      // xlen bytes (optional)
  const uint8_t *fname;      // zero-terminated original filename if present
  const uint8_t *fcomment;   // zero-terminated comment if present
  const uint8_t *comp_data;  // compressed data
  size_t comp_len;           // Compressed data length
  uint32_t crc32;            // CRC32 of uncompressed data
  uint32_t isize;            // Input size modulo 2^32
};

struct zip_archive_s {
  const zip_eocd_s *eocd;    // end of central directory
  const zip64_eocdl *eocdl;  // end of central dir locator (optional)
  const zip_cdfh_s *cdfh;    // start of central directory file headers
};

bool ParseGZArchive(gz_archive_s *dst, const uint8_t *raw, size_t len)
{
  const gz_file_header_s *fhdr;

  if (!dst) return false;
  memset(dst, 0, sizeof(gz_archive_s));
  if (len < sizeof(gz_file_header_s) + 8) return false;
  fhdr = (gz_file_header_s *)raw;
  if (fhdr->id1 != 0x1f || fhdr->id2 != 0x8b) return false;
  dst->fhdr = fhdr;
  raw += sizeof(gz_file_header_s);
  len -= sizeof(gz_file_header_s);
  if (fhdr->flags & GZ_FLG_FEXTRA) {
    uint32_t xlen;

    if (len < 2) return false;
    xlen = raw[0] | (raw[1] << 8);
    raw += 2;
    len -= 2;
    if (len < xlen) return false;
    dst->xlen  = (uint16_t)xlen;
    dst->fxtra = raw;
    raw += xlen;
    len -= xlen;
  }
  if (fhdr->flags & GZ_FLG_FNAME) {
    size_t l = 0;
    uint8_t c;
    do {
      if (l >= len) return false;
      c = raw[l];
      l++;
    } while (c != 0);
    dst->fname = raw;
    raw += l;
    len -= l;
  }
  if (fhdr->flags & GZ_FLG_FCOMMENT) {
    size_t l = 0;
    uint8_t c;
    do {
      if (l >= len) return false;
      c = raw[l];
      l++;
    } while (c != 0);
    dst->fcomment = raw;
    raw += l;
    len -= l;
  }
  if (fhdr->flags & GZ_FLG_FHCRC) {
    if (len < 2) return false;
    dst->hcrc16 = raw[0] | (raw[1] << 8);
    raw += 2;
    len -= 2;
  }
  if (len < 8) return false;
  dst->crc32 = raw[len - 8] | (raw[len - 7] << 8) | (raw[len - 6] << 16) | (raw[len - 5] << 24);
  dst->isize = raw[len - 4] | (raw[len - 3] << 8) | (raw[len - 2] << 16) | (raw[len - 1] << 24);
  len -= 8;
  dst->comp_data = raw;
  dst->comp_len  = len;
  return (fhdr->comp_mthd == 8 && len > 0);
}

bool OpenZipArchive(zip_archive_s *dst, const uint8_t *raw, size_t len)
{
  memset(dst, 0, sizeof(zip_archive_s));
  // Find the end of central directory
  if (len >= sizeof(zip_eocd_s) + 2) {
    for (size_t i = len - sizeof(zip_eocd_s) - 2; i + sizeof(zip_eocd_s) + 2 + 0xffff >= len; i--) {
      const zip_eocd_s *eocd = (zip_eocd_s *)(raw + i);
      if (eocd->sig == 0x06054b50 &&
          eocd->disk_id == eocd->start_disk  // multi-file archives not supported
          && eocd->num_entries == eocd->total_entries &&
          eocd->cdir_size >= sizeof(zip_cdfh_s) * eocd->num_entries && eocd->cdir_offset < len &&
          i + *(const uint16_t *)(eocd + 1) <= len) {
        const zip_cdfh_s *cdfh = (const zip_cdfh_s *)(raw + eocd->cdir_offset);
        dst->eocd              = eocd;
        if (i >= sizeof(zip64_eocdl)) {
          const zip64_eocdl *eocdl = (const zip64_eocdl *)(raw + i - sizeof(zip64_eocdl));
          if (eocdl->sig == 0x07064b50) { dst->eocdl = eocdl; }
        }
        // Start of central directory
        if (cdfh->sig == 0x02014b50) { dst->cdfh = cdfh; }
      }
    }
  }
  return (dst->eocd && dst->cdfh);
}

int cpu_inflate(uint8_t *uncomp_data, size_t *destLen, const uint8_t *comp_data, size_t comp_len)
{
  int zerr;
  z_stream strm;

  memset(&strm, 0, sizeof(strm));
  strm.next_in   = (Bytef *)comp_data;
  strm.avail_in  = comp_len;
  strm.total_in  = 0;
  strm.next_out  = uncomp_data;
  strm.avail_out = *destLen;
  strm.total_out = 0;
  zerr           = inflateInit2(&strm, -15);  // -15 for raw data without GZIP headers
  if (zerr != 0) {
    *destLen = 0;
    return zerr;
  }
  zerr     = inflate(&strm, Z_FINISH);
  *destLen = strm.total_out;
  inflateEnd(&strm);
  return (zerr == Z_STREAM_END) ? Z_OK : zerr;
}

/**
 * @Brief Uncompresses a raw DEFLATE stream to a char vector.
 * The vector will be grown to match the uncompressed size
 * Optimized for the case where the initial size is the uncompressed
 * size truncated to 32-bit, and grows the buffer in 1GB increments.
 *
 * @param dst[out] Destination vector
 * @param comp_data[in] Raw compressed data
 * @param comp_len[in] Compressed data size
 */
int cpu_inflate_vector(std::vector<char> &dst, const uint8_t *comp_data, size_t comp_len)
{
  int zerr;
  z_stream strm;

  memset(&strm, 0, sizeof(strm));
  strm.next_in   = (Bytef *)comp_data;
  strm.avail_in  = comp_len;
  strm.total_in  = 0;
  strm.next_out  = reinterpret_cast<uint8_t *>(dst.data());
  strm.avail_out = dst.size();
  strm.total_out = 0;
  zerr           = inflateInit2(&strm, -15);  // -15 for raw data without GZIP headers
  if (zerr != 0) {
    dst.resize(0);
    return zerr;
  }
  do {
    if (strm.avail_out == 0) {
      dst.resize(strm.total_out + (1 << 30));
      strm.avail_out = dst.size() - strm.total_out;
      strm.next_out  = reinterpret_cast<uint8_t *>(dst.data()) + strm.total_out;
    }
    zerr = inflate(&strm, Z_SYNC_FLUSH);
  } while ((zerr == Z_BUF_ERROR || zerr == Z_OK) && strm.avail_out == 0 &&
           strm.total_out == dst.size());
  dst.resize(strm.total_out);
  inflateEnd(&strm);
  return (zerr == Z_STREAM_END) ? Z_OK : zerr;
}

/**
 * @Brief Uncompresses a gzip/zip/bzip2/xz file stored in system memory.
 *
 * The result is allocated and stored in a vector.
 * If the function call fails, the output vector is empty.
 *
 * @param src[in] Pointer to the compressed data in system memory
 * @param src_size[in] The size of the compressed data, in bytes
 * @param stream_type[in] Type of compression of the input data
 *
 * @return Vector containing the uncompressed output
 */
std::vector<char> io_uncompress_single_h2d(const void *src, size_t src_size, int stream_type)
{
  const uint8_t *raw       = (const uint8_t *)src;
  const uint8_t *comp_data = nullptr;
  size_t comp_len          = 0;
  size_t uncomp_len        = 0;

  CUDF_EXPECTS(src != nullptr, "Decompression: Source cannot be nullptr");
  CUDF_EXPECTS(src_size != 0, "Decompression: Source size cannot be 0");

  switch (stream_type) {
    case IO_UNCOMP_STREAM_TYPE_INFER:
    case IO_UNCOMP_STREAM_TYPE_GZIP: {
      gz_archive_s gz;
      if (ParseGZArchive(&gz, raw, src_size)) {
        stream_type = IO_UNCOMP_STREAM_TYPE_GZIP;
        comp_data   = gz.comp_data;
        comp_len    = gz.comp_len;
        uncomp_len  = gz.isize;
      }
      if (stream_type != IO_UNCOMP_STREAM_TYPE_INFER) break;  // Fall through for INFER
    }
    case IO_UNCOMP_STREAM_TYPE_ZIP: {
      zip_archive_s za;
      if (OpenZipArchive(&za, raw, src_size)) {
        size_t cdfh_ofs = 0;
        for (int i = 0; i < za.eocd->num_entries; i++) {
          const zip_cdfh_s *cdfh = (const zip_cdfh_s *)(((const uint8_t *)za.cdfh) + cdfh_ofs);
          int cdfh_len = sizeof(zip_cdfh_s) + cdfh->fname_len + cdfh->extra_len + cdfh->comment_len;
          if (cdfh_ofs + cdfh_len > za.eocd->cdir_size || cdfh->sig != 0x02014b50) {
            // Bad cdir
            break;
          }
          // For now, only accept with non-zero file sizes and DEFLATE
          if (cdfh->comp_method == 8 && cdfh->comp_size > 0 && cdfh->uncomp_size > 0) {
            size_t lfh_ofs       = cdfh->hdr_ofs;
            const zip_lfh_s *lfh = (const zip_lfh_s *)(raw + lfh_ofs);
            if (lfh_ofs + sizeof(zip_lfh_s) <= src_size && lfh->sig == 0x04034b50 &&
                lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len <= src_size) {
              if (lfh->comp_method == 8 && lfh->comp_size > 0 && lfh->uncomp_size > 0) {
                size_t file_start = lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len;
                size_t file_end   = file_start + lfh->comp_size;
                if (file_end <= src_size) {
                  // Pick the first valid file of non-zero size (only 1 file expected in archive)
                  stream_type = IO_UNCOMP_STREAM_TYPE_ZIP;
                  comp_data   = raw + file_start;
                  comp_len    = lfh->comp_size;
                  uncomp_len  = lfh->uncomp_size;
                  break;
                }
              }
            }
          }
          cdfh_ofs += cdfh_len;
        }
      }
    }
      if (stream_type != IO_UNCOMP_STREAM_TYPE_INFER) break;  // Fall through for INFER
    case IO_UNCOMP_STREAM_TYPE_BZIP2:
      if (src_size > 4) {
        const bz2_file_header_s *fhdr = (const bz2_file_header_s *)raw;
        // Check for BZIP2 file signature "BZh1" to "BZh9"
        if (fhdr->sig[0] == 'B' && fhdr->sig[1] == 'Z' && fhdr->sig[2] == 'h' &&
            fhdr->blksz >= '1' && fhdr->blksz <= '9') {
          stream_type = IO_UNCOMP_STREAM_TYPE_BZIP2;
          comp_data   = raw;
          comp_len    = src_size;
          uncomp_len  = 0;
        }
      }
      if (stream_type != IO_UNCOMP_STREAM_TYPE_INFER) break;  // Fall through for INFER
    default:
      // Unsupported format
      break;
  }
  CUDF_EXPECTS(comp_data != nullptr, "Unsupported compressed stream type");
  CUDF_EXPECTS(comp_len > 0, "Unsupported compressed stream type");

  if (uncomp_len <= 0) {
    uncomp_len = comp_len * 4 + 4096;  // In case uncompressed size isn't known in advance, assume
                                       // ~4:1 compression for initial size
  }

  if (stream_type == IO_UNCOMP_STREAM_TYPE_GZIP || stream_type == IO_UNCOMP_STREAM_TYPE_ZIP) {
    // INFLATE
    std::vector<char> dst(uncomp_len);
    CUDF_EXPECTS(cpu_inflate_vector(dst, comp_data, comp_len) == 0,
                 "Decompression: error in stream");
    return dst;
  }
  if (stream_type == IO_UNCOMP_STREAM_TYPE_BZIP2) {
    size_t src_ofs = 0;
    size_t dst_ofs = 0;
    int bz_err     = 0;
    std::vector<char> dst(uncomp_len);
    do {
      size_t dst_len = uncomp_len - dst_ofs;
      bz_err         = cpu_bz2_uncompress(
        comp_data, comp_len, ((uint8_t *)dst.data()) + dst_ofs, &dst_len, &src_ofs);
      if (bz_err == BZ_OUTBUFF_FULL) {
        // TBD: We could infer the compression ratio based on produced/consumed byte counts
        // in order to minimize realloc events and over-allocation
        dst_ofs = dst_len;
        dst_len = uncomp_len + (uncomp_len / 2);
        dst.resize(dst_len);
        uncomp_len = dst_len;
      } else if (bz_err == 0) {
        uncomp_len = dst_len;
        dst.resize(uncomp_len);
      }
    } while (bz_err == BZ_OUTBUFF_FULL);
    CUDF_EXPECTS(bz_err == 0, "Decompression: error in stream");
    return dst;
  }

  CUDF_FAIL("Unsupported compressed stream type");
}

/**
 * @brief Uncompresses the input data and stores the allocated result into
 * a vector.
 *
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] num_bytes Size of the input data, in bytes
 * @param[in] compression String describing the compression type
 *
 * @return Vector containing the output uncompressed data
 */
std::vector<char> getUncompressedHostData(const char *h_data,
                                          size_t num_bytes,
                                          const std::string &compression)
{
  int comp_type = IO_UNCOMP_STREAM_TYPE_INFER;
  if (compression == "gzip")
    comp_type = IO_UNCOMP_STREAM_TYPE_GZIP;
  else if (compression == "zip")
    comp_type = IO_UNCOMP_STREAM_TYPE_ZIP;
  else if (compression == "bz2")
    comp_type = IO_UNCOMP_STREAM_TYPE_BZIP2;
  else if (compression == "xz")
    comp_type = IO_UNCOMP_STREAM_TYPE_XZ;

  return io_uncompress_single_h2d(h_data, num_bytes, comp_type);
}

/**
 * @Brief ZLIB host decompressor class
 */
class HostDecompressor_ZLIB : public HostDecompressor {
 public:
  HostDecompressor_ZLIB(bool gz_hdr_) : gz_hdr(gz_hdr_) {}
  size_t Decompress(uint8_t *dstBytes,
                    size_t dstLen,
                    const uint8_t *srcBytes,
                    size_t srcLen) override
  {
    if (gz_hdr) {
      gz_archive_s gz;
      if (!ParseGZArchive(&gz, srcBytes, srcLen)) { return 0; }
      srcBytes = gz.comp_data;
      srcLen   = gz.comp_len;
    }
    if (0 == cpu_inflate(dstBytes, &dstLen, srcBytes, srcLen)) {
      return dstLen;
    } else {
      return 0;
    }
  }

 protected:
  const bool gz_hdr;
};

/**
 * @Brief SNAPPY host decompressor class
 */
class HostDecompressor_SNAPPY : public HostDecompressor {
 public:
  HostDecompressor_SNAPPY() {}
  size_t Decompress(uint8_t *dstBytes,
                    size_t dstLen,
                    const uint8_t *srcBytes,
                    size_t srcLen) override
  {
    uint32_t uncompressed_size, bytes_left, dst_pos;
    const uint8_t *cur = srcBytes;
    const uint8_t *end = srcBytes + srcLen;

    if (!dstBytes || srcLen < 1) { return 0; }
    // Read uncompressed length (varint)
    {
      uint32_t l        = 0, c;
      uncompressed_size = 0;
      do {
        uint32_t lo7;
        c   = *cur++;
        lo7 = c & 0x7f;
        if (l >= 28 && c > 0xf) { return 0; }
        uncompressed_size |= lo7 << l;
        l += 7;
      } while (c > 0x7f && cur < end);
      if (!uncompressed_size || uncompressed_size > dstLen || cur >= end) {
        // Destination buffer too small or zero size
        return 0;
      }
    }
    // Decode lz77
    dst_pos    = 0;
    bytes_left = uncompressed_size;
    do {
      uint32_t blen = *cur++;

      if (blen & 3) {
        // Copy
        uint32_t offset;
        if (blen & 2) {
          // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
          if (cur + 2 > end) break;
          offset = *(const uint16_t *)cur;
          cur += 2;
          if (blen & 1)  // 4-byte offset
          {
            if (cur + 2 > end) break;
            offset |= (*(const uint16_t *)cur) << 16;
            cur += 2;
          }
          blen = (blen >> 2) + 1;
        } else {
          // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
          if (cur >= end) break;
          offset = ((blen & 0xe0) << 3) | (*cur++);
          blen   = ((blen >> 2) & 7) + 4;
        }
        if (offset - 1u >= dst_pos || blen > bytes_left) break;
        bytes_left -= blen;
        do {
          dstBytes[dst_pos] = dstBytes[dst_pos - offset];
          dst_pos++;
        } while (--blen);
      } else {
        // xxxxxx00: literal
        blen >>= 2;
        if (blen >= 60) {
          uint32_t num_bytes = blen - 59;
          if (cur + num_bytes >= end) break;
          blen = cur[0];
          if (num_bytes > 1) {
            blen |= cur[1] << 8;
            if (num_bytes > 2) {
              blen |= cur[2] << 16;
              if (num_bytes > 3) { blen |= cur[3] << 24; }
            }
          }
          cur += num_bytes;
        }
        blen++;
        if (cur + blen > end || blen > bytes_left) break;
        memcpy(dstBytes + dst_pos, cur, blen);
        cur += blen;
        dst_pos += blen;
        bytes_left -= blen;
      }
    } while (bytes_left && cur < end);
    return (bytes_left) ? 0 : uncompressed_size;
  }
};

/**
 * @Brief CPU decompression class
 *
 * @param stream_type[in] compression method (IO_UNCOMP_STREAM_TYPE_XXX)
 *
 * @returns corresponding HostDecompressor class, nullptr if failure
 */
std::unique_ptr<HostDecompressor> HostDecompressor::Create(int stream_type)
{
  switch (stream_type) {
    case IO_UNCOMP_STREAM_TYPE_GZIP: return std::make_unique<HostDecompressor_ZLIB>(true);
    case IO_UNCOMP_STREAM_TYPE_INFLATE: return std::make_unique<HostDecompressor_ZLIB>(false);
    case IO_UNCOMP_STREAM_TYPE_SNAPPY: return std::make_unique<HostDecompressor_SNAPPY>();
  }
  CUDF_FAIL("Unsupported compression type");
}

}  // namespace io
}  // namespace cudf
