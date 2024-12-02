/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "io/utilities/hostdevice_vector.hpp"
#include "io_uncomp.hpp"
#include "nvcomp_adapter.hpp"
#include "unbz2.hpp"  // bz2 uncompress

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <zlib.h>  // uncompress

#include <cstring>  // memset

namespace cudf::io::detail {

#pragma pack(push, 1)

namespace {

struct gz_file_header_s {
  uint8_t id1;        // 0x1f
  uint8_t id2;        // 0x8b
  uint8_t comp_mthd;  // compression method (0-7=reserved, 8=deflate)
  uint8_t flags;      // flags (GZIPHeaderFlag)
  uint8_t mtime[4];   // If non-zero: modification time (Unix format)  // NOLINT
  uint8_t xflags;     // Extra compressor-specific flags
  uint8_t os;         // OS id
};

struct zip_eocd_s  // end of central directory
{
  uint32_t sig;            // 0x0605'4b50
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
  uint32_t sig;         // 0x0706'4b50
  uint32_t disk_start;  // number of the disk with the start of the zip64 end of central directory
  uint64_t eocdr_ofs;   // relative offset of the zip64 end of central directory record
  uint32_t num_disks;   // total number of disks
};

struct zip_cdfh_s  // central directory file header
{
  uint32_t sig;          // 0x0201'4b50
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
  uint32_t sig;          // 0x0403'4b50
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
  uint8_t sig[3];  // "BZh" // NOLINT
  uint8_t blksz;   // block size 1..9 in 100kB units (post-RLE)
};

#pragma pack(pop)

struct gz_archive_s {
  gz_file_header_s const* fhdr;
  uint16_t hcrc16;  // header crc16 if present
  uint16_t xlen;
  uint8_t const* fxtra;      // xlen bytes (optional)
  uint8_t const* fname;      // zero-terminated original filename if present
  uint8_t const* fcomment;   // zero-terminated comment if present
  uint8_t const* comp_data;  // compressed data
  size_t comp_len;           // Compressed data length
  uint32_t crc32;            // CRC32 of uncompressed data
  uint32_t isize;            // Input size modulo 2^32
};

struct zip_archive_s {
  zip_eocd_s const* eocd;    // end of central directory
  zip64_eocdl const* eocdl;  // end of central dir locator (optional)
  zip_cdfh_s const* cdfh;    // start of central directory file headers
};

bool ParseGZArchive(gz_archive_s* dst, uint8_t const* raw, size_t len)
{
  gz_file_header_s const* fhdr;

  if (!dst) return false;
  memset(dst, 0, sizeof(gz_archive_s));
  if (len < sizeof(gz_file_header_s) + 8) return false;
  fhdr = reinterpret_cast<gz_file_header_s const*>(raw);
  if (fhdr->id1 != 0x1f || fhdr->id2 != 0x8b) return false;
  dst->fhdr = fhdr;
  raw += sizeof(gz_file_header_s);
  len -= sizeof(gz_file_header_s);
  if (fhdr->flags & GZIPHeaderFlag::fextra) {
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
  if (fhdr->flags & GZIPHeaderFlag::fname) {
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
  if (fhdr->flags & GZIPHeaderFlag::fcomment) {
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
  if (fhdr->flags & GZIPHeaderFlag::fhcrc) {
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

bool OpenZipArchive(zip_archive_s* dst, uint8_t const* raw, size_t len)
{
  memset(dst, 0, sizeof(zip_archive_s));
  // Find the end of central directory
  if (len >= sizeof(zip_eocd_s) + 2) {
    for (ptrdiff_t i = len - sizeof(zip_eocd_s) - 2;
         i + sizeof(zip_eocd_s) + 2 + 0xffff >= len && i >= 0;
         i--) {
      auto const* eocd = reinterpret_cast<zip_eocd_s const*>(raw + i);
      if (eocd->sig == 0x0605'4b50 &&
          eocd->disk_id == eocd->start_disk  // multi-file archives not supported
          && eocd->num_entries == eocd->total_entries &&
          eocd->cdir_size >= sizeof(zip_cdfh_s) * eocd->num_entries && eocd->cdir_offset < len &&
          i + *reinterpret_cast<uint16_t const*>(eocd + 1) <= static_cast<ptrdiff_t>(len)) {
        auto const* cdfh = reinterpret_cast<zip_cdfh_s const*>(raw + eocd->cdir_offset);
        dst->eocd        = eocd;
        if (i >= static_cast<ptrdiff_t>(sizeof(zip64_eocdl))) {
          auto const* eocdl = reinterpret_cast<zip64_eocdl const*>(raw + i - sizeof(zip64_eocdl));
          if (eocdl->sig == 0x0706'4b50) { dst->eocdl = eocdl; }
        }
        // Start of central directory
        if (cdfh->sig == 0x0201'4b50) { dst->cdfh = cdfh; }
      }
    }
  }
  return (dst->eocd && dst->cdfh);
}

int cpu_inflate(uint8_t* uncomp_data, size_t* destLen, uint8_t const* comp_data, size_t comp_len)
{
  int zerr;
  z_stream strm;

  memset(&strm, 0, sizeof(strm));
  strm.next_in   = const_cast<Bytef*>(reinterpret_cast<Bytef const*>(comp_data));
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
 * @brief Uncompresses a raw DEFLATE stream to a char vector.
 * The vector will be grown to match the uncompressed size
 * Optimized for the case where the initial size is the uncompressed
 * size truncated to 32-bit, and grows the buffer in 1GB increments.
 *
 * @param[out] dst Destination vector
 * @param[in] comp_data Raw compressed data
 * @param[in] comp_len Compressed data size
 */
void cpu_inflate_vector(std::vector<uint8_t>& dst, uint8_t const* comp_data, size_t comp_len)
{
  z_stream strm{};
  strm.next_in   = const_cast<Bytef*>(reinterpret_cast<Bytef const*>(comp_data));
  strm.avail_in  = comp_len;
  strm.total_in  = 0;
  strm.next_out  = dst.data();
  strm.avail_out = dst.size();
  strm.total_out = 0;
  auto zerr      = inflateInit2(&strm, -15);  // -15 for raw data without GZIP headers
  CUDF_EXPECTS(zerr == 0, "Error in DEFLATE stream: inflateInit2 failed");
  do {
    if (strm.avail_out == 0) {
      dst.resize(strm.total_out + (1 << 30));
      strm.avail_out = dst.size() - strm.total_out;
      strm.next_out  = reinterpret_cast<uint8_t*>(dst.data()) + strm.total_out;
    }
    zerr = inflate(&strm, Z_SYNC_FLUSH);
  } while ((zerr == Z_BUF_ERROR || zerr == Z_OK) && strm.avail_out == 0 &&
           strm.total_out == dst.size());
  dst.resize(strm.total_out);
  inflateEnd(&strm);
  CUDF_EXPECTS(zerr == Z_STREAM_END, "Error in DEFLATE stream: Z_STREAM_END not encountered");
}

/**
 * @brief ZLIB host decompressor (no header)
 */
size_t decompress_zlib(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  size_t uncomp_size = dst.size();
  CUDF_EXPECTS(0 == cpu_inflate(dst.data(), &uncomp_size, src.data(), src.size()),
               "ZLIB decompression failed");
  return uncomp_size;
}

/**
 * @brief GZIP host decompressor (includes header)
 */
size_t decompress_gzip(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  gz_archive_s gz;
  auto const parse_succeeded = ParseGZArchive(&gz, src.data(), src.size());
  CUDF_EXPECTS(parse_succeeded, "Failed to parse GZIP header");
  return decompress_zlib({gz.comp_data, gz.comp_len}, dst);
}

/**
 * @brief SNAPPY host decompressor
 */
size_t decompress_snappy(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  CUDF_EXPECTS(not dst.empty() and src.size() >= 1, "invalid Snappy decompress inputs");
  uint32_t uncompressed_size, bytes_left, dst_pos;
  auto cur       = src.begin();
  auto const end = src.end();
  // Read uncompressed length (varint)
  {
    uint32_t l        = 0, c;
    uncompressed_size = 0;
    do {
      c              = *cur++;
      auto const lo7 = c & 0x7f;
      if (l >= 28 && c > 0xf) { return 0; }
      uncompressed_size |= lo7 << l;
      l += 7;
    } while (c > 0x7f && cur < end);
    CUDF_EXPECTS(uncompressed_size != 0 and uncompressed_size <= dst.size() and cur < end,
                 "Destination buffer too small");
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
        offset = *reinterpret_cast<uint16_t const*>(cur);
        cur += 2;
        if (blen & 1)  // 4-byte offset
        {
          if (cur + 2 > end) break;
          offset |= (*reinterpret_cast<uint16_t const*>(cur)) << 16;
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
        dst[dst_pos] = dst[dst_pos - offset];
        dst_pos++;
      } while (--blen);
    } else {
      // xxxxxx00: literal
      blen >>= 2;
      if (blen >= 60) {
        uint32_t const num_bytes = blen - 59;
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
      memcpy(dst.data() + dst_pos, cur, blen);
      cur += blen;
      dst_pos += blen;
      bytes_left -= blen;
    }
  } while (bytes_left && cur < end);
  CUDF_EXPECTS(bytes_left == 0, "Snappy Decompression failed");
  return uncompressed_size;
}

/**
 * @brief ZSTD decompressor that uses nvcomp
 */
size_t decompress_zstd(host_span<uint8_t const> src,
                       host_span<uint8_t> dst,
                       rmm::cuda_stream_view stream)
{
  // Init device span of spans (source)
  auto const d_src =
    cudf::detail::make_device_uvector_async(src, stream, cudf::get_current_device_resource_ref());
  auto hd_srcs = cudf::detail::hostdevice_vector<device_span<uint8_t const>>(1, stream);
  hd_srcs[0]   = d_src;
  hd_srcs.host_to_device_async(stream);

  // Init device span of spans (temporary destination)
  auto d_dst   = rmm::device_uvector<uint8_t>(dst.size(), stream);
  auto hd_dsts = cudf::detail::hostdevice_vector<device_span<uint8_t>>(1, stream);
  hd_dsts[0]   = d_dst;
  hd_dsts.host_to_device_async(stream);

  auto hd_stats = cudf::detail::hostdevice_vector<compression_result>(1, stream);
  hd_stats[0]   = compression_result{0, compression_status::FAILURE};
  hd_stats.host_to_device_async(stream);
  auto const max_uncomp_page_size = dst.size();
  nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                             hd_srcs,
                             hd_dsts,
                             hd_stats,
                             max_uncomp_page_size,
                             max_uncomp_page_size,
                             stream);

  hd_stats.device_to_host_sync(stream);
  CUDF_EXPECTS(hd_stats[0].status == compression_status::SUCCESS, "ZSTD decompression failed");

  // Copy temporary output to `dst`
  cudf::detail::cuda_memcpy(dst.subspan(0, hd_stats[0].bytes_written),
                            device_span<uint8_t const>{d_dst.data(), hd_stats[0].bytes_written},
                            stream);

  return hd_stats[0].bytes_written;
}

struct source_properties {
  compression_type compression = compression_type::NONE;
  uint8_t const* comp_data     = nullptr;
  size_t comp_len              = 0;
  size_t uncomp_len            = 0;
};

source_properties get_source_properties(compression_type compression, host_span<uint8_t const> src)
{
  auto raw                 = src.data();
  uint8_t const* comp_data = nullptr;
  size_t comp_len          = 0;
  size_t uncomp_len        = 0;

  switch (compression) {
    case compression_type::AUTO:
    case compression_type::GZIP: {
      gz_archive_s gz;
      auto const parse_succeeded = ParseGZArchive(&gz, src.data(), src.size());
      CUDF_EXPECTS(parse_succeeded, "Failed to parse GZIP header while fetching source properties");
      compression = compression_type::GZIP;
      comp_data   = gz.comp_data;
      comp_len    = gz.comp_len;
      uncomp_len  = gz.isize;
      if (compression != compression_type::AUTO) break;
      [[fallthrough]];
    }
    case compression_type::ZIP: {
      zip_archive_s za;
      if (OpenZipArchive(&za, raw, src.size())) {
        size_t cdfh_ofs = 0;
        for (int i = 0; i < za.eocd->num_entries; i++) {
          auto const* cdfh = reinterpret_cast<zip_cdfh_s const*>(
            reinterpret_cast<uint8_t const*>(za.cdfh) + cdfh_ofs);
          int cdfh_len = sizeof(zip_cdfh_s) + cdfh->fname_len + cdfh->extra_len + cdfh->comment_len;
          if (cdfh_ofs + cdfh_len > za.eocd->cdir_size || cdfh->sig != 0x0201'4b50) {
            // Bad cdir
            break;
          }
          // For now, only accept with non-zero file sizes and DEFLATE
          if (cdfh->comp_method == 8 && cdfh->comp_size > 0 && cdfh->uncomp_size > 0) {
            size_t lfh_ofs  = cdfh->hdr_ofs;
            auto const* lfh = reinterpret_cast<zip_lfh_s const*>(raw + lfh_ofs);
            if (lfh_ofs + sizeof(zip_lfh_s) <= src.size() && lfh->sig == 0x0403'4b50 &&
                lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len <= src.size()) {
              if (lfh->comp_method == 8 && lfh->comp_size > 0 && lfh->uncomp_size > 0) {
                size_t file_start = lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len;
                size_t file_end   = file_start + lfh->comp_size;
                if (file_end <= src.size()) {
                  // Pick the first valid file of non-zero size (only 1 file expected in archive)
                  compression = compression_type::ZIP;
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
      if (compression != compression_type::AUTO) break;
      [[fallthrough]];
    }
    case compression_type::BZIP2: {
      if (src.size() > 4) {
        auto const* fhdr = reinterpret_cast<bz2_file_header_s const*>(raw);
        // Check for BZIP2 file signature "BZh1" to "BZh9"
        if (fhdr->sig[0] == 'B' && fhdr->sig[1] == 'Z' && fhdr->sig[2] == 'h' &&
            fhdr->blksz >= '1' && fhdr->blksz <= '9') {
          compression = compression_type::BZIP2;
          comp_data   = raw;
          comp_len    = src.size();
          uncomp_len  = 0;
        }
      }
      if (compression != compression_type::AUTO) break;
      [[fallthrough]];
    }
    case compression_type::SNAPPY: {
      uncomp_len     = 0;
      auto cur       = src.begin();
      auto const end = src.end();
      // Read uncompressed length (varint)
      {
        uint32_t l = 0, c;
        do {
          c              = *cur++;
          auto const lo7 = c & 0x7f;
          if (l >= 28 && c > 0xf) {
            uncomp_len = 0;
            break;
          }
          uncomp_len |= lo7 << l;
          l += 7;
        } while (c > 0x7f && cur < end);
        CUDF_EXPECTS(uncomp_len != 0 and cur < end, "Error in retrieving SNAPPY source properties");
      }
      comp_data = raw;
      comp_len  = src.size();
      if (compression != compression_type::AUTO) break;
      [[fallthrough]];
    }
    default: CUDF_FAIL("Unsupported compressed stream type");
  }

  return source_properties{compression, comp_data, comp_len, uncomp_len};
}

}  // namespace

size_t get_uncompressed_size(compression_type compression, host_span<uint8_t const> src)
{
  return get_source_properties(compression, src).uncomp_len;
}

size_t decompress(compression_type compression,
                  host_span<uint8_t const> src,
                  host_span<uint8_t> dst,
                  rmm::cuda_stream_view stream)
{
  switch (compression) {
    case compression_type::GZIP: return decompress_gzip(src, dst);
    case compression_type::ZLIB: return decompress_zlib(src, dst);
    case compression_type::SNAPPY: return decompress_snappy(src, dst);
    case compression_type::ZSTD: return decompress_zstd(src, dst, stream);
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::vector<uint8_t> decompress(compression_type compression, host_span<uint8_t const> src)
{
  CUDF_EXPECTS(src.data() != nullptr, "Decompression: Source cannot be nullptr");
  CUDF_EXPECTS(not src.empty(), "Decompression: Source size cannot be 0");

  auto srcprops = get_source_properties(compression, src);
  CUDF_EXPECTS(srcprops.comp_data != nullptr and srcprops.comp_len > 0,
               "Unsupported compressed stream type");

  if (srcprops.uncomp_len <= 0) {
    srcprops.uncomp_len =
      srcprops.comp_len * 4 + 4096;  // In case uncompressed size isn't known in advance, assume
                                     // ~4:1 compression for initial size
  }

  if (compression == compression_type::GZIP) {
    // INFLATE
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    decompress_gzip(src, dst);
    return dst;
  }
  if (compression == compression_type::ZIP) {
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    cpu_inflate_vector(dst, srcprops.comp_data, srcprops.comp_len);
    return dst;
  }
  if (compression == compression_type::BZIP2) {
    size_t src_ofs = 0;
    size_t dst_ofs = 0;
    int bz_err     = 0;
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    do {
      size_t dst_len = srcprops.uncomp_len - dst_ofs;
      bz_err         = cpu_bz2_uncompress(
        srcprops.comp_data, srcprops.comp_len, dst.data() + dst_ofs, &dst_len, &src_ofs);
      if (bz_err == BZ_OUTBUFF_FULL) {
        // TBD: We could infer the compression ratio based on produced/consumed byte counts
        // in order to minimize realloc events and over-allocation
        dst_ofs = dst_len;
        dst_len = srcprops.uncomp_len + (srcprops.uncomp_len / 2);
        dst.resize(dst_len);
        srcprops.uncomp_len = dst_len;
      } else if (bz_err == 0) {
        srcprops.uncomp_len = dst_len;
        dst.resize(srcprops.uncomp_len);
      }
    } while (bz_err == BZ_OUTBUFF_FULL);
    CUDF_EXPECTS(bz_err == 0, "Decompression: error in stream");
    return dst;
  }
  if (compression == compression_type::SNAPPY) {
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    decompress_snappy(src, dst);
    return dst;
  }

  CUDF_FAIL("Unsupported compressed stream type");
}

}  // namespace cudf::io::detail
