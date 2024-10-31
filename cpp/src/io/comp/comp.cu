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

#include "comp.hpp"
#include "gpuinflate.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <zlib.h>  // compress
#include <cstring>  // memset

namespace cudf {
namespace io {

#pragma pack(push, 1)

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

/**
 * @brief GZIP host compressor (includes header)
 */
std::vector<std::uint8_t> compress_gzip(host_span<uint8_t const> src, rmm::cuda_stream_view stream)
{
  z_stream zs;
  zs.zalloc = Z_NULL;
  zs.zfree = Z_NULL;
  zs.opaque = Z_NULL;
  zs.avail_in = src.size();
  zs.next_in = reinterpret_cast<unsigned char*>(const_cast<unsigned char*>(src.data()));

  std::vector<uint8_t> dst(src.size());
  zs.avail_out = src.size();
  zs.next_out = dst.data();

  int windowbits = 15;
  int gzip_encoding = 16;
  int ret = deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowbits | gzip_encoding, 8, Z_DEFAULT_STRATEGY);
  CUDF_EXPECTS(ret == Z_OK, "GZIP DEFLATE compression initialization failed.");

  deflate(&zs, Z_FINISH);
  deflateEnd(&zs);

  dst.resize(dst.size() - zs.avail_out);
  return dst;
}

/**
 * @brief SNAPPY host decompressor
 */
std::vector<std::uint8_t> compress_snappy(host_span<uint8_t const> src, rmm::cuda_stream_view stream)
{
  // TODO: to be completed
  rmm::device_uvector<std::uint8_t> d_src(src.size(), stream);
  cudf::detail::cuda_memcpy_async(device_span<uint8_t>{d_src}, src, stream);
  rmm::device_uvector<device_span<std::uint8_t const> const> d_srcspan(1, stream);

  rmm::device_uvector<std::uint8_t> d_dst(src.size(), stream);

  rmm::device_uvector<compression_result> d_status(1, stream);

  /*
  gpu_snap(device_span<device_span<std::uint8_t const> const>{d_src}, 
      device_span<device_span<std::uint8_t> const>{d_dst}, d_status, stream);
  */
  
  std::vector<uint8_t> dst(d_dst.size());
  cudf::detail::cuda_memcpy(host_span<uint8_t>{dst}, device_span<uint8_t const>{d_dst}, stream);
  return dst;
}

std::vector<std::uint8_t> compress(compression_type compression,
                  host_span<uint8_t const> src,
                  rmm::cuda_stream_view stream)
{
  switch (compression) {
    case compression_type::GZIP: return compress_gzip(src, stream);
    case compression_type::SNAPPY: return compress_snappy(src, stream);
    default: CUDF_FAIL("Unsupported compression type");
  }
}

}  // namespace io
}  // namespace cudf
