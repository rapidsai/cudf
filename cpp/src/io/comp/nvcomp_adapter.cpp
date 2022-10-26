/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "nvcomp_adapter.hpp"
#include "nvcomp_adapter.cuh"

#include <cudf/utilities/error.hpp>
#include <io/utilities/config_utils.hpp>

#include <nvcomp/snappy.h>

#define NVCOMP_DEFLATE_HEADER <nvcomp/deflate.h>
#if __has_include(NVCOMP_DEFLATE_HEADER)
#include NVCOMP_DEFLATE_HEADER
#endif

#define NVCOMP_ZSTD_HEADER <nvcomp/zstd.h>
#if __has_include(NVCOMP_ZSTD_HEADER)
#include NVCOMP_ZSTD_HEADER
#endif

constexpr bool NVCOMP_HAS_ZSTD_DECOMP = NVCOMP_MAJOR_VERSION > 2 or
                                        (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION >= 3);

constexpr bool NVCOMP_HAS_ZSTD_COMP = NVCOMP_MAJOR_VERSION > 2 or
                                      (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION >= 4);

constexpr bool NVCOMP_HAS_DEFLATE = NVCOMP_MAJOR_VERSION > 2 or
                                    (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION >= 3);

constexpr bool NVCOMP_HAS_TEMPSIZE_EX = NVCOMP_MAJOR_VERSION > 2 or
                                        (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION > 3) or
                                        (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION == 3 and
                                         NVCOMP_PATCH_VERSION >= 1);

// ZSTD is stable for nvcomp 2.3.2 or newer
constexpr bool NVCOMP_ZSTD_IS_STABLE = NVCOMP_MAJOR_VERSION > 2 or
                                       (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION > 3) or
                                       (NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION == 3 and
                                        NVCOMP_PATCH_VERSION >= 2);

// Issue https://github.com/NVIDIA/spark-rapids/issues/6614 impacts nvCOMP 2.4.0 ZSTD decompression
// on compute 6.x
constexpr bool NVCOMP_ZSTD_IS_DISABLED_ON_PASCAL =
  NVCOMP_MAJOR_VERSION == 2 and NVCOMP_MINOR_VERSION == 4 and NVCOMP_PATCH_VERSION == 0;

namespace cudf::io::nvcomp {

// Dispatcher for nvcompBatched<format>DecompressGetTempSizeEx
template <typename... Args>
std::optional<nvcompStatus_t> batched_decompress_get_temp_size_ex(compression_type compression,
                                                                  Args&&... args)
{
  if constexpr (NVCOMP_HAS_TEMPSIZE_EX) {
    switch (compression) {
      case compression_type::SNAPPY:
        return nvcompBatchedSnappyDecompressGetTempSizeEx(std::forward<Args>(args)...);
      case compression_type::ZSTD:
        if constexpr (NVCOMP_HAS_ZSTD_DECOMP) {
          return nvcompBatchedZstdDecompressGetTempSizeEx(std::forward<Args>(args)...);
        } else {
          return std::nullopt;
        }
      case compression_type::DEFLATE: [[fallthrough]];
      default: return std::nullopt;
    }
  }
  return std::nullopt;
}

// Dispatcher for nvcompBatched<format>DecompressGetTempSize
template <typename... Args>
auto batched_decompress_get_temp_size(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_DECOMP) {
        return nvcompBatchedZstdDecompressGetTempSize(std::forward<Args>(args)...);
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    case compression_type::DEFLATE:
      if constexpr (NVCOMP_HAS_DEFLATE) {
        return nvcompBatchedDeflateDecompressGetTempSize(std::forward<Args>(args)...);
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    default: CUDF_FAIL("Unsupported compression type");
  }
}

// Dispatcher for nvcompBatched<format>DecompressAsync
template <typename... Args>
auto batched_decompress_async(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressAsync(std::forward<Args>(args)...);
    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_DECOMP) {
        return nvcompBatchedZstdDecompressAsync(std::forward<Args>(args)...);
      } else {
        CUDF_FAIL("Unsupported compression type");
      }

    case compression_type::DEFLATE:
      if constexpr (NVCOMP_HAS_DEFLATE) {
        return nvcompBatchedDeflateDecompressAsync(std::forward<Args>(args)...);
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    default: CUDF_FAIL("Unsupported compression type");
  }
}

size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size)
{
  size_t temp_size   = 0;
  auto nvcomp_status = batched_decompress_get_temp_size_ex(
    compression, num_chunks, max_uncomp_chunk_size, &temp_size, max_total_uncomp_size);

  if (nvcomp_status.value_or(nvcompStatus_t::nvcompErrorInternal) !=
      nvcompStatus_t::nvcompSuccess) {
    nvcomp_status =
      batched_decompress_get_temp_size(compression, num_chunks, max_uncomp_chunk_size, &temp_size);
  }

  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for decompression");

  return temp_size;
}

void check_is_zstd_enabled()
{
  CUDF_EXPECTS(NVCOMP_HAS_ZSTD_DECOMP, "nvCOMP 2.3 or newer is required for Zstandard compression");
  CUDF_EXPECTS(NVCOMP_ZSTD_IS_STABLE or cudf::io::detail::nvcomp_integration::is_all_enabled(),
               "Zstandard compression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.");

  if constexpr (NVCOMP_ZSTD_IS_DISABLED_ON_PASCAL) {
    int device;
    int cc_major;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDF_EXPECTS(cc_major != 6, "Zstandard decompression is disabled on Pascal GPUs");
  }
}

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<compression_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream)
{
  if (compression == compression_type::ZSTD) { check_is_zstd_enabled(); }

  auto const num_chunks = inputs.size();

  // cuDF inflate inputs converted to nvcomp inputs
  auto const nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);
  rmm::device_uvector<size_t> actual_uncompressed_data_sizes(num_chunks, stream);
  rmm::device_uvector<nvcompStatus_t> nvcomp_statuses(num_chunks, stream);
  // Temporary space required for decompression
  auto const temp_size = batched_decompress_temp_size(
    compression, num_chunks, max_uncomp_chunk_size, max_total_uncomp_size);
  rmm::device_buffer scratch(temp_size, stream);
  auto const nvcomp_status = batched_decompress_async(compression,
                                                      nvcomp_args.input_data_ptrs.data(),
                                                      nvcomp_args.input_data_sizes.data(),
                                                      nvcomp_args.output_data_sizes.data(),
                                                      actual_uncompressed_data_sizes.data(),
                                                      num_chunks,
                                                      scratch.data(),
                                                      scratch.size(),
                                                      nvcomp_args.output_data_ptrs.data(),
                                                      nvcomp_statuses.data(),
                                                      stream.value());
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess, "unable to perform decompression");

  update_compression_results(nvcomp_statuses, actual_uncompressed_data_sizes, results, stream);
}

// Dispatcher for nvcompBatched<format>CompressGetTempSize
auto batched_compress_temp_size(compression_type compression,
                                size_t batch_size,
                                size_t max_uncompressed_chunk_bytes)
{
  size_t temp_size             = 0;
  nvcompStatus_t nvcomp_status = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      nvcomp_status = nvcompBatchedSnappyCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, nvcompBatchedSnappyDefaultOpts, &temp_size);
      break;
    case compression_type::DEFLATE:
      if constexpr (NVCOMP_HAS_DEFLATE) {
        nvcomp_status = nvcompBatchedDeflateCompressGetTempSize(
          batch_size, max_uncompressed_chunk_bytes, nvcompBatchedDeflateDefaultOpts, &temp_size);
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_COMP) {
        nvcomp_status = nvcompBatchedZstdCompressGetTempSize(
          batch_size, max_uncompressed_chunk_bytes, nvcompBatchedZstdDefaultOpts, &temp_size);
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }

    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for compression");
  return temp_size;
}

size_t compress_max_output_chunk_size(compression_type compression,
                                      uint32_t max_uncompressed_chunk_bytes)
{
  auto const capped_uncomp_bytes = std::min<size_t>(
    compress_max_allowed_chunk_size(compression).value_or(max_uncompressed_chunk_bytes),
    max_uncompressed_chunk_bytes);

  size_t max_comp_chunk_size = 0;
  nvcompStatus_t status      = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedSnappyDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::DEFLATE:
      if constexpr (NVCOMP_HAS_DEFLATE) {
        status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
          capped_uncomp_bytes, nvcompBatchedDeflateDefaultOpts, &max_comp_chunk_size);
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_COMP) {
        status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
          capped_uncomp_bytes, nvcompBatchedZstdDefaultOpts, &max_comp_chunk_size);
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(status == nvcompStatus_t::nvcompSuccess,
               "failed to get max uncompressed chunk size");
  return max_comp_chunk_size;
}

// Dispatcher for nvcompBatched<format>CompressAsync
static void batched_compress_async(compression_type compression,
                                   const void* const* device_uncompressed_ptrs,
                                   const size_t* device_uncompressed_bytes,
                                   size_t max_uncompressed_chunk_bytes,
                                   size_t batch_size,
                                   void* device_temp_ptr,
                                   size_t temp_bytes,
                                   void* const* device_compressed_ptrs,
                                   size_t* device_compressed_bytes,
                                   rmm::cuda_stream_view stream)
{
  nvcompStatus_t nvcomp_status = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      nvcomp_status = nvcompBatchedSnappyCompressAsync(device_uncompressed_ptrs,
                                                       device_uncompressed_bytes,
                                                       max_uncompressed_chunk_bytes,
                                                       batch_size,
                                                       device_temp_ptr,
                                                       temp_bytes,
                                                       device_compressed_ptrs,
                                                       device_compressed_bytes,
                                                       nvcompBatchedSnappyDefaultOpts,
                                                       stream.value());
      break;
    case compression_type::DEFLATE:
      if constexpr (NVCOMP_HAS_DEFLATE) {
        nvcomp_status = nvcompBatchedDeflateCompressAsync(device_uncompressed_ptrs,
                                                          device_uncompressed_bytes,
                                                          max_uncompressed_chunk_bytes,
                                                          batch_size,
                                                          device_temp_ptr,
                                                          temp_bytes,
                                                          device_compressed_ptrs,
                                                          device_compressed_bytes,
                                                          nvcompBatchedDeflateDefaultOpts,
                                                          stream.value());
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }

    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_COMP) {
        nvcomp_status = nvcompBatchedZstdCompressAsync(device_uncompressed_ptrs,
                                                       device_uncompressed_bytes,
                                                       max_uncompressed_chunk_bytes,
                                                       batch_size,
                                                       device_temp_ptr,
                                                       temp_bytes,
                                                       device_compressed_ptrs,
                                                       device_compressed_bytes,
                                                       nvcompBatchedZstdDefaultOpts,
                                                       stream.value());
        break;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }

    default: CUDF_FAIL("Unsupported compression type");
  }
  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess, "Error in compression");
}

bool is_aligned(void const* ptr, std::uintptr_t alignment) noexcept
{
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<compression_result> results,
                      rmm::cuda_stream_view stream)
{
  auto const num_chunks = inputs.size();

  auto nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);

  auto const max_uncomp_chunk_size = skip_unsupported_inputs(
    nvcomp_args.input_data_sizes, results, compress_max_allowed_chunk_size(compression), stream);

  auto const temp_size = batched_compress_temp_size(compression, num_chunks, max_uncomp_chunk_size);
  rmm::device_buffer scratch(temp_size, stream);
  CUDF_EXPECTS(is_aligned(scratch.data(), 8), "Compression failed, misaligned scratch buffer");

  rmm::device_uvector<size_t> actual_compressed_data_sizes(num_chunks, stream);

  batched_compress_async(compression,
                         nvcomp_args.input_data_ptrs.data(),
                         nvcomp_args.input_data_sizes.data(),
                         max_uncomp_chunk_size,
                         num_chunks,
                         scratch.data(),
                         scratch.size(),
                         nvcomp_args.output_data_ptrs.data(),
                         actual_compressed_data_sizes.data(),
                         stream.value());

  update_compression_results(actual_compressed_data_sizes, results, stream);
}

bool is_compression_enabled(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE:
      // See https://github.com/rapidsai/cudf/issues/11812
      return false;
    case compression_type::SNAPPY: return detail::nvcomp_integration::is_stable_enabled();
    case compression_type::ZSTD:
      return NVCOMP_HAS_ZSTD_COMP and detail::nvcomp_integration::is_all_enabled();
    default: return false;
  }
  return false;
}

size_t compress_input_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 0;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 2;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

size_t compress_output_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 3;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 0;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 64 * 1024;
    case compression_type::SNAPPY: return std::nullopt;
    case compression_type::ZSTD:
      if constexpr (NVCOMP_HAS_ZSTD_COMP) {
        return nvcompZstdCompressionMaxAllowedChunkSize;
      } else {
        CUDF_FAIL("Unsupported compression type");
      }
    default: return std::nullopt;
  }
}

}  // namespace cudf::io::nvcomp
