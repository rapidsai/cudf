/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "io/utilities/getenv_or.hpp"
#include "nvcomp_adapter.cuh"

#include <cudf/io/config_utils.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

#include <mutex>

namespace cudf::io::detail::nvcomp {
namespace {

[[nodiscard]] std::string nvcomp_status_to_string(nvcompStatus_t status)
{
  switch (status) {
    case nvcompStatus_t::nvcompSuccess: return "nvcompSuccess";
    case nvcompStatus_t::nvcompErrorInvalidValue: return "nvcompErrorInvalidValue";
    case nvcompStatus_t::nvcompErrorNotSupported: return "nvcompErrorNotSupported";
    case nvcompStatus_t::nvcompErrorCannotDecompress: return "nvcompErrorCannotDecompress";
    case nvcompStatus_t::nvcompErrorBadChecksum: return "nvcompErrorBadChecksum";
    case nvcompStatus_t::nvcompErrorCannotVerifyChecksums:
      return "nvcompErrorCannotVerifyChecksums";
    case nvcompStatus_t::nvcompErrorOutputBufferTooSmall: return "nvcompErrorOutputBufferTooSmall";
    case nvcompStatus_t::nvcompErrorWrongHeaderLength: return "nvcompErrorWrongHeaderLength";
    case nvcompStatus_t::nvcompErrorAlignment: return "nvcompErrorAlignment";
    case nvcompStatus_t::nvcompErrorChunkSizeTooLarge: return "nvcompErrorChunkSizeTooLarge";
    case nvcompStatus_t::nvcompErrorCudaError: return "nvcompErrorCudaError";
    case nvcompStatus_t::nvcompErrorInternal: return "nvcompErrorInternal";
  }
  return "nvcompStatus_t(" + std::to_string(static_cast<int>(status)) + ")";
}

[[nodiscard]] std::string compression_type_name(compression_type compression)
{
  switch (compression) {
    case compression_type::SNAPPY: return "Snappy";
    case compression_type::ZSTD: return "Zstandard";
    case compression_type::DEFLATE: return "Deflate";
    case compression_type::LZ4: return "LZ4";
    case compression_type::GZIP: return "GZIP";
  }
  return "compression_type(" + std::to_string(static_cast<int>(compression)) + ")";
}

#if NVCOMP_VER_MAJOR >= 5
[[nodiscard]] std::optional<bool> use_hw_decompression()
{
  auto const env = getenv("LIBCUDF_HW_DECOMPRESSION");
  if (env == nullptr) { return std::nullopt; }
  std::string val{env};
  std::transform(
    val.begin(), val.end(), val.begin(), [](unsigned char c) { return std::toupper(c); });
  return val == "ON";
}
#endif

#define CHECK_NVCOMP_STATUS(status)                                   \
  do {                                                                \
    CUDF_EXPECTS(status == nvcompStatus_t::nvcompSuccess,             \
                 "nvCOMP error: " + nvcomp_status_to_string(status)); \
  } while (0)

#define UNSUPPORTED_COMPRESSION(compression)                                          \
  do {                                                                                \
    CUDF_FAIL("Unsupported compression type: " + compression_type_name(compression)); \
  } while (0)

#if NVCOMP_VER_MAJOR >= 5
// Dispatcher for nvcompBatched<format>DecompressGetTempSize
template <typename... Args>
auto batched_decompress_get_temp_size(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressGetTempSize(std::forward<Args>(args)...);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}
#else
// Dispatcher for nvcompBatched<format>DecompressGetTempSizeEx
template <typename... Args>
auto batched_decompress_get_temp_size_ex(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressGetTempSizeEx(std::forward<Args>(args)...);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}
#endif

#if NVCOMP_VER_MAJOR >= 5
// Dispatcher for nvcompBatched<format>DecompressAsync
template <typename... Args>
auto batched_decompress_async(compression_type compression,
                              std::optional<bool> use_hw_decompression,
                              void const* const* device_compressed_chunk_ptrs,
                              size_t const* device_compressed_chunk_bytes,
                              size_t const* device_uncompressed_buffer_bytes,
                              size_t* device_uncompressed_chunk_bytes,
                              size_t num_chunks,
                              void* device_temp_ptr,
                              size_t temp_bytes,
                              void* const* device_uncompressed_chunk_ptrs,
                              nvcompStatus_t* device_statuses,
                              rmm::cuda_stream_view stream)
{
  switch (compression) {
    case compression_type::SNAPPY: {
      auto opts = nvcompBatchedSnappyDecompressDefaultOpts;
      if (use_hw_decompression.has_value()) {
        opts.backend = *use_hw_decompression ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE
                                             : NVCOMP_DECOMPRESS_BACKEND_CUDA;
      }
      return nvcompBatchedSnappyDecompressAsync(device_compressed_chunk_ptrs,
                                                device_compressed_chunk_bytes,
                                                device_uncompressed_buffer_bytes,
                                                device_uncompressed_chunk_bytes,
                                                num_chunks,
                                                device_temp_ptr,
                                                temp_bytes,
                                                device_uncompressed_chunk_ptrs,
                                                opts,
                                                device_statuses,
                                                stream.value());
    }
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressAsync(device_compressed_chunk_ptrs,
                                              device_compressed_chunk_bytes,
                                              device_uncompressed_buffer_bytes,
                                              device_uncompressed_chunk_bytes,
                                              num_chunks,
                                              device_temp_ptr,
                                              temp_bytes,
                                              device_uncompressed_chunk_ptrs,
                                              nvcompBatchedZstdDecompressDefaultOpts,
                                              device_statuses,
                                              stream.value());
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressAsync(device_compressed_chunk_ptrs,
                                                 device_compressed_chunk_bytes,
                                                 device_uncompressed_buffer_bytes,
                                                 device_uncompressed_chunk_bytes,
                                                 num_chunks,
                                                 device_temp_ptr,
                                                 temp_bytes,
                                                 device_uncompressed_chunk_ptrs,
                                                 nvcompBatchedDeflateDecompressDefaultOpts,
                                                 device_statuses,
                                                 stream.value());
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressAsync(device_compressed_chunk_ptrs,
                                             device_compressed_chunk_bytes,
                                             device_uncompressed_buffer_bytes,
                                             device_uncompressed_chunk_bytes,
                                             num_chunks,
                                             device_temp_ptr,
                                             temp_bytes,
                                             device_uncompressed_chunk_ptrs,
                                             nvcompBatchedLZ4DecompressDefaultOpts,
                                             device_statuses,
                                             stream.value());
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressAsync(device_compressed_chunk_ptrs,
                                              device_compressed_chunk_bytes,
                                              device_uncompressed_buffer_bytes,
                                              device_uncompressed_chunk_bytes,
                                              num_chunks,
                                              device_temp_ptr,
                                              temp_bytes,
                                              device_uncompressed_chunk_ptrs,
                                              nvcompBatchedGzipDecompressDefaultOpts,
                                              device_statuses,
                                              stream.value());
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}
#else
// Dispatcher for nvcompBatched<format>DecompressAsync
template <typename... Args>
auto batched_decompress_async(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressAsync(std::forward<Args>(args)...);
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressAsync(std::forward<Args>(args)...);
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressAsync(std::forward<Args>(args)...);
    case compression_type::LZ4: return nvcompBatchedLZ4DecompressAsync(std::forward<Args>(args)...);
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressAsync(std::forward<Args>(args)...);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}
#endif

size_t batched_compress_temp_size(compression_type compression,
                                  size_t batch_size,
                                  size_t max_uncompressed_chunk_bytes,
                                  size_t max_total_uncompressed_bytes)
{
  size_t temp_size             = 0;
  nvcompStatus_t nvcomp_status = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      nvcomp_status = nvcompBatchedSnappyCompressGetTempSizeEx(batch_size,
                                                               max_uncompressed_chunk_bytes,
                                                               nvcompBatchedSnappyDefaultOpts,
                                                               &temp_size,
                                                               max_total_uncompressed_bytes);
      break;
    case compression_type::DEFLATE:
      nvcomp_status = nvcompBatchedDeflateCompressGetTempSizeEx(batch_size,
                                                                max_uncompressed_chunk_bytes,
                                                                nvcompBatchedDeflateDefaultOpts,
                                                                &temp_size,
                                                                max_total_uncompressed_bytes);
      break;
    case compression_type::ZSTD:
      nvcomp_status = nvcompBatchedZstdCompressGetTempSizeEx(batch_size,
                                                             max_uncompressed_chunk_bytes,
                                                             nvcompBatchedZstdDefaultOpts,
                                                             &temp_size,
                                                             max_total_uncompressed_bytes);
      break;
    case compression_type::LZ4:
      nvcomp_status = nvcompBatchedLZ4CompressGetTempSizeEx(batch_size,
                                                            max_uncompressed_chunk_bytes,
                                                            nvcompBatchedLZ4DefaultOpts,
                                                            &temp_size,
                                                            max_total_uncompressed_bytes);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(nvcomp_status);
  return temp_size;
}

#if NVCOMP_VER_MAJOR >= 5
// Dispatcher for nvcompBatched<format>CompressAsync
void batched_compress_async(compression_type compression,
                            void const* const* device_uncompressed_ptrs,
                            size_t const* device_uncompressed_bytes,
                            size_t max_uncompressed_chunk_bytes,
                            size_t batch_size,
                            void* device_temp_ptr,
                            size_t temp_bytes,
                            void* const* device_compressed_ptrs,
                            size_t* device_compressed_bytes,
                            nvcompStatus_t* device_nvcomp_statuses,
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
                                                       nvcompBatchedSnappyCompressionDefaultOpts,
                                                       device_nvcomp_statuses,
                                                       stream.value());
      break;
    case compression_type::DEFLATE:
      nvcomp_status = nvcompBatchedDeflateCompressAsync(device_uncompressed_ptrs,
                                                        device_uncompressed_bytes,
                                                        max_uncompressed_chunk_bytes,
                                                        batch_size,
                                                        device_temp_ptr,
                                                        temp_bytes,
                                                        device_compressed_ptrs,
                                                        device_compressed_bytes,
                                                        nvcompBatchedDeflateCompressionDefaultOpts,
                                                        device_nvcomp_statuses,
                                                        stream.value());
      break;
    case compression_type::ZSTD:
      nvcomp_status = nvcompBatchedZstdCompressAsync(device_uncompressed_ptrs,
                                                     device_uncompressed_bytes,
                                                     max_uncompressed_chunk_bytes,
                                                     batch_size,
                                                     device_temp_ptr,
                                                     temp_bytes,
                                                     device_compressed_ptrs,
                                                     device_compressed_bytes,
                                                     nvcompBatchedZstdCompressionDefaultOpts,
                                                     device_nvcomp_statuses,
                                                     stream.value());
      break;
    case compression_type::LZ4:
      nvcomp_status = nvcompBatchedLZ4CompressAsync(device_uncompressed_ptrs,
                                                    device_uncompressed_bytes,
                                                    max_uncompressed_chunk_bytes,
                                                    batch_size,
                                                    device_temp_ptr,
                                                    temp_bytes,
                                                    device_compressed_ptrs,
                                                    device_compressed_bytes,
                                                    nvcompBatchedLZ4CompressionDefaultOpts,
                                                    device_nvcomp_statuses,
                                                    stream.value());
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(nvcomp_status);
}
#else
// Dispatcher for nvcompBatched<format>CompressAsync
void batched_compress_async(compression_type compression,
                            void const* const* device_uncompressed_ptrs,
                            size_t const* device_uncompressed_bytes,
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
    case compression_type::ZSTD:
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
    case compression_type::LZ4:
      nvcomp_status = nvcompBatchedLZ4CompressAsync(device_uncompressed_ptrs,
                                                    device_uncompressed_bytes,
                                                    max_uncompressed_chunk_bytes,
                                                    batch_size,
                                                    device_temp_ptr,
                                                    temp_bytes,
                                                    device_compressed_ptrs,
                                                    device_compressed_bytes,
                                                    nvcompBatchedLZ4DefaultOpts,
                                                    stream.value());
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(nvcomp_status);
}
#endif

bool is_aligned(void const* ptr, std::uintptr_t alignment) noexcept
{
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

std::optional<std::string> is_compression_disabled_impl(compression_type compression,
                                                        feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE: {
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE compression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::LZ4:
    case compression_type::SNAPPY:
    case compression_type::ZSTD:
      if (not params.are_stable_integrations_enabled) {
        return "nvCOMP use is disabled through the `LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    default: return "Unsupported compression type";
  }
}

std::optional<std::string> is_decompression_disabled_impl(compression_type compression,
                                                          feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE:
    case compression_type::GZIP: {
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE decompression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::LZ4:
    case compression_type::SNAPPY:
    case compression_type::ZSTD: {
      if (not params.are_stable_integrations_enabled) {
        return "nvCOMP use is disabled through the `LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
  }
  return "Unsupported compression type";
}

}  // namespace

size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size)
{
  size_t temp_size = 0;
#if NVCOMP_VER_MAJOR >= 5
  // TODO: decompression options are expected to be added as parameters in the future
  nvcompStatus_t const nvcomp_status = batched_decompress_get_temp_size(
    compression, num_chunks, max_uncomp_chunk_size, max_total_uncomp_size, &temp_size);
#else
  nvcompStatus_t const nvcomp_status = batched_decompress_get_temp_size_ex(
    compression, num_chunks, max_uncomp_chunk_size, &temp_size, max_total_uncomp_size);
#endif
  CHECK_NVCOMP_STATUS(nvcomp_status);
  return temp_size;
}

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<codec_exec_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream)
{
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
#if NVCOMP_VER_MAJOR >= 5
                                                      use_hw_decompression(),
#endif
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
  CHECK_NVCOMP_STATUS(nvcomp_status);

  update_compression_results(nvcomp_statuses, actual_uncompressed_data_sizes, results, stream);
}

// Wrapper for nvcompBatched<format>CompressGetMaxOutputChunkSize
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
    case compression_type::GZIP:  // HACK!
      status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedDeflateDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::ZSTD:
      status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedZstdDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::LZ4:
      status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedLZ4DefaultOpts, &max_comp_chunk_size);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return max_comp_chunk_size;
}

void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<codec_exec_result> results,
                      rmm::cuda_stream_view stream)
{
  auto const num_chunks = inputs.size();

  auto nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);

  skip_unsupported_inputs(
    nvcomp_args.input_data_sizes, results, compress_max_allowed_chunk_size(compression), stream);

  auto const [max_uncomp_chunk_size, total_uncomp_size] =
    max_chunk_and_total_input_size(nvcomp_args.input_data_sizes, stream);

  auto const temp_size =
    batched_compress_temp_size(compression, num_chunks, max_uncomp_chunk_size, total_uncomp_size);

  rmm::device_buffer scratch(temp_size, stream);
  CUDF_EXPECTS(is_aligned(scratch.data(), 8), "Compression failed, misaligned scratch buffer");

  rmm::device_uvector<size_t> actual_compressed_data_sizes(num_chunks, stream);
#if NVCOMP_VER_MAJOR >= 5
  rmm::device_uvector<nvcompStatus_t> nvcomp_statuses(num_chunks, stream);
#endif

  batched_compress_async(compression,
                         nvcomp_args.input_data_ptrs.data(),
                         nvcomp_args.input_data_sizes.data(),
                         max_uncomp_chunk_size,
                         num_chunks,
                         scratch.data(),
                         scratch.size(),
                         nvcomp_args.output_data_ptrs.data(),
                         actual_compressed_data_sizes.data(),
#if NVCOMP_VER_MAJOR >= 5
                         nvcomp_statuses.data(),
#endif
                         stream.value());

#if NVCOMP_VER_MAJOR >= 5
  update_compression_results(nvcomp_statuses, actual_compressed_data_sizes, results, stream);
#else
  update_compression_results(actual_compressed_data_sizes, results, stream);
#endif
}

feature_status_parameters::feature_status_parameters()
  : feature_status_parameters(nvcomp_integration::is_all_enabled(),
                              nvcomp_integration::is_stable_enabled())
{
}

feature_status_parameters::feature_status_parameters(bool all_enabled, bool stable_enabled)
  : lib_major_version{NVCOMP_VER_MAJOR},
    lib_minor_version{NVCOMP_VER_MINOR},
    lib_patch_version{NVCOMP_VER_PATCH},
    are_all_integrations_enabled{all_enabled},
    are_stable_integrations_enabled{stable_enabled}
{
}

// Represents all parameters required to determine status of a compression/decompression feature
using feature_status_inputs = std::pair<compression_type, feature_status_parameters>;
struct hash_feature_status_inputs {
  size_t operator()(feature_status_inputs const& fsi) const
  {
    // Outside of unit tests, the same `feature_status_parameters` value will always be passed
    // within a run; for simplicity, only use `compression_type` for the hash
    return std::hash<compression_type>{}(fsi.first);
  }
};

// Hash map type that stores feature status for different combinations of input parameters
using feature_status_memo_map =
  std::unordered_map<feature_status_inputs, std::optional<std::string>, hash_feature_status_inputs>;

std::optional<std::string> is_compression_disabled(compression_type compression,
                                                   feature_status_parameters params)
{
  static feature_status_memo_map comp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = comp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != comp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                         = is_compression_disabled_impl(compression, params);
  comp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("nvCOMP is disabled for %s compression; reason: %s",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for %s compression", compression_type_name(compression));
  }

  return reason;
}

std::optional<std::string> is_decompression_disabled(compression_type compression,
                                                     feature_status_parameters params)
{
  static feature_status_memo_map decomp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = decomp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != decomp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                           = is_decompression_disabled_impl(compression, params);
  decomp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("nvCOMP is disabled for %s decompression; reason: %s",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for %s decompression", compression_type_name(compression));
  }

  return reason;
}
#if NVCOMP_VER_MAJOR >= 5
size_t compress_required_alignment(compression_type compression)
{
  nvcompAlignmentRequirements_t alignments{};
  nvcompStatus_t status;
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::DEFLATE:
      status = nvcompBatchedDeflateCompressGetRequiredAlignments(nvcompBatchedDeflateDefaultOpts,
                                                                 &alignments);
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyCompressGetRequiredAlignments(nvcompBatchedSnappyDefaultOpts,
                                                                &alignments);
    case compression_type::ZSTD:
      status =
        nvcompBatchedZstdCompressGetRequiredAlignments(nvcompBatchedZstdDefaultOpts, &alignments);
    case compression_type::LZ4:
      status =
        nvcompBatchedLZ4CompressGetRequiredAlignments(nvcompBatchedLZ4DefaultOpts, &alignments);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return std::max(
    {alignments.input_alignment, alignments.output_alignment, alignments.temp_alignment});
}

size_t decompress_required_alignment(compression_type compression)
{
  nvcompAlignmentRequirements_t alignments{};
  nvcompStatus_t status;
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::DEFLATE:
      status = nvcompBatchedDeflateDecompressGetRequiredAlignments(&alignments);
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyDecompressGetRequiredAlignments(&alignments);
    case compression_type::ZSTD:
      status = nvcompBatchedZstdDecompressGetRequiredAlignments(&alignments);
    case compression_type::LZ4:
      status = nvcompBatchedLZ4DecompressGetRequiredAlignments(&alignments);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return std::max(
    {alignments.input_alignment, alignments.output_alignment, alignments.temp_alignment});
}
#else
size_t compress_required_alignment(compression_type compression)
{
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::DEFLATE: return nvcompDeflateRequiredAlignment;
    case compression_type::SNAPPY: return nvcompSnappyRequiredAlignment;
    case compression_type::ZSTD: return nvcompZstdRequiredAlignment;
    case compression_type::LZ4: return nvcompLZ4RequiredAlignment;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

// TODO: check alignment in readers; we can't align input, but should make sure output is aligned
size_t decompress_required_alignment(compression_type compression)
{
  // nvcompBatched<format>DecompressGetRequiredAlignments is not available in nvcomp < 5.0
  return compress_required_alignment(compression);
}
#endif

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE:
    case compression_type::GZIP: return nvcompDeflateCompressionMaxAllowedChunkSize;
    case compression_type::SNAPPY: return nvcompSnappyCompressionMaxAllowedChunkSize;
    case compression_type::ZSTD: return nvcompZstdCompressionMaxAllowedChunkSize;
    case compression_type::LZ4: return nvcompLZ4CompressionMaxAllowedChunkSize;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

}  // namespace cudf::io::detail::nvcomp
