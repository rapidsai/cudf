/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "io/utilities/config_utils.hpp"
#include "nvcomp_adapter.cuh"

#include <cudf/utilities/error.hpp>

#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>

#include <mutex>

#define NVCOMP_DEFLATE_HEADER <nvcomp/deflate.h>
#if __has_include(NVCOMP_DEFLATE_HEADER)
#include NVCOMP_DEFLATE_HEADER
#endif

#define NVCOMP_ZSTD_HEADER <nvcomp/zstd.h>
#if __has_include(NVCOMP_ZSTD_HEADER)
#include NVCOMP_ZSTD_HEADER
#endif

#ifndef NVCOMP_MAJOR_VERSION
#define NVCOMP_MAJOR_VERSION NVCOMP_VER_MAJOR
#define NVCOMP_MINOR_VERSION NVCOMP_VER_MINOR
#define NVCOMP_PATCH_VERSION NVCOMP_VER_PATCH
#endif

#define NVCOMP_HAS_ZSTD_DECOMP(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 3))

#define NVCOMP_HAS_ZSTD_COMP(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 4))

#define NVCOMP_HAS_DEFLATE(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 5))

#define NVCOMP_HAS_DECOMP_TEMPSIZE_EX(MAJOR, MINOR, PATCH) \
  (MAJOR > 2 or (MAJOR == 2 and MINOR > 3) or (MAJOR == 2 and MINOR == 3 and PATCH >= 1))

#define NVCOMP_HAS_COMP_TEMPSIZE_EX(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 6))

// ZSTD is stable for nvcomp 2.3.2 or newer
#define NVCOMP_ZSTD_DECOMP_IS_STABLE(MAJOR, MINOR, PATCH) \
  (MAJOR > 2 or (MAJOR == 2 and MINOR > 3) or (MAJOR == 2 and MINOR == 3 and PATCH >= 2))

namespace cudf::io::nvcomp {

// Dispatcher for nvcompBatched<format>DecompressGetTempSizeEx
template <typename... Args>
std::optional<nvcompStatus_t> batched_decompress_get_temp_size_ex(compression_type compression,
                                                                  Args&&... args)
{
#if NVCOMP_HAS_DECOMP_TEMPSIZE_EX(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::ZSTD:
#if NVCOMP_HAS_ZSTD_DECOMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompBatchedZstdDecompressGetTempSizeEx(std::forward<Args>(args)...);
#else
      return std::nullopt;
#endif
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::DEFLATE: [[fallthrough]];
    default: return std::nullopt;
  }
#endif
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
#if NVCOMP_HAS_ZSTD_DECOMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompBatchedZstdDecompressGetTempSize(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                nvcomp::is_decompression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
    case compression_type::DEFLATE:
#if NVCOMP_HAS_DEFLATE(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompBatchedDeflateDecompressGetTempSize(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                nvcomp::is_decompression_disabled(nvcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSize(std::forward<Args>(args)...);
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
#if NVCOMP_HAS_ZSTD_DECOMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompBatchedZstdDecompressAsync(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                nvcomp::is_decompression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
    case compression_type::DEFLATE:
#if NVCOMP_HAS_DEFLATE(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompBatchedDeflateDecompressAsync(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                nvcomp::is_decompression_disabled(nvcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::LZ4: return nvcompBatchedLZ4DecompressAsync(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::string compression_type_name(compression_type compression)
{
  switch (compression) {
    case compression_type::SNAPPY: return "Snappy";
    case compression_type::ZSTD: return "Zstandard";
    case compression_type::DEFLATE: return "Deflate";
    case compression_type::LZ4: return "LZ4";
  }
  return "compression_type(" + std::to_string(static_cast<int>(compression)) + ")";
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

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<compression_result> results,
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

// Wrapper for nvcompBatched<format>CompressGetTempSize
auto batched_compress_get_temp_size(compression_type compression,
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
#if NVCOMP_HAS_DEFLATE(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      nvcomp_status = nvcompBatchedDeflateCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, nvcompBatchedDeflateDefaultOpts, &temp_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if NVCOMP_HAS_ZSTD_COMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      nvcomp_status = nvcompBatchedZstdCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, nvcompBatchedZstdDefaultOpts, &temp_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
    case compression_type::LZ4:
      nvcomp_status = nvcompBatchedLZ4CompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, nvcompBatchedLZ4DefaultOpts, &temp_size);
      break;
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for compression");
  return temp_size;
}

#if NVCOMP_HAS_COMP_TEMPSIZE_EX(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
// Wrapper for nvcompBatched<format>CompressGetTempSizeEx
auto batched_compress_get_temp_size_ex(compression_type compression,
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
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(nvcomp_status == nvcompStatus_t::nvcompSuccess,
               "Unable to get scratch size for compression");
  return temp_size;
}
#endif

size_t batched_compress_temp_size(compression_type compression,
                                  size_t num_chunks,
                                  size_t max_uncomp_chunk_size,
                                  size_t max_total_uncomp_size)
{
#if NVCOMP_HAS_COMP_TEMPSIZE_EX(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
  try {
    return batched_compress_get_temp_size_ex(
      compression, num_chunks, max_uncomp_chunk_size, max_total_uncomp_size);
  } catch (...) {
    // Ignore errors in the expanded version; fall back to the old API in case of failure
    CUDF_LOG_WARN(
      "CompressGetTempSizeEx call failed, falling back to CompressGetTempSize; this may increase "
      "the memory usage");
  }
#endif

  return batched_compress_get_temp_size(compression, num_chunks, max_uncomp_chunk_size);
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
#if NVCOMP_HAS_DEFLATE(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedDeflateDefaultOpts, &max_comp_chunk_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if NVCOMP_HAS_ZSTD_COMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedZstdDefaultOpts, &max_comp_chunk_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
    case compression_type::LZ4:
      status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedLZ4DefaultOpts, &max_comp_chunk_size);
      break;
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(status == nvcompStatus_t::nvcompSuccess,
               "failed to get max uncompressed chunk size");
  return max_comp_chunk_size;
}

// Dispatcher for nvcompBatched<format>CompressAsync
static void batched_compress_async(compression_type compression,
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
#if NVCOMP_HAS_DEFLATE(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
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
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if NVCOMP_HAS_ZSTD_COMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
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
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
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

  skip_unsupported_inputs(
    nvcomp_args.input_data_sizes, results, compress_max_allowed_chunk_size(compression), stream);

  auto const [max_uncomp_chunk_size, total_uncomp_size] =
    max_chunk_and_total_input_size(nvcomp_args.input_data_sizes, stream);

  auto const temp_size =
    batched_compress_temp_size(compression, num_chunks, max_uncomp_chunk_size, total_uncomp_size);

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

feature_status_parameters::feature_status_parameters()
  : lib_major_version{NVCOMP_MAJOR_VERSION},
    lib_minor_version{NVCOMP_MINOR_VERSION},
    lib_patch_version{NVCOMP_PATCH_VERSION},
    are_all_integrations_enabled{detail::nvcomp_integration::is_all_enabled()},
    are_stable_integrations_enabled{detail::nvcomp_integration::is_stable_enabled()}
{
  int device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, device));
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

std::optional<std::string> is_compression_disabled_impl(compression_type compression,
                                                        feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE: {
      if (not NVCOMP_HAS_DEFLATE(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "nvCOMP 2.5 or newer is required for Deflate compression";
      }
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE compression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::SNAPPY: {
      if (not params.are_stable_integrations_enabled) {
        return "Snappy compression has been disabled through the `LIBCUDF_NVCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::ZSTD: {
      if (not NVCOMP_HAS_ZSTD_COMP(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "nvCOMP 2.4 or newer is required for Zstandard compression";
      }
      if (not params.are_stable_integrations_enabled) {
        return "Zstandard compression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::LZ4:
      if (not params.are_stable_integrations_enabled) {
        return "LZ4 compression has been disabled through the `LIBCUDF_NVCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    default: return "Unsupported compression type";
  }
  return "Unsupported compression type";
}

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
    CUDF_LOG_INFO("nvCOMP is disabled for {} compression; reason: {}",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for {} compression", compression_type_name(compression));
  }

  return reason;
}

std::optional<std::string> is_zstd_decomp_disabled(feature_status_parameters const& params)
{
  if (not NVCOMP_HAS_ZSTD_DECOMP(
        params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
    return "nvCOMP 2.3 or newer is required for Zstandard decompression";
  }

  if (NVCOMP_ZSTD_DECOMP_IS_STABLE(
        params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
    if (not params.are_stable_integrations_enabled) {
      return "Zstandard decompression has been disabled through the `LIBCUDF_NVCOMP_POLICY` "
             "environment variable.";
    }
  } else if (not params.are_all_integrations_enabled) {
    return "Zstandard decompression is experimental, you can enable it through "
           "`LIBCUDF_NVCOMP_POLICY` environment variable.";
  }

  return std::nullopt;
}

std::optional<std::string> is_decompression_disabled_impl(compression_type compression,
                                                          feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE: {
      if (not NVCOMP_HAS_DEFLATE(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "nvCOMP 2.5 or newer is required for Deflate decompression";
      }
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE decompression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::SNAPPY: {
      if (not params.are_stable_integrations_enabled) {
        return "Snappy decompression has been disabled through the `LIBCUDF_NVCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::ZSTD: return is_zstd_decomp_disabled(params);
    case compression_type::LZ4: {
      if (not params.are_stable_integrations_enabled) {
        return "LZ4 decompression has been disabled through the `LIBCUDF_NVCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    }
    default: return "Unsupported compression type";
  }
  return "Unsupported compression type";
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
    CUDF_LOG_INFO("nvCOMP is disabled for {} decompression; reason: {}",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for {} decompression", compression_type_name(compression));
  }

  return reason;
}

size_t compress_input_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 0;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 2;
    case compression_type::LZ4: return 2;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

size_t compress_output_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 3;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 0;
    case compression_type::LZ4: return 2;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 64 * 1024;
    case compression_type::SNAPPY: return std::nullopt;
    case compression_type::ZSTD:
#if NVCOMP_HAS_ZSTD_COMP(NVCOMP_MAJOR_VERSION, NVCOMP_MINOR_VERSION, NVCOMP_PATCH_VERSION)
      return nvcompZstdCompressionMaxAllowedChunkSize;
#else
      CUDF_FAIL("Compression error: " +
                nvcomp::is_compression_disabled(nvcomp::compression_type::ZSTD).value());
#endif
    case compression_type::LZ4: return 16 * 1024 * 1024;
    default: return std::nullopt;
  }
}

}  // namespace cudf::io::nvcomp
