/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include "check_nvcomp_output_sizes.hpp"
#include "cudf_jni_apis.hpp"

#include <rmm/device_uvector.hpp>

#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

namespace {

constexpr char const* NVCOMP_ERROR_CLASS      = "ai/rapids/cudf/nvcomp/NvcompException";
constexpr char const* NVCOMP_CUDA_ERROR_CLASS = "ai/rapids/cudf/nvcomp/NvcompCudaException";
constexpr char const* ILLEGAL_ARG_CLASS       = "java/lang/IllegalArgumentException";
constexpr char const* UNSUPPORTED_CLASS       = "java/lang/UnsupportedOperationException";

void check_nvcomp_status(JNIEnv* env, nvcompStatus_t status)
{
  switch (status) {
    case nvcompSuccess: break;
    case nvcompErrorInvalidValue:
      cudf::jni::throw_java_exception(env, ILLEGAL_ARG_CLASS, "nvcomp invalid value");
      break;
    case nvcompErrorNotSupported:
      cudf::jni::throw_java_exception(env, UNSUPPORTED_CLASS, "nvcomp unsupported");
      break;
    case nvcompErrorCannotDecompress:
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "nvcomp cannot decompress");
      break;
    case nvcompErrorCudaError:
      cudf::jni::throw_java_exception(env, NVCOMP_CUDA_ERROR_CLASS, "nvcomp CUDA error");
      break;
    case nvcompErrorInternal:
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "nvcomp internal error");
      break;
    default:
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "nvcomp unknown error");
      break;
  }
}

}  // anonymous namespace

extern "C" {

// methods for lz4
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_max_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto max_chunk_size   = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedLZ4CompressGetTempSize(
      batch_size, max_chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetMaxOutputChunkSize(JNIEnv* env,
                                                                             jclass,
                                                                             jlong j_max_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto max_chunk_size         = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t max_output_size = 0;
    auto status                 = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      max_chunk_size, nvcompBatchedLZ4DefaultOpts, &max_output_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(max_output_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressAsync(JNIEnv* env,
                                                             jclass,
                                                             jlong j_in_ptrs,
                                                             jlong j_in_sizes,
                                                             jlong j_chunk_size,
                                                             jlong j_batch_size,
                                                             jlong j_temp_ptr,
                                                             jlong j_temp_size,
                                                             jlong j_out_ptrs,
                                                             jlong j_compressed_sizes_out_ptr,
                                                             jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptrs              = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto in_sizes             = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto chunk_size           = static_cast<std::size_t>(j_chunk_size);
    auto batch_size           = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr             = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size            = static_cast<std::size_t>(j_temp_size);
    auto out_ptrs             = reinterpret_cast<void* const*>(j_out_ptrs);
    auto compressed_out_sizes = reinterpret_cast<std::size_t*>(j_compressed_sizes_out_ptr);
    auto stream               = reinterpret_cast<cudaStream_t>(j_stream);
    auto status               = nvcompBatchedLZ4CompressAsync(in_ptrs,
                                                in_sizes,
                                                chunk_size,
                                                batch_size,
                                                temp_ptr,
                                                temp_size,
                                                out_ptrs,
                                                compressed_out_sizes,
                                                nvcompBatchedLZ4DefaultOpts,
                                                stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto chunk_size       = static_cast<std::size_t>(j_chunk_size);
    std::size_t temp_size = 0;
    auto status = nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressAsync(JNIEnv* env,
                                                               jclass,
                                                               jlong j_in_ptrs,
                                                               jlong j_in_sizes,
                                                               jlong j_out_sizes,
                                                               jlong j_batch_size,
                                                               jlong j_temp_ptr,
                                                               jlong j_temp_size,
                                                               jlong j_out_ptrs,
                                                               jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs           = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes          = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes        = reinterpret_cast<std::size_t const*>(j_out_sizes);
    auto batch_size                = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr                  = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size                 = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptrs         = reinterpret_cast<void* const*>(j_out_ptrs);
    auto stream                    = reinterpret_cast<cudaStream_t>(j_stream);
    auto uncompressed_statuses     = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto actual_uncompressed_sizes = rmm::device_uvector<std::size_t>(batch_size, stream);
    auto status                    = nvcompBatchedLZ4DecompressAsync(compressed_ptrs,
                                                  compressed_sizes,
                                                  uncompressed_sizes,
                                                  actual_uncompressed_sizes.data(),
                                                  batch_size,
                                                  temp_ptr,
                                                  temp_size,
                                                  uncompressed_ptrs,
                                                  uncompressed_statuses.data(),
                                                  stream);
    check_nvcomp_status(env, status);
    if (!cudf::java::check_nvcomp_output_sizes(
          uncompressed_sizes, actual_uncompressed_sizes.data(), batch_size, stream)) {
      cudf::jni::throw_java_exception(
        env, NVCOMP_ERROR_CLASS, "nvcomp decompress output size mismatch");
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4GetDecompressSizeAsync(JNIEnv* env,
                                                                      jclass,
                                                                      jlong j_in_ptrs,
                                                                      jlong j_in_sizes,
                                                                      jlong j_out_sizes,
                                                                      jlong j_batch_size,
                                                                      jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs    = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes   = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t*>(j_out_sizes);
    auto batch_size         = static_cast<std::size_t>(j_batch_size);
    auto stream             = reinterpret_cast<cudaStream_t>(j_stream);
    auto status             = nvcompBatchedLZ4GetDecompressSizeAsync(
      compressed_ptrs, compressed_sizes, uncompressed_sizes, batch_size, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

// methods for zstd
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_max_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto max_chunk_size   = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t temp_size = 0;
    auto status           = nvcompBatchedZstdCompressGetTempSize(
      batch_size, max_chunk_size, nvcompBatchedZstdDefaultOpts, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressGetMaxOutputChunkSize(
  JNIEnv* env, jclass, jlong j_max_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto max_chunk_size         = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t max_output_size = 0;
    auto status                 = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
      max_chunk_size, nvcompBatchedZstdDefaultOpts, &max_output_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(max_output_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdCompressAsync(JNIEnv* env,
                                                              jclass,
                                                              jlong j_in_ptrs,
                                                              jlong j_in_sizes,
                                                              jlong j_chunk_size,
                                                              jlong j_batch_size,
                                                              jlong j_temp_ptr,
                                                              jlong j_temp_size,
                                                              jlong j_out_ptrs,
                                                              jlong j_compressed_sizes_out_ptr,
                                                              jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptrs              = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto in_sizes             = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto chunk_size           = static_cast<std::size_t>(j_chunk_size);
    auto batch_size           = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr             = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size            = static_cast<std::size_t>(j_temp_size);
    auto out_ptrs             = reinterpret_cast<void* const*>(j_out_ptrs);
    auto compressed_out_sizes = reinterpret_cast<std::size_t*>(j_compressed_sizes_out_ptr);
    auto stream               = reinterpret_cast<cudaStream_t>(j_stream);
    auto status               = nvcompBatchedZstdCompressAsync(in_ptrs,
                                                 in_sizes,
                                                 chunk_size,
                                                 batch_size,
                                                 temp_ptr,
                                                 temp_size,
                                                 out_ptrs,
                                                 compressed_out_sizes,
                                                 nvcompBatchedZstdDefaultOpts,
                                                 stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdDecompressGetTempSize(
  JNIEnv* env, jclass, jlong j_batch_size, jlong j_chunk_size)
{
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size       = static_cast<std::size_t>(j_batch_size);
    auto chunk_size       = static_cast<std::size_t>(j_chunk_size);
    std::size_t temp_size = 0;
    auto status = nvcompBatchedZstdDecompressGetTempSize(batch_size, chunk_size, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdDecompressAsync(JNIEnv* env,
                                                                jclass,
                                                                jlong j_in_ptrs,
                                                                jlong j_in_sizes,
                                                                jlong j_out_sizes,
                                                                jlong j_batch_size,
                                                                jlong j_temp_ptr,
                                                                jlong j_temp_size,
                                                                jlong j_out_ptrs,
                                                                jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs           = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes          = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes        = reinterpret_cast<std::size_t const*>(j_out_sizes);
    auto batch_size                = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr                  = reinterpret_cast<void*>(j_temp_ptr);
    auto temp_size                 = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptrs         = reinterpret_cast<void* const*>(j_out_ptrs);
    auto stream                    = reinterpret_cast<cudaStream_t>(j_stream);
    auto uncompressed_statuses     = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto actual_uncompressed_sizes = rmm::device_uvector<std::size_t>(batch_size, stream);
    auto status                    = nvcompBatchedZstdDecompressAsync(compressed_ptrs,
                                                   compressed_sizes,
                                                   uncompressed_sizes,
                                                   actual_uncompressed_sizes.data(),
                                                   batch_size,
                                                   temp_ptr,
                                                   temp_size,
                                                   uncompressed_ptrs,
                                                   uncompressed_statuses.data(),
                                                   stream);
    check_nvcomp_status(env, status);
    if (!cudf::java::check_nvcomp_output_sizes(
          uncompressed_sizes, actual_uncompressed_sizes.data(), batch_size, stream)) {
      cudf::jni::throw_java_exception(
        env, NVCOMP_ERROR_CLASS, "nvcomp decompress output size mismatch");
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedZstdGetDecompressSizeAsync(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_in_ptrs,
                                                                       jlong j_in_sizes,
                                                                       jlong j_out_sizes,
                                                                       jlong j_batch_size,
                                                                       jlong j_stream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs    = reinterpret_cast<void const* const*>(j_in_ptrs);
    auto compressed_sizes   = reinterpret_cast<std::size_t const*>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t*>(j_out_sizes);
    auto batch_size         = static_cast<std::size_t>(j_batch_size);
    auto stream             = reinterpret_cast<cudaStream_t>(j_stream);
    auto status             = nvcompBatchedZstdGetDecompressSizeAsync(
      compressed_ptrs, compressed_sizes, uncompressed_sizes, batch_size, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

}  // extern "C"
