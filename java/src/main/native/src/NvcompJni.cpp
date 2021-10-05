/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <nvcomp.h>

#include <nvcomp/lz4.h>
#include <rmm/device_uvector.hpp>

#include "check_output_sizes.hpp"
#include "cudf_jni_apis.hpp"

namespace {

constexpr char const *NVCOMP_ERROR_CLASS = "ai/rapids/cudf/nvcomp/NvcompException";
constexpr char const *NVCOMP_CUDA_ERROR_CLASS = "ai/rapids/cudf/nvcomp/NvcompCudaException";
constexpr char const *ILLEGAL_ARG_CLASS = "java/lang/IllegalArgumentException";
constexpr char const *UNSUPPORTED_CLASS = "java/lang/UnsupportedOperationException";

void check_nvcomp_status(JNIEnv *env, nvcompStatus_t status) {
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

} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetMetadata(
    JNIEnv *env, jclass, jlong in_ptr, jlong in_size, jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    void *metadata_ptr = nullptr;
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompDecompressGetMetadata(reinterpret_cast<void *>(in_ptr), in_size,
                                              &metadata_ptr, stream);
    check_nvcomp_status(env, status);
    return reinterpret_cast<jlong>(metadata_ptr);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressDestroyMetadata(
    JNIEnv *env, jclass, jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompDecompressDestroyMetadata(reinterpret_cast<void *>(metadata_ptr));
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetTempSize(
    JNIEnv *env, jclass, jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    std::size_t temp_size = 0;
    auto status = nvcompDecompressGetTempSize(reinterpret_cast<void *>(metadata_ptr), &temp_size);
    check_nvcomp_status(env, status);
    return temp_size;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetOutputSize(
    JNIEnv *env, jclass, jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    std::size_t out_size = 0;
    auto status = nvcompDecompressGetOutputSize(reinterpret_cast<void *>(metadata_ptr), &out_size);
    check_nvcomp_status(env, status);
    return out_size;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetType(JNIEnv *env, jclass,
                                                                              jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompType_t out_type;
    auto status = nvcompDecompressGetType(reinterpret_cast<void *>(metadata_ptr), &out_type);
    check_nvcomp_status(env, status);
    return static_cast<jint>(out_type);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressAsync(
    JNIEnv *env, jclass, jlong j_in_ptr, jlong j_in_size, jlong j_temp_ptr, jlong j_temp_size,
    jlong j_metadata_ptr, jlong j_out_ptr, jlong j_out_size, jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptr = reinterpret_cast<void const *>(j_in_ptr);
    auto in_size = static_cast<std::size_t>(j_in_size);
    auto temp_ptr = reinterpret_cast<void *>(j_temp_ptr);
    auto temp_size = static_cast<std::size_t>(j_temp_size);
    auto metadata_ptr = reinterpret_cast<void *>(j_metadata_ptr);
    auto out_ptr = reinterpret_cast<void *>(j_out_ptr);
    auto out_size = static_cast<std::size_t>(j_out_size);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompDecompressAsync(in_ptr, in_size, temp_ptr, temp_size, metadata_ptr, out_ptr,
                                        out_size, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_isLZ4Data(JNIEnv *env, jclass,
                                                                          jlong j_in_ptr,
                                                                          jlong j_in_size,
                                                                          jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptr = reinterpret_cast<void const *>(j_in_ptr);
    auto in_size = static_cast<std::size_t>(j_in_size);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    return LZ4IsData(in_ptr, in_size, stream);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_isLZ4Metadata(JNIEnv *env, jclass,
                                                                              jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    return LZ4IsMetadata(reinterpret_cast<void *>(metadata_ptr));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4CompressConfigure(
    JNIEnv *env, jclass, jint j_chunk_size, jlong j_uncompressed_size) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = static_cast<std::size_t>(j_chunk_size);
    auto uncompressed_size = static_cast<std::size_t>(j_uncompressed_size);
    std::size_t metadata_bytes = 0;
    std::size_t temp_bytes = 0;
    std::size_t out_bytes = 0;
    auto status = nvcompLZ4CompressConfigure(&opts, NVCOMP_TYPE_CHAR, uncompressed_size,
                                             &metadata_bytes, &temp_bytes, &out_bytes);
    check_nvcomp_status(env, status);
    cudf::jni::native_jlongArray result(env, 3);
    result[0] = static_cast<jlong>(metadata_bytes);
    result[1] = static_cast<jlong>(temp_bytes);
    result[2] = static_cast<jlong>(out_bytes);
    return result.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4CompressAsync(
    JNIEnv *env, jclass, jlong j_compressed_size_ptr, jlong j_in_ptr, jlong j_in_size,
    jint j_input_type, jlong j_chunk_size, jlong j_temp_ptr, jlong j_temp_size, jlong j_out_ptr,
    jlong j_out_size, jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptr = reinterpret_cast<void const *>(j_in_ptr);
    auto in_size = static_cast<std::size_t>(j_in_size);
    auto comp_type = static_cast<nvcompType_t>(j_input_type);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = static_cast<std::size_t>(j_chunk_size);
    auto temp_ptr = reinterpret_cast<void *>(j_temp_ptr);
    auto temp_size = static_cast<std::size_t>(j_temp_size);
    auto out_ptr = reinterpret_cast<void *>(j_out_ptr);
    auto compressed_size_ptr = reinterpret_cast<std::size_t *>(j_compressed_size_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompLZ4CompressAsync(&opts, comp_type, in_ptr, in_size, temp_ptr, temp_size,
                                         out_ptr, compressed_size_ptr, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4DecompressConfigure(
    JNIEnv *env, jclass, jlong j_input_ptr, jlong j_input_size, jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptr = reinterpret_cast<void const *>(j_input_ptr);
    auto compressed_bytes = static_cast<std::size_t>(j_input_size);
    void *metadata_ptr = nullptr;
    std::size_t metadata_bytes = 0;
    std::size_t temp_bytes = 0;
    std::size_t uncompressed_bytes = 0;
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status =
        nvcompLZ4DecompressConfigure(compressed_ptr, compressed_bytes, &metadata_ptr,
                                     &metadata_bytes, &temp_bytes, &uncompressed_bytes, stream);
    check_nvcomp_status(env, status);
    cudf::jni::native_jlongArray result(env, 4);
    result[0] = reinterpret_cast<jlong>(metadata_ptr);
    result[1] = static_cast<jlong>(metadata_bytes);
    result[2] = static_cast<jlong>(temp_bytes);
    result[3] = static_cast<jlong>(uncompressed_bytes);
    return result.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4DecompressAsync(
    JNIEnv *env, jclass, jlong j_in_ptr, jlong j_in_size, jlong j_metadata_ptr,
    jlong j_metadata_size, jlong j_temp_ptr, jlong j_temp_size, jlong j_out_ptr, jlong j_out_size,
    jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptr = reinterpret_cast<void const *>(j_in_ptr);
    auto compressed_bytes = static_cast<std::size_t>(j_in_size);
    auto metadata_ptr = reinterpret_cast<void const *>(j_metadata_ptr);
    auto metadata_bytes = static_cast<std::size_t>(j_metadata_size);
    auto temp_ptr = reinterpret_cast<void *>(j_temp_ptr);
    auto temp_bytes = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptr = reinterpret_cast<void *>(j_out_ptr);
    auto uncompressed_bytes = static_cast<std::size_t>(j_out_size);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompLZ4DecompressAsync(compressed_ptr, compressed_bytes, metadata_ptr,
                                           metadata_bytes, temp_ptr, temp_bytes, uncompressed_ptr,
                                           uncompressed_bytes, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4DestroyMetadata(JNIEnv *env, jclass,
                                                                               jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompLZ4DestroyMetadata(reinterpret_cast<void *>(metadata_ptr));
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetTempSize(
    JNIEnv *env, jclass, jlong j_batch_size, jlong j_max_chunk_size) {
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size = static_cast<std::size_t>(j_batch_size);
    auto max_chunk_size = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t temp_size = 0;
    auto status = nvcompBatchedLZ4CompressGetTempSize(batch_size, max_chunk_size,
                                                      nvcompBatchedLZ4DefaultOpts, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetMaxOutputChunkSize(
    JNIEnv *env, jclass, jlong j_max_chunk_size) {
  try {
    cudf::jni::auto_set_device(env);
    auto max_chunk_size = static_cast<std::size_t>(j_max_chunk_size);
    std::size_t max_output_size = 0;
    auto status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        max_chunk_size, nvcompBatchedLZ4DefaultOpts, &max_output_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(max_output_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressAsync(
    JNIEnv *env, jclass, jlong j_in_ptrs, jlong j_in_sizes, jlong j_chunk_size, jlong j_batch_size,
    jlong j_temp_ptr, jlong j_temp_size, jlong j_out_ptrs, jlong j_compressed_sizes_out_ptr,
    jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto in_ptrs = reinterpret_cast<void const *const *>(j_in_ptrs);
    auto in_sizes = reinterpret_cast<std::size_t const *>(j_in_sizes);
    auto chunk_size = static_cast<std::size_t>(j_chunk_size);
    auto batch_size = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr = reinterpret_cast<void *>(j_temp_ptr);
    auto temp_size = static_cast<std::size_t>(j_temp_size);
    auto out_ptrs = reinterpret_cast<void *const *>(j_out_ptrs);
    auto compressed_out_sizes = reinterpret_cast<std::size_t *>(j_compressed_sizes_out_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompBatchedLZ4CompressAsync(in_ptrs, in_sizes, chunk_size, batch_size, temp_ptr,
                                                temp_size, out_ptrs, compressed_out_sizes,
                                                nvcompBatchedLZ4DefaultOpts, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetTempSize(
    JNIEnv *env, jclass, jlong j_batch_size, jlong j_chunk_size) {
  try {
    cudf::jni::auto_set_device(env);
    auto batch_size = static_cast<std::size_t>(j_batch_size);
    auto chunk_size = static_cast<std::size_t>(j_chunk_size);
    std::size_t temp_size = 0;
    auto status = nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressAsync(
    JNIEnv *env, jclass, jlong j_in_ptrs, jlong j_in_sizes, jlong j_out_sizes, jlong j_batch_size,
    jlong j_temp_ptr, jlong j_temp_size, jlong j_out_ptrs, jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs = reinterpret_cast<void const *const *>(j_in_ptrs);
    auto compressed_sizes = reinterpret_cast<std::size_t const *>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t const *>(j_out_sizes);
    auto batch_size = static_cast<std::size_t>(j_batch_size);
    auto temp_ptr = reinterpret_cast<void *>(j_temp_ptr);
    auto temp_size = static_cast<std::size_t>(j_temp_size);
    auto uncompressed_ptrs = reinterpret_cast<void *const *>(j_out_ptrs);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto uncompressed_statuses = rmm::device_uvector<nvcompStatus_t>(batch_size, stream);
    auto actual_uncompressed_sizes = rmm::device_uvector<std::size_t>(batch_size, stream);
    auto status = nvcompBatchedLZ4DecompressAsync(
        compressed_ptrs, compressed_sizes, uncompressed_sizes, actual_uncompressed_sizes.data(),
        batch_size, temp_ptr, temp_size, uncompressed_ptrs, uncompressed_statuses.data(), stream);
    check_nvcomp_status(env, status);
    if (!cudf::java::check_nvcomp_output_sizes(uncompressed_sizes, actual_uncompressed_sizes.data(),
                                               batch_size, stream)) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS,
                                      "nvcomp decompress output size mismatch");
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4GetDecompressSizeAsync(
    JNIEnv *env, jclass, jlong j_in_ptrs, jlong j_in_sizes, jlong j_out_sizes, jlong j_batch_size,
    jlong j_stream) {
  try {
    cudf::jni::auto_set_device(env);
    auto compressed_ptrs = reinterpret_cast<void const *const *>(j_in_ptrs);
    auto compressed_sizes = reinterpret_cast<std::size_t const *>(j_in_sizes);
    auto uncompressed_sizes = reinterpret_cast<std::size_t *>(j_out_sizes);
    auto batch_size = static_cast<std::size_t>(j_batch_size);
    auto stream = reinterpret_cast<cudaStream_t>(j_stream);
    auto status = nvcompBatchedLZ4GetDecompressSizeAsync(compressed_ptrs, compressed_sizes,
                                                         uncompressed_sizes, batch_size, stream);
    check_nvcomp_status(env, status);
  }
  CATCH_STD(env, );
}

} // extern "C"
