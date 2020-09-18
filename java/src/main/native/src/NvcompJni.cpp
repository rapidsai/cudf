/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cascaded.h>
#include <lz4.h>
#include <nvcomp.h>

#include "cudf_jni_apis.hpp"

namespace {

constexpr char const *NVCOMP_ERROR_CLASS = "ai/rapids/cudf/nvcomp/NvcompException";
constexpr char const *NVCOMP_CUDA_ERROR_CLASS = "ai/rapids/cudf/nvcomp/NvcompCudaException";
constexpr char const *ILLEGAL_ARG_CLASS = "java/lang/IllegalArgumentException";
constexpr char const *UNSUPPORTED_CLASS = "java/lang/UnsupportedOperationException";

void check_nvcomp_status(JNIEnv *env, nvcompError_t status) {
  switch (status) {
    case nvcompSuccess:
      break;
    case nvcompErrorInvalidValue:
      cudf::jni::throw_java_exception(env, ILLEGAL_ARG_CLASS, "nvcomp invalid value");
      break;
    case nvcompErrorNotSupported:
      cudf::jni::throw_java_exception(env, UNSUPPORTED_CLASS, "nvcomp unsupported");
      break;
    case nvcompErrorCudaError:
      cudf::jni::throw_java_exception(env, NVCOMP_CUDA_ERROR_CLASS, "nvcomp CUDA error");
      break;
    default:
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "nvcomp unknown error");
      break;
  }
}

} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetMetadata(JNIEnv *env, jclass,
                                                           jlong in_ptr, jlong in_size,
                                                           jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    void *metadata_ptr;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto status = nvcompDecompressGetMetadata(reinterpret_cast<void *>(in_ptr), in_size,
                                              &metadata_ptr, stream);
    check_nvcomp_status(env, status);
    return reinterpret_cast<jlong>(metadata_ptr);
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressDestroyMetadata(JNIEnv *env, jclass,
                                                               jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompDecompressDestroyMetadata(reinterpret_cast<void *>(metadata_ptr));
  } CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetTempSize(JNIEnv *env, jclass,
                                                           jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    size_t temp_size;
    auto status = nvcompDecompressGetTempSize(reinterpret_cast<void *>(metadata_ptr), &temp_size);
    check_nvcomp_status(env, status);
    return temp_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressGetOutputSize(JNIEnv *env, jclass,
                                                             jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    size_t out_size;
    auto status = nvcompDecompressGetOutputSize(reinterpret_cast<void *>(metadata_ptr), &out_size);
    check_nvcomp_status(env, status);
    return out_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_decompressAsync(JNIEnv *env, jclass,
                                                     jlong in_ptr, jlong in_size,
                                                     jlong temp_ptr, jlong temp_size,
                                                     jlong metadata_ptr,
                                                     jlong out_ptr, jlong out_size, jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto status = nvcompDecompressAsync(reinterpret_cast<void *>(in_ptr), in_size,
                                        reinterpret_cast<void *>(temp_ptr), temp_size,
                                        reinterpret_cast<void *>(metadata_ptr),
                                        reinterpret_cast<void *>(out_ptr), out_size,
                                        stream);
    check_nvcomp_status(env, status);
  } CATCH_STD(env, );
}

JNIEXPORT jboolean JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_isLZ4Data(JNIEnv *env, jclass, jlong in_ptr, jlong in_size) {
  try {
    cudf::jni::auto_set_device(env);
    return LZ4IsData(reinterpret_cast<void *>(in_ptr), in_size);
  } CATCH_STD(env, 0)
}

JNIEXPORT jboolean JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_isLZ4Metadata(JNIEnv *env, jclass, jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    return LZ4IsMetadata(reinterpret_cast<void *>(metadata_ptr));
  } CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4CompressGetTempSize(JNIEnv *env, jclass,
                                                            jlong in_ptr, jlong in_size,
                                                            jint input_type, jlong chunk_size) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    size_t temp_size;
    auto status = nvcompLZ4CompressGetTempSize(reinterpret_cast<void *>(in_ptr), in_size,
                                               comp_type, &opts, &temp_size);
    check_nvcomp_status(env, status);
    return temp_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4CompressGetOutputSize(JNIEnv *env, jclass,
                                                              jlong in_ptr, jlong in_size,
                                                              jint input_type, jlong chunk_size,
                                                              jlong temp_ptr, jlong temp_size,
                                                              jboolean compute_exact) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    size_t out_size;
    auto status = nvcompLZ4CompressGetOutputSize(reinterpret_cast<void *>(in_ptr), in_size,
                                                 comp_type, &opts,
                                                 reinterpret_cast<void *>(temp_ptr), temp_size,
                                                 &out_size, compute_exact);
    check_nvcomp_status(env, status);
    return out_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4Compress(JNIEnv *env, jclass,
                                                 jlong in_ptr, jlong in_size,
                                                 jint input_type, jlong chunk_size,
                                                 jlong temp_ptr, jlong temp_size,
                                                 jlong out_ptr, jlong out_size,
                                                 jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    size_t compressed_size = out_size;
    auto status = nvcompLZ4CompressAsync(reinterpret_cast<void *>(in_ptr), in_size,
                                         comp_type, &opts,
                                         reinterpret_cast<void *>(temp_ptr), temp_size,
                                         reinterpret_cast<void *>(out_ptr), &compressed_size,
                                         stream);
    check_nvcomp_status(env, status);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      JNI_THROW_NEW(env, NVCOMP_CUDA_ERROR_CLASS, "Error synchronizing stream", 0);
    }
    return compressed_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_lz4CompressAsync(JNIEnv *env, jclass,
                                                      jlong compressed_output_ptr,
                                                      jlong in_ptr, jlong in_size,
                                                      jint input_type, jlong chunk_size,
                                                      jlong temp_ptr, jlong temp_size,
                                                      jlong out_ptr, jlong out_size,
                                                      jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto compressed_size_ptr = reinterpret_cast<size_t *>(compressed_output_ptr);
    *compressed_size_ptr = out_size;
    auto status = nvcompLZ4CompressAsync(reinterpret_cast<void *>(in_ptr), in_size,
                                         comp_type, &opts,
                                         reinterpret_cast<void *>(temp_ptr), temp_size,
                                         reinterpret_cast<void *>(out_ptr), compressed_size_ptr,
                                         stream);
    check_nvcomp_status(env, status);
  } CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetMetadata(JNIEnv* env, jclass,
                                                                     jlongArray in_ptrs,
                                                                     jlongArray in_sizes,
                                                                     jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jpointerArray<void const> input_ptrs(env, in_ptrs);
    cudf::jni::native_jlongArray input_jsizes(env, in_sizes);
    if (input_ptrs.size() != input_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input array size mismatch");
    }
    std::vector<size_t> sizes;
    std::transform(input_jsizes.data(), input_jsizes.data() + input_jsizes.size(),
                   std::back_inserter(sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    void* metadata_ptr = nullptr;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto status = nvcompBatchedLZ4DecompressGetMetadata(input_ptrs.data(), sizes.data(),
                                                        input_ptrs.size(), &metadata_ptr, stream);
    check_nvcomp_status(env, status);
    return reinterpret_cast<jlong>(metadata_ptr);
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressDestroyMetadata(JNIEnv* env, jclass,
                                                                         jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    nvcompBatchedLZ4DecompressDestroyMetadata(reinterpret_cast<void*>(metadata_ptr));
  } CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetTempSize(JNIEnv* env, jclass,
                                                                     jlong metadata_ptr) {
  try {
    cudf::jni::auto_set_device(env);
    size_t temp_size;
    auto status = nvcompBatchedLZ4DecompressGetTempSize(reinterpret_cast<void*>(metadata_ptr),
                                                        &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  } CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressGetOutputSize(JNIEnv* env, jclass,
                                                                       jlong metadata_ptr,
                                                                       jint num_outputs) {
  try {
    cudf::jni::auto_set_device(env);
    std::vector<size_t> sizes(num_outputs);
    auto status = nvcompBatchedLZ4DecompressGetOutputSize(reinterpret_cast<void*>(metadata_ptr),
                                                          num_outputs,
                                                          sizes.data());
    check_nvcomp_status(env, status);
    cudf::jni::native_jlongArray jsizes(env, num_outputs);
    std::transform(sizes.begin(), sizes.end(), jsizes.data(),
                   [](size_t x) -> jlong { return static_cast<jlong>(x); });
    return jsizes.get_jArray();
  } CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4DecompressAsync(JNIEnv* env, jclass,
                                                               jlongArray in_ptrs,
                                                               jlongArray in_sizes,
                                                               jlong temp_ptr,
                                                               jlong temp_size,
                                                               jlong metadata_ptr,
                                                               jlongArray out_ptrs,
                                                               jlongArray out_sizes,
                                                               jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<void const> input_ptrs(env, in_ptrs);
    cudf::jni::native_jlongArray input_jsizes(env, in_sizes);
    if (input_ptrs.size() != input_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input array size mismatch");
    }
    std::vector<size_t> input_sizes;
    std::transform(input_jsizes.data(), input_jsizes.data() + input_jsizes.size(),
                   std::back_inserter(input_sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    cudf::jni::native_jpointerArray<void> output_ptrs(env, out_ptrs);
    cudf::jni::native_jlongArray output_jsizes(env, out_sizes);
    if (output_ptrs.size() != output_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "output array size mismatch");
    }
    if (input_ptrs.size() != output_ptrs.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input/output array size mismatch");
    }
    std::vector<size_t> output_sizes;
    std::transform(output_jsizes.data(), output_jsizes.data() + output_jsizes.size(),
                   std::back_inserter(output_sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto status = nvcompBatchedLZ4DecompressAsync(input_ptrs.data(), input_sizes.data(),
                                                  input_ptrs.size(),
                                                  reinterpret_cast<void*>(temp_ptr),
                                                  static_cast<size_t>(temp_size),
                                                  reinterpret_cast<void*>(metadata_ptr),
                                                  output_ptrs.data(),
                                                  output_sizes.data(),
                                                  stream);
    check_nvcomp_status(env, status);
  } CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetTempSize(JNIEnv* env, jclass,
                                                                   jlongArray in_ptrs,
                                                                   jlongArray in_sizes,
                                                                   jlong chunk_size) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<void const> input_ptrs(env, in_ptrs);
    cudf::jni::native_jlongArray input_jsizes(env, in_sizes);
    if (input_ptrs.size() != input_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input array size mismatch");
    }
    std::vector<size_t> sizes;
    std::transform(input_jsizes.data(), input_jsizes.data() + input_jsizes.size(),
                   std::back_inserter(sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    size_t temp_size = 0;
    auto status = nvcompBatchedLZ4CompressGetTempSize(input_ptrs.data(), sizes.data(),
                                                      input_ptrs.size(), &opts, &temp_size);
    check_nvcomp_status(env, status);
    return static_cast<jlong>(temp_size);
  } CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressGetOutputSize(JNIEnv* env, jclass,
                                                                     jlongArray in_ptrs,
                                                                     jlongArray in_sizes,
                                                                     jlong chunk_size,
                                                                     jlong temp_ptr,
                                                                     jlong temp_size) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<void const> input_ptrs(env, in_ptrs);
    cudf::jni::native_jlongArray input_jsizes(env, in_sizes);
    if (input_ptrs.size() != input_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input array size mismatch");
    }
    std::vector<size_t> input_sizes;
    std::transform(input_jsizes.data(), input_jsizes.data() + input_jsizes.size(),
                   std::back_inserter(input_sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    std::vector<size_t> output_sizes(input_ptrs.size());
    auto status = nvcompBatchedLZ4CompressGetOutputSize(input_ptrs.data(), input_sizes.data(),
                                                        input_ptrs.size(), &opts,
                                                        reinterpret_cast<void*>(temp_ptr),
                                                        static_cast<size_t>(temp_size),
                                                        output_sizes.data());
    check_nvcomp_status(env, status);
    cudf::jni::native_jlongArray jsizes(env, input_ptrs.size());
    std::transform(output_sizes.begin(), output_sizes.end(), jsizes.data(),
                   [](size_t x) -> jlong { return static_cast<jlong>(x); });
    return jsizes.get_jArray();
  } CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_batchedLZ4CompressAsync(JNIEnv* env, jclass,
                                                             jlong compressed_sizes_out_ptr,
                                                             jlongArray in_ptrs,
                                                             jlongArray in_sizes,
                                                             jlong chunk_size,
                                                             jlong temp_ptr,
                                                             jlong temp_size,
                                                             jlongArray out_ptrs,
                                                             jlongArray out_sizes,
                                                             jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<void const> input_ptrs(env, in_ptrs);
    cudf::jni::native_jlongArray input_jsizes(env, in_sizes);
    if (input_ptrs.size() != input_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input array size mismatch");
    }
    std::vector<size_t> input_sizes;
    std::transform(input_jsizes.data(), input_jsizes.data() + input_jsizes.size(),
                   std::back_inserter(input_sizes),
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    cudf::jni::native_jpointerArray<void> output_ptrs(env, out_ptrs);
    cudf::jni::native_jlongArray output_jsizes(env, out_sizes);
    if (output_ptrs.size() != output_jsizes.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "output array size mismatch");
    }
    if (input_ptrs.size() != output_ptrs.size()) {
      cudf::jni::throw_java_exception(env, NVCOMP_ERROR_CLASS, "input/output array size mismatch");
    }

    auto output_sizes = reinterpret_cast<size_t*>(compressed_sizes_out_ptr);
    std::transform(output_jsizes.data(), output_jsizes.data() + output_jsizes.size(),
                   output_sizes,
                   [](jlong x) -> size_t { return static_cast<size_t>(x); });

    nvcompLZ4FormatOpts opts{};
    opts.chunk_size = chunk_size;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto status = nvcompBatchedLZ4CompressAsync(input_ptrs.data(), input_sizes.data(),
                                                input_ptrs.size(), &opts,
                                                reinterpret_cast<void*>(temp_ptr),
                                                static_cast<size_t>(temp_size),
                                                output_ptrs.data(),
                                                output_sizes,  // input/output parameter
                                                stream);
    check_nvcomp_status(env, status);
  } CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_cascadedCompressGetTempSize(JNIEnv *env, jclass,
                                                                 jlong in_ptr, jlong in_size,
                                                                 jint input_type, jint num_rles,
                                                                 jint num_deltas, jboolean use_bp) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompCascadedFormatOpts opts{};
    opts.num_RLEs = num_rles;
    opts.num_deltas = num_deltas;
    opts.use_bp = use_bp;
    size_t temp_size;
    auto status = nvcompCascadedCompressGetTempSize(reinterpret_cast<void *>(in_ptr), in_size,
                                                    comp_type, &opts, &temp_size);
    check_nvcomp_status(env, status);
    return temp_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_cascadedCompressGetOutputSize(JNIEnv *env, jclass,
                                                                   jlong in_ptr, jlong in_size,
                                                                   jint input_type, jint num_rles,
                                                                   jint num_deltas, jboolean use_bp,
                                                                   jlong temp_ptr, jlong temp_size,
                                                                   jboolean compute_exact) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompCascadedFormatOpts opts{};
    opts.num_RLEs = num_rles;
    opts.num_deltas = num_deltas;
    opts.use_bp = use_bp;
    size_t out_size;
    auto status = nvcompCascadedCompressGetOutputSize(reinterpret_cast<void *>(in_ptr), in_size,
                                                      comp_type, &opts,
                                                      reinterpret_cast<void *>(temp_ptr), temp_size,
                                                      &out_size, compute_exact);
    check_nvcomp_status(env, status);
    return out_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_cascadedCompress(JNIEnv *env, jclass,
                                                      jlong in_ptr, jlong in_size,
                                                      jint input_type, jint num_rles,
                                                      jint num_deltas, jboolean use_bp,
                                                      jlong temp_ptr, jlong temp_size,
                                                      jlong out_ptr, jlong out_size,
                                                      jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompCascadedFormatOpts opts{};
    opts.num_RLEs = num_rles;
    opts.num_deltas = num_deltas;
    opts.use_bp = use_bp;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    size_t compressed_size = out_size;
    auto status = nvcompCascadedCompressAsync(reinterpret_cast<void *>(in_ptr), in_size,
                                              comp_type, &opts,
                                              reinterpret_cast<void *>(temp_ptr), temp_size,
                                              reinterpret_cast<void *>(out_ptr), &compressed_size,
                                              stream);
    check_nvcomp_status(env, status);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      JNI_THROW_NEW(env, NVCOMP_CUDA_ERROR_CLASS, "Error synchronizing stream", 0);
    }
    return compressed_size;
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_nvcomp_NvcompJni_cascadedCompressAsync(JNIEnv *env, jclass,
                                                           jlong compressed_output_ptr,
                                                           jlong in_ptr, jlong in_size,
                                                           jint input_type, jint num_rles,
                                                           jint num_deltas, jboolean use_bp,
                                                           jlong temp_ptr, jlong temp_size,
                                                           jlong out_ptr, jlong out_size,
                                                           jlong jstream) {
  try {
    cudf::jni::auto_set_device(env);
    auto comp_type = static_cast<nvcompType_t>(input_type);
    nvcompCascadedFormatOpts opts{};
    opts.num_RLEs = num_rles;
    opts.num_deltas = num_deltas;
    opts.use_bp = use_bp;
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto compressed_size_ptr = reinterpret_cast<size_t *>(compressed_output_ptr);
    *compressed_size_ptr = out_size;
    auto status = nvcompCascadedCompressAsync(reinterpret_cast<void *>(in_ptr), in_size,
                                              comp_type, &opts,
                                              reinterpret_cast<void *>(temp_ptr), temp_size,
                                              reinterpret_cast<void *>(out_ptr),
                                              compressed_size_ptr, stream);
    check_nvcomp_status(env, status);
  } CATCH_STD(env, );
}

} // extern "C"
