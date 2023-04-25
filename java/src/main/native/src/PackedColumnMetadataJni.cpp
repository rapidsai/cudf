/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"

namespace {

#define PACKED_COLUMN_META_CLASS "ai/rapids/cudf/PackedColumnMetadata"
#define PACKED_COLUMN_META_FACTORY_SIG(param_sig) "(" param_sig ")L" PACKED_COLUMN_META_CLASS ";"

jclass Packed_columns_meta_jclass;
jmethodID From_packed_column_meta_method;

} // anonymous namespace

namespace cudf {
namespace jni {

bool cache_packed_column_meta_jni(JNIEnv *env) {
  jclass cls = env->FindClass(PACKED_COLUMN_META_CLASS);
  if (cls == nullptr) {
    return false;
  }

  From_packed_column_meta_method =
      env->GetStaticMethodID(cls, "fromPackedColumnMeta", PACKED_COLUMN_META_FACTORY_SIG("J"));
  if (From_packed_column_meta_method == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  Packed_columns_meta_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Packed_columns_meta_jclass == nullptr) {
    return false;
  }
  return true;
}

jobject packed_column_metadata_from(JNIEnv *env, std::unique_ptr<std::vector<uint8_t>> meta) {
  jlong metadata_address = reinterpret_cast<jlong>(meta.release());
  return env->CallStaticObjectMethod(Packed_columns_meta_jclass, From_packed_column_meta_method,
                                     metadata_address);
}

void release_packed_column_meta_jni(JNIEnv *env) {
  if (Packed_columns_meta_jclass != nullptr) {
    env->DeleteGlobalRef(Packed_columns_meta_jclass);
    Packed_columns_meta_jclass = nullptr;
  }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_createMetadataDirectBuffer(
    JNIEnv *env, jclass, jlong j_metadata_ptr) {
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", nullptr);
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t> *>(j_metadata_ptr);
    return env->NewDirectByteBuffer(const_cast<uint8_t *>(metadata->data()), metadata->size());
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_PackedColumnMetadata_closeMetadata(JNIEnv *env, jclass, jlong j_metadata_ptr) {
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", );
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t> *>(j_metadata_ptr);
    delete metadata;
  }
  CATCH_STD(env, );
}

} // extern "C"
