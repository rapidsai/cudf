/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_createMetadataDirectBuffer(
  JNIEnv* env, jclass, jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", nullptr);
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t>*>(j_metadata_ptr);
    return env->NewDirectByteBuffer(const_cast<uint8_t*>(metadata->data()), metadata->size());
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_closeMetadata(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", );
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t>*>(j_metadata_ptr);
    delete metadata;
  }
  CATCH_STD(env, );
}

}  // extern "C"
