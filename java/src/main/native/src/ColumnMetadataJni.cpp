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

#include <cudf/interop.hpp>

#include "jni_utils.hpp"

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnMetadata_close(JNIEnv *env,
                                                                jclass,
                                                                jlong j_handle) {
  JNI_NULL_CHECK(env, j_handle, "column metadata handle is null", );
  try {
    auto to_del = reinterpret_cast<cudf::column_metadata *>(j_handle);
    delete to_del;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnMetadata_create(JNIEnv *env,
                                                                  jclass,
                                                                  jstring j_name,
                                                                  jlongArray j_children) {
  try {
    // No need to set device since no GPU ops here.
    cudf::jni::native_jstring col_name(env, j_name);
    cudf::jni::native_jlongArray meta_children(env, j_children);
    // Create a meta with empty name if `col_name` is NULL.
    auto name = std::string(col_name.is_null() ? "" : col_name.get());
    cudf::column_metadata *cm = new cudf::column_metadata(name);
    if (!meta_children.is_null()) {
      // add the children
      for (int i = 0; i < meta_children.size(); i++) {
        cudf::column_metadata *child = reinterpret_cast<cudf::column_metadata *>(meta_children[i]);
        // copy to `this`.
        cm->children_meta.push_back(*child);
      }
    }
    return reinterpret_cast<jlong>(cm);
  }
  CATCH_STD(env, 0);
}

} // extern "C"
