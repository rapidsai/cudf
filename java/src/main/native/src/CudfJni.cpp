/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <memory>

#include "jni_utils.hpp"

namespace cudf {
namespace jni {

static const jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

static jclass contiguous_table_jclass;
static jmethodID from_contiguous_column_views;

#define CONTIGUOUS_TABLE_CLASS "ai/rapids/cudf/ContiguousTable"
#define CONTIGUOUS_TABLE_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLE_CLASS ";"

static bool cache_contiguous_table_jni(JNIEnv *env) {
  jclass cls = env->FindClass(CONTIGUOUS_TABLE_CLASS);
  if (cls == nullptr) {
    return false;
  }

  from_contiguous_column_views = env->GetStaticMethodID(cls, 
          "fromContiguousColumnViews", CONTIGUOUS_TABLE_FACTORY_SIG("[JJJJ"));
  if (from_contiguous_column_views == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  contiguous_table_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (contiguous_table_jclass == nullptr) {
    return false;
  }
  return true;
}

static void release_contiguous_table_jni(JNIEnv *env) {
  if (contiguous_table_jclass != nullptr) {
    env->DeleteGlobalRef(contiguous_table_jclass);
    contiguous_table_jclass = nullptr;
  }
}

jobject contiguous_table_from(JNIEnv* env, cudf::experimental::contiguous_split_result & split) {
  jlong address = reinterpret_cast<jlong>(split.all_data->data());
  jlong size = static_cast<jlong>(split.all_data->size());
  jlong buff_address = reinterpret_cast<jlong>(split.all_data.get());
  int num_columns = split.table.num_columns();
  cudf::jni::native_jlongArray views(env, num_columns);
  for (int i = 0; i < num_columns; i++) {
    //TODO Exception handling is not ideal, if no exceptions are thrown ownership of the new cv
    // is passed to java. If an exception is thrown we need to free it, but this needs to be
    // coordinated with the java side because one column may have changed ownership while
    // another may not have. We don't want to double free the view so for now we just let it
    // leak because it should be a small amount of host memory.
    //
    // In the ideal case we would keep the view where it is at, and pass in a pointer to it
    // That pointer would then be copied when java takes ownership of it, but that adds an
    // extra JNI call that I would like to avoid for performance reasons.
    cudf::column_view * cv = new cudf::column_view(split.table.column(i));
    views[i] = reinterpret_cast<jlong>(cv);
  }

  views.commit();
  jobject ret = env->CallStaticObjectMethod(contiguous_table_jclass, from_contiguous_column_views,
          views.get_jArray(),
          address, size, buff_address);

  if (ret != nullptr) {
    split.all_data.release();
  }
  return ret;
}

native_jobjectArray<jobject> contiguous_table_array(JNIEnv* env, jsize length) {
  return native_jobjectArray<jobject>(env, env->NewObjectArray(length, contiguous_table_jclass, nullptr));
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // cache any class objects and method IDs here
  if (!cudf::jni::cache_contiguous_table_jni(env)) {
    return JNI_ERR;
  }

  return cudf::jni::MINIMUM_JNI_VERSION;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return;
  }

  // release cached class objects here.
  cudf::jni::release_contiguous_table_jni(env);
}

} // extern "C"
