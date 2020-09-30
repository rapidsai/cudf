/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <sstream>

#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

#include "jni_utils.hpp"

namespace {

// handles detaching a thread from the JVM when the thread terminates
class jvm_detach_on_destruct {
public:
  explicit jvm_detach_on_destruct(JavaVM *jvm) : jvm{jvm} {}

  ~jvm_detach_on_destruct() { jvm->DetachCurrentThread(); }

private:
  JavaVM *jvm;
};

} // anonymous namespace

namespace cudf {
namespace jni {

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
constexpr bool is_ptds_enabled{true};
#else
constexpr bool is_ptds_enabled{false};
#endif

static jclass Contiguous_table_jclass;
static jmethodID From_contiguous_column_views;

#define CONTIGUOUS_TABLE_CLASS "ai/rapids/cudf/ContiguousTable"
#define CONTIGUOUS_TABLE_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLE_CLASS ";"

static bool cache_contiguous_table_jni(JNIEnv *env) {
  jclass cls = env->FindClass(CONTIGUOUS_TABLE_CLASS);
  if (cls == nullptr) {
    return false;
  }

  From_contiguous_column_views = env->GetStaticMethodID(cls, "fromContiguousColumnViews",
                                                        CONTIGUOUS_TABLE_FACTORY_SIG("[JJJJ"));
  if (From_contiguous_column_views == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  Contiguous_table_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Contiguous_table_jclass == nullptr) {
    return false;
  }
  return true;
}

static void release_contiguous_table_jni(JNIEnv *env) {
  if (Contiguous_table_jclass != nullptr) {
    env->DeleteGlobalRef(Contiguous_table_jclass);
    Contiguous_table_jclass = nullptr;
  }
}

jobject contiguous_table_from(JNIEnv *env, cudf::contiguous_split_result &split) {
  jlong address = reinterpret_cast<jlong>(split.all_data->data());
  jlong size = static_cast<jlong>(split.all_data->size());
  jlong buff_address = reinterpret_cast<jlong>(split.all_data.get());
  int num_columns = split.table.num_columns();
  cudf::jni::native_jlongArray views(env, num_columns);
  for (int i = 0; i < num_columns; i++) {
    // TODO Exception handling is not ideal, if no exceptions are thrown ownership of the new cv
    // is passed to java. If an exception is thrown we need to free it, but this needs to be
    // coordinated with the java side because one column may have changed ownership while
    // another may not have. We don't want to double free the view so for now we just let it
    // leak because it should be a small amount of host memory.
    //
    // In the ideal case we would keep the view where it is at, and pass in a pointer to it
    // That pointer would then be copied when java takes ownership of it, but that adds an
    // extra JNI call that I would like to avoid for performance reasons.
    cudf::column_view *cv = new cudf::column_view(split.table.column(i));
    views[i] = reinterpret_cast<jlong>(cv);
  }

  views.commit();
  jobject ret = env->CallStaticObjectMethod(Contiguous_table_jclass, From_contiguous_column_views,
                                            views.get_jArray(), address, size, buff_address);

  if (ret != nullptr) {
    split.all_data.release();
  }
  return ret;
}

native_jobjectArray<jobject> contiguous_table_array(JNIEnv *env, jsize length) {
  return native_jobjectArray<jobject>(
      env, env->NewObjectArray(length, Contiguous_table_jclass, nullptr));
}

static jclass Host_memory_buffer_jclass;
static jmethodID Host_buffer_allocate;
static jfieldID Host_buffer_address;
static jfieldID Host_buffer_length;

#define HOST_MEMORY_BUFFER_CLASS "ai/rapids/cudf/HostMemoryBuffer"
#define HOST_MEMORY_BUFFER_SIG(param_sig) "(" param_sig ")L" HOST_MEMORY_BUFFER_CLASS ";"

static bool cache_host_memory_buffer_jni(JNIEnv *env) {
  jclass cls = env->FindClass(HOST_MEMORY_BUFFER_CLASS);
  if (cls == nullptr) {
    return false;
  }

  Host_buffer_allocate = env->GetStaticMethodID(cls, "allocate", HOST_MEMORY_BUFFER_SIG("JZ"));
  if (Host_buffer_allocate == nullptr) {
    return false;
  }

  Host_buffer_address = env->GetFieldID(cls, "address", "J");
  if (Host_buffer_address == nullptr) {
    return false;
  }

  Host_buffer_length = env->GetFieldID(cls, "length", "J");
  if (Host_buffer_length == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  Host_memory_buffer_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Host_memory_buffer_jclass == nullptr) {
    return false;
  }
  return true;
}

static void release_host_memory_buffer_jni(JNIEnv *env) {
  if (Host_memory_buffer_jclass != nullptr) {
    env->DeleteGlobalRef(Host_memory_buffer_jclass);
    Host_memory_buffer_jclass = nullptr;
  }
}

jobject allocate_host_buffer(JNIEnv *env, jlong amount, jboolean prefer_pinned) {
  jobject ret = env->CallStaticObjectMethod(Host_memory_buffer_jclass, Host_buffer_allocate, amount,
                                            prefer_pinned);

  if (env->ExceptionCheck()) {
    throw std::runtime_error("allocateHostBuffer threw an exception");
  }
  return ret;
}

jlong get_host_buffer_address(JNIEnv *env, jobject buffer) {
  return env->GetLongField(buffer, Host_buffer_address);
}

jlong get_host_buffer_length(JNIEnv *env, jobject buffer) {
  return env->GetLongField(buffer, Host_buffer_length);
}

// Get the JNI environment, attaching the current thread to the JVM if necessary. If the thread
// needs to be attached, the thread will automatically detach when the thread terminates.
JNIEnv *get_jni_env(JavaVM *jvm) {
  JNIEnv *env = nullptr;
  jint rc = jvm->GetEnv(reinterpret_cast<void **>(&env), MINIMUM_JNI_VERSION);
  if (rc == JNI_OK) {
    return env;
  }
  if (rc == JNI_EDETACHED) {
    JavaVMAttachArgs attach_args;
    attach_args.version = MINIMUM_JNI_VERSION;
    attach_args.name = const_cast<char *>("cudf thread");
    attach_args.group = NULL;

    if (jvm->AttachCurrentThreadAsDaemon(reinterpret_cast<void **>(&env), &attach_args) == JNI_OK) {
      // use thread_local object to detach the thread from the JVM when thread terminates.
      thread_local jvm_detach_on_destruct detacher(jvm);
    } else {
      throw std::runtime_error("unable to attach to JVM");
    }

    return env;
  }

  throw std::runtime_error("error detecting thread attach state with JVM");
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // make sure libcudf and the JNI library are built with the same PTDS mode
  if (cudf::is_ptds_enabled() != cudf::jni::is_ptds_enabled) {
    std::ostringstream ss;
    ss << "Libcudf is_ptds_enabled=" << cudf::is_ptds_enabled()
       << ", which does not match cudf jni is_ptds_enabled=" << cudf::jni::is_ptds_enabled
       << ". They need to be built with the same per-thread default stream flag.";
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), ss.str().c_str());
    return JNI_ERR;
  }

  // cache any class objects and method IDs here
  if (!cudf::jni::cache_contiguous_table_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                    "Unable to locate contiguous table methods needed by JNI");
    }
    return JNI_ERR;
  }

  if (!cudf::jni::cache_host_memory_buffer_jni(env)) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                    "Unable to locate host memory buffer methods needed by JNI");
    }
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
  cudf::jni::release_host_memory_buffer_jni(env);
}

} // extern "C"
