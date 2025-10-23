/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "jni_utils.hpp"

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column.hpp>

#include <jni.h>

#include <mutex>
#include <sstream>

using cudf::jni::ptr_as_jlong;
using cudf::jni::release_as_jlong;

namespace cudf {
namespace jni {
// cache the ColumnVector things
#define COLUMN_VECTOR_CLASS "ai/rapids/cudf/ColumnVector"

static std::mutex column_vector_cache_lock;
static bool is_column_vector_class_cached = false;
static jclass ColumnVector_jclass;
static jmethodID ColumnVector_getViewHandle_method;
static jmethodID ColumnVector_close_method;

// Cache the class and its methods for better performance.
// (This can not be called inside the "JNI_OnLoad" method, because loading a
//  ColumnVector will trigger the JNI loading recursively, leading to a hang.)
void cache_column_vector_jni(JNIEnv* env)
{
  // A quick check without lock for better perf.
  if (is_column_vector_class_cached) { return; }

  // Multiple threads may call into this, so try to ask for the lock first.
  std::lock_guard lock(column_vector_cache_lock);
  // need to check again to avoid duplicate cache
  if (is_column_vector_class_cached) { return; }

  jclass cls = env->FindClass(COLUMN_VECTOR_CLASS);
  if (cls == nullptr) { throw jni_exception("Failed to find the Java column class"); }

  ColumnVector_getViewHandle_method = env->GetMethodID(cls, "getNativeView", "()J");
  if (ColumnVector_getViewHandle_method == nullptr) {
    throw jni_exception("Failed to find the 'getNativeView' method in the Java column class");
  }

  ColumnVector_close_method = env->GetMethodID(cls, "close", "()V");
  if (ColumnVector_close_method == nullptr) {
    throw jni_exception("Failed to find the 'close' method in the Java column class");
  }

  // Convert local reference to global so it cannot be garbage collected.
  ColumnVector_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (ColumnVector_jclass == nullptr) {
    throw jni_exception("Failed to cache the Java column class");
  }
  is_column_vector_class_cached = true;
}

// This is called inside the "JNI_OnUnload", so no lock is needed.
void release_cloumn_vector_jni(JNIEnv* env)
{
  if (is_column_vector_class_cached) {
    ColumnVector_jclass           = del_global_ref(env, ColumnVector_jclass);
    is_column_vector_class_cached = false;
  }
}

static cudf::column_view* column_vector_get_view(JNIEnv* env, jobject j_column_vector)
{
  cache_column_vector_jni(env);
  jlong ret = env->CallLongMethod(j_column_vector, ColumnVector_getViewHandle_method);
  if (env->ExceptionCheck() || ret == 0L) {
    throw std::runtime_error("failed to get the view handle from the Java column");
  }
  return reinterpret_cast<cudf::column_view*>(ret);
}

static void column_vector_close(JNIEnv* env, jobject j_column_vector)
{
  cache_column_vector_jni(env);
  env->CallVoidMethod(j_column_vector, ColumnVector_close_method);
  if (env->ExceptionCheck()) { throw std::runtime_error("ColumnVector.close threw an exception"); }
}

// A helper class to close the Java Column when being destroyed.
struct jcolumn_auto_cleaner {
  jcolumn_auto_cleaner(jobject jcv, JNIEnv* env)
  {
    this->jcv = jcv;
    this->env = env;
  }
  ~jcolumn_auto_cleaner() { column_vector_close(env, jcv); }

 private:
  JNIEnv* env;
  jobject jcv;
};

// the JNI groupby host_udf
class jni_groupby_host_udf final : public cudf::groupby_host_udf {
 public:
  jni_groupby_host_udf(JNIEnv* env, jobject j_host_udf)
    : jni_groupby_host_udf(env, j_host_udf, ptr_as_jlong(j_host_udf))
  {
  }

  virtual ~jni_groupby_host_udf()
  {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), MINIMUM_JNI_VERSION) == JNI_OK) {
      j_host_udf = del_global_ref(env, j_host_udf);
    }
    j_host_udf = nullptr;
  }

  // ======== Must overrides ========
  [[nodiscard]] bool is_equal(cudf::host_udf_base const& other) const override
  {
    auto o      = dynamic_cast<jni_groupby_host_udf const*>(&other);
    JNIEnv* env = get_jni_env(jvm);
    return o != nullptr && env->IsSameObject(o->j_host_udf, this->j_host_udf) &&
           o->j_udf_id == this->j_udf_id;
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    return 31 * std::hash<std::string>{}({"jni_host_groupby_udf"}) + std::hash<long>{}(j_udf_id);
  }

  [[nodiscard]] std::unique_ptr<cudf::host_udf_base> clone() const override
  {
    return std::unique_ptr<jni_groupby_host_udf>(
      new jni_groupby_host_udf(get_jni_env(jvm), j_host_udf, j_udf_id));
  }

  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return call_udf_method(this->method_aggregate, "'aggregate' threw an exception", true);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return call_udf_method(this->method_empty_output, "'getEmptyOutput' threw an exception", false);
  }

  // methods to access the group information
  using groupby_host_udf::get_group_offsets;
  using groupby_host_udf::get_grouped_values;
  using groupby_host_udf::get_num_groups;

 private:
  // Designed for 'clone' to use the same udf id to generate the same hash value if 'j_host_udf'
  // points to the same Java udf instance. `j_host_udf` can not be used for hash computation
  // because it varies among the "jni_groupby_host_udf" objects.
  jni_groupby_host_udf(JNIEnv* env, jobject j_host_udf, long udf_id) : j_udf_id(udf_id)
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(j_host_udf);
    if (cls == nullptr) { throw jni_exception("Host UDF class not found"); }

    this->method_empty_output =
      env->GetMethodID(cls, "getEmptyOutput", "()Lai/rapids/cudf/ColumnVector;");
    if (method_empty_output == nullptr) {
      throw jni_exception("'getEmptyOutput' method not found");
    }

    this->method_aggregate = env->GetMethodID(cls, "aggregate", "()Lai/rapids/cudf/ColumnVector;");
    if (method_empty_output == nullptr) { throw jni_exception("'aggregate' method not found"); }

    this->j_host_udf = add_global_ref(env, j_host_udf);
  }

  std::unique_ptr<cudf::column> call_udf_method(const jmethodID jmethod,
                                                const char* err_mess,
                                                const bool check_result) const
  {
    JNIEnv* env   = get_jni_env(jvm);
    jobject j_col = env->CallObjectMethod(j_host_udf, jmethod);
    jcolumn_auto_cleaner cleaner(j_col, env);  // for autoclose
    if (env->ExceptionCheck()) { throw std::runtime_error(err_mess); }
    cudf::column_view* col_view = column_vector_get_view(env, j_col);
    // a basic rows number check
    if (check_result && get_num_groups() != col_view->size()) {
      std::stringstream log;
      log << "Wrong rows number, expected: " << get_num_groups();
      log << ", but got " << col_view->size();
      throw std::runtime_error(log.str());
    }
    // Java Column has its own lifecycle control (aka reference count), so here clone a new one.
    return std::make_unique<cudf::column>(*col_view);
  }

  JavaVM* jvm;
  jobject j_host_udf;
  jmethodID method_empty_output;
  jmethodID method_aggregate;
  long j_udf_id;
};

}  // namespace jni
}  // namespace cudf

// ======== JNI Interfaces ========
extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_GroupByHostUDF_createNativeInstance(JNIEnv* env,
                                                                                jobject judf)
{
  try {
    cudf::jni::auto_set_device(env);
    return ptr_as_jlong(new cudf::jni::jni_groupby_host_udf(env, judf));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_GroupByHostUDF_getGroupOffsetsView(JNIEnv* env,
                                                                               jclass judf_class,
                                                                               jobject udf_handle)
{
  JNI_NULL_CHECK(env, udf_handle, "udf handle cannot be null.", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto native_udf = reinterpret_cast<cudf::jni::jni_groupby_host_udf*>(udf_handle);
    auto ret_span   = native_udf->get_group_offsets();
    return ptr_as_jlong(new cudf::column_view(ret_span));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_GroupByHostUDF_getGroupedValuesView(JNIEnv* env,
                                                                                jclass judf_class,
                                                                                jobject udf_handle)
{
  JNI_NULL_CHECK(env, udf_handle, "udf handle cannot be null.", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto native_udf = reinterpret_cast<cudf::jni::jni_groupby_host_udf*>(udf_handle);
    return ptr_as_jlong(new cudf::column_view(native_udf->get_grouped_values()));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_GroupByHostUDF_getNumGroups(JNIEnv* env,
                                                                        jclass judf_class,
                                                                        jobject udf_handle)
{
  JNI_NULL_CHECK(env, udf_handle, "udf handle cannot be null.", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto native_udf = reinterpret_cast<cudf::jni::jni_groupby_host_udf*>(udf_handle);
    return static_cast<jlong>(native_udf->get_num_groups());
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
