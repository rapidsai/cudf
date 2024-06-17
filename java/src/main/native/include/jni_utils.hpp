/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/utilities/error.hpp>

#include <rmm/detail/error.hpp>

#include <jni.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf {
namespace jni {

constexpr jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

constexpr char const* CUDA_ERROR_CLASS          = "ai/rapids/cudf/CudaException";
constexpr char const* CUDA_FATAL_ERROR_CLASS    = "ai/rapids/cudf/CudaFatalException";
constexpr char const* CUDF_ERROR_CLASS          = "ai/rapids/cudf/CudfException";
constexpr char const* CUDF_OVERFLOW_ERROR_CLASS = "ai/rapids/cudf/CudfColumnSizeOverflowException";
constexpr char const* CUDF_DTYPE_ERROR_CLASS    = "ai/rapids/cudf/CudfException";
constexpr char const* INDEX_OOB_CLASS           = "java/lang/ArrayIndexOutOfBoundsException";
constexpr char const* ILLEGAL_ARG_CLASS         = "java/lang/IllegalArgumentException";
constexpr char const* NPE_CLASS                 = "java/lang/NullPointerException";
constexpr char const* OOM_CLASS                 = "java/lang/OutOfMemoryError";

/**
 * @brief indicates that a JNI error of some kind was thrown and the main
 * function should return.
 */
class jni_exception : public std::runtime_error {
 public:
  jni_exception(char const* const message) : std::runtime_error(message) {}
  jni_exception(std::string const& message) : std::runtime_error(message) {}
};

/**
 * @brief throw a java exception and a C++ one for flow control.
 */
inline void throw_java_exception(JNIEnv* const env, char const* class_name, char const* message)
{
  jclass ex_class = env->FindClass(class_name);
  if (ex_class != NULL) { env->ThrowNew(ex_class, message); }
  throw jni_exception(message);
}

/**
 * @brief check if an java exceptions have been thrown and if so throw a C++
 * exception so the flow control stop processing.
 */
inline void check_java_exception(JNIEnv* const env)
{
  if (env->ExceptionCheck()) {
    // Not going to try to get the message out of the Exception, too complex and
    // might fail.
    throw jni_exception("JNI Exception...");
  }
}

/**
 * @brief Helper to convert a pointer to a jlong.
 *
 * This is useful when, for instance, converting a cudf::column pointer
 * to a jlong, for use in JNI.
 */
template <typename T>
jlong ptr_as_jlong(T* ptr)
{
  return reinterpret_cast<jlong>(ptr);
}

/**
 * @brief Helper to release the data held by a unique_ptr, and return
 * the pointer as a jlong.
 */
template <typename T>
jlong release_as_jlong(std::unique_ptr<T>&& ptr)
{
  return ptr_as_jlong(ptr.release());
}

/**
 * @brief Helper to release the data held by a unique_ptr, and return
 * the pointer as a jlong.
 */
template <typename T>
jlong release_as_jlong(std::unique_ptr<T>& ptr)
{
  return release_as_jlong(std::move(ptr));
}

class native_jdoubleArray_accessor {
 public:
  jdouble* getArrayElements(JNIEnv* const env, jdoubleArray arr) const
  {
    return env->GetDoubleArrayElements(arr, NULL);
  }

  jdoubleArray newArray(JNIEnv* const env, int len) const { return env->NewDoubleArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jdoubleArray jarr, int start, int len, jdouble const* arr) const
  {
    env->SetDoubleArrayRegion(jarr, start, len, arr);
  }

  void releaseArrayElements(JNIEnv* const env, jdoubleArray jarr, jdouble* arr, jint mode) const
  {
    env->ReleaseDoubleArrayElements(jarr, arr, mode);
  }
};

class native_jlongArray_accessor {
 public:
  jlong* getArrayElements(JNIEnv* const env, jlongArray arr) const
  {
    return env->GetLongArrayElements(arr, NULL);
  }

  jlongArray newArray(JNIEnv* const env, int len) const { return env->NewLongArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jlongArray jarr, int start, int len, jlong const* arr) const
  {
    env->SetLongArrayRegion(jarr, start, len, arr);
  }

  void releaseArrayElements(JNIEnv* const env, jlongArray jarr, jlong* arr, jint mode) const
  {
    env->ReleaseLongArrayElements(jarr, arr, mode);
  }
};

class native_jintArray_accessor {
 public:
  jint* getArrayElements(JNIEnv* const env, jintArray arr) const
  {
    return env->GetIntArrayElements(arr, NULL);
  }

  jintArray newArray(JNIEnv* const env, int len) const { return env->NewIntArray(len); }

  void setArrayRegion(JNIEnv* const env, jintArray jarr, int start, int len, jint const* arr) const
  {
    env->SetIntArrayRegion(jarr, start, len, arr);
  }

  void releaseArrayElements(JNIEnv* const env, jintArray jarr, jint* arr, jint mode) const
  {
    env->ReleaseIntArrayElements(jarr, arr, mode);
  }
};

class native_jbyteArray_accessor {
 public:
  jbyte* getArrayElements(JNIEnv* const env, jbyteArray arr) const
  {
    return env->GetByteArrayElements(arr, NULL);
  }

  jbyteArray newArray(JNIEnv* const env, int len) const { return env->NewByteArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jbyteArray jarr, int start, int len, jbyte const* arr) const
  {
    env->SetByteArrayRegion(jarr, start, len, arr);
  }

  void releaseArrayElements(JNIEnv* const env, jbyteArray jarr, jbyte* arr, jint mode) const
  {
    env->ReleaseByteArrayElements(jarr, arr, mode);
  }
};

class native_jbooleanArray_accessor {
 public:
  jboolean* getArrayElements(JNIEnv* const env, jbooleanArray arr) const
  {
    return env->GetBooleanArrayElements(arr, NULL);
  }

  jbooleanArray newArray(JNIEnv* const env, int len) const { return env->NewBooleanArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jbooleanArray jarr, int start, int len, jboolean const* arr) const
  {
    env->SetBooleanArrayRegion(jarr, start, len, arr);
  }

  void releaseArrayElements(JNIEnv* const env, jbooleanArray jarr, jboolean* arr, jint mode) const
  {
    env->ReleaseBooleanArrayElements(jarr, arr, mode);
  }
};

/**
 * @brief RAII for java arrays to be sure it is handled correctly.
 *
 * By default any changes to the array will be committed back when
 * the destructor is called unless cancel is called first.
 */
template <typename N_TYPE, typename J_ARRAY_TYPE, typename ACCESSOR>
class native_jArray {
 private:
  ACCESSOR access{};
  JNIEnv* const env;
  J_ARRAY_TYPE orig;
  int len;
  mutable N_TYPE* data_ptr;

  void init_data_ptr() const
  {
    if (orig != nullptr && data_ptr == nullptr) {
      data_ptr = access.getArrayElements(env, orig);
      check_java_exception(env);
    }
  }

 public:
  native_jArray(native_jArray const&)            = delete;
  native_jArray& operator=(native_jArray const&) = delete;

  native_jArray(JNIEnv* const env, J_ARRAY_TYPE orig) : env(env), orig(orig), len(0), data_ptr(NULL)
  {
    if (orig != NULL) {
      len = env->GetArrayLength(orig);
      check_java_exception(env);
    }
  }

  native_jArray(JNIEnv* const env, int len)
    : env(env), orig(access.newArray(env, len)), len(len), data_ptr(NULL)
  {
    check_java_exception(env);
  }

  native_jArray(JNIEnv* const env, N_TYPE const* arr, int len)
    : env(env), orig(access.newArray(env, len)), len(len), data_ptr(NULL)
  {
    check_java_exception(env);
    access.setArrayRegion(env, orig, 0, len, arr);
    check_java_exception(env);
  }

  native_jArray(JNIEnv* const env, std::vector<N_TYPE> const& arr)
    : env(env), orig(access.newArray(env, arr.size())), len(arr.size()), data_ptr(NULL)
  {
    check_java_exception(env);
    access.setArrayRegion(env, orig, 0, len, arr.data());
    check_java_exception(env);
  }

  bool is_null() const noexcept { return orig == NULL; }

  int size() const noexcept { return len; }

  N_TYPE operator[](int index) const
  {
    if (orig == NULL) { throw_java_exception(env, NPE_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= len) { throw_java_exception(env, INDEX_OOB_CLASS, "NOT IN BOUNDS"); }
    return data()[index];
  }

  N_TYPE& operator[](int index)
  {
    if (orig == NULL) { throw_java_exception(env, NPE_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= len) { throw_java_exception(env, INDEX_OOB_CLASS, "NOT IN BOUNDS"); }
    return data()[index];
  }

  const N_TYPE* const data() const
  {
    init_data_ptr();
    return data_ptr;
  }

  N_TYPE* data()
  {
    init_data_ptr();
    return data_ptr;
  }

  const N_TYPE* const begin() const { return data(); }

  N_TYPE* begin() { return data(); }

  const N_TYPE* const end() const { return data() + size(); }

  N_TYPE* end() { return data() + size(); }

  const J_ARRAY_TYPE get_jArray() const { return orig; }

  J_ARRAY_TYPE get_jArray() { return orig; }

  /**
   * @brief Conversion to std::vector
   *
   * @tparam target_t Target data type
   * @return std::vector<target_t> Vector with the copied contents
   */
  template <typename target_t = N_TYPE>
  std::vector<target_t> to_vector() const
  {
    std::vector<target_t> ret;
    ret.reserve(size());
    std::copy(begin(), end(), std::back_inserter(ret));
    return ret;
  }

  /**
   * @brief if data has been written back into this array, don't commit
   * it.
   */
  void cancel()
  {
    if (data_ptr != NULL && orig != NULL) {
      access.releaseArrayElements(env, orig, data_ptr, JNI_ABORT);
      data_ptr = NULL;
    }
  }

  void commit()
  {
    if (data_ptr != NULL && orig != NULL) {
      access.releaseArrayElements(env, orig, data_ptr, 0);
      data_ptr = NULL;
    }
  }

  ~native_jArray() { commit(); }
};

using native_jdoubleArray = native_jArray<jdouble, jdoubleArray, native_jdoubleArray_accessor>;
using native_jlongArray   = native_jArray<jlong, jlongArray, native_jlongArray_accessor>;
using native_jintArray    = native_jArray<jint, jintArray, native_jintArray_accessor>;
using native_jbyteArray   = native_jArray<jbyte, jbyteArray, native_jbyteArray_accessor>;

/**
 * @brief Specialization of native_jArray for jboolean
 *
 * This class adds special support for conversion to std::vector<X>, where the element
 * value is chosen depending on the jboolean value.
 */
struct native_jbooleanArray
  : native_jArray<jboolean, jbooleanArray, native_jbooleanArray_accessor> {
  native_jbooleanArray(JNIEnv* const env, jbooleanArray orig)
    : native_jArray<jboolean, jbooleanArray, native_jbooleanArray_accessor>(env, orig)
  {
  }

  native_jbooleanArray(native_jbooleanArray const&)            = delete;
  native_jbooleanArray& operator=(native_jbooleanArray const&) = delete;

  template <typename target_t>
  std::vector<target_t> transform_if_else(target_t const& if_true, target_t const& if_false) const
  {
    std::vector<target_t> ret;
    ret.reserve(size());
    std::transform(begin(), end(), std::back_inserter(ret), [&](jboolean const& b) {
      return b ? if_true : if_false;
    });
    return ret;
  }
};

/**
 * @brief wrapper around native_jlongArray to make it take pointers instead.
 *
 * By default any changes to the array will be committed back when
 * the destructor is called unless cancel is called first.
 */
template <typename T>
class native_jpointerArray {
 private:
  native_jlongArray wrapped;
  JNIEnv* const env;

 public:
  native_jpointerArray(native_jpointerArray const&)            = delete;
  native_jpointerArray& operator=(native_jpointerArray const&) = delete;

  native_jpointerArray(JNIEnv* const env, jlongArray orig) : wrapped(env, orig), env(env) {}

  native_jpointerArray(JNIEnv* const env, int len) : wrapped(env, len), env(env) {}

  native_jpointerArray(JNIEnv* const env, T* arr, int len) : wrapped(env, arr, len), env(env) {}

  bool is_null() const noexcept { return wrapped.is_null(); }

  int size() const noexcept { return wrapped.size(); }

  T* operator[](int index) const
  {
    if (data() == NULL) { throw_java_exception(env, NPE_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= wrapped.size()) {
      throw_java_exception(env, INDEX_OOB_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  T*& operator[](int index)
  {
    if (data() == NULL) { throw_java_exception(env, NPE_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= wrapped.size()) {
      throw_java_exception(env, INDEX_OOB_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  T* const* data() const { return reinterpret_cast<T* const*>(wrapped.data()); }

  T** data() { return reinterpret_cast<T**>(wrapped.data()); }

  T* const* begin() const { return data(); }
  T* const* end() const { return data() + size(); }

  const jlongArray get_jArray() const { return wrapped.get_jArray(); }

  jlongArray get_jArray() { return wrapped.get_jArray(); }

  void assert_no_nulls() const
  {
    if (std::any_of(data(), data() + size(), [](T* const ptr) { return ptr == nullptr; })) {
      throw_java_exception(env, NPE_CLASS, "pointer is NULL");
    }
  }

  /**
   * @brief Convert from `T*[]` to `vector<T>`.
   */
  std::vector<T> get_dereferenced() const
  {
    assert_no_nulls();
    auto ret = std::vector<T>{};
    ret.reserve(size());
    std::transform(
      data(), data() + size(), std::back_inserter(ret), [](T* const& p) { return *p; });
    return ret;
  }

  /**
   * @brief if data has been written back into this array, don't commit
   * it.
   */
  void cancel() { wrapped.cancel(); }

  void commit() { wrapped.commit(); }
};

/**
 * @brief wrapper around native_jlongArray to hold pointers that are deleted
 * if not released, like std::unique_ptr.
 *
 * By default any changes to the array will be committed back when
 * released unless cancel is called first.
 */
template <typename T, typename D = std::default_delete<T>>
class unique_jpointerArray {
 private:
  std::unique_ptr<native_jpointerArray<T>> wrapped;
  D del;

 public:
  unique_jpointerArray(unique_jpointerArray const&)            = delete;
  unique_jpointerArray& operator=(unique_jpointerArray const&) = delete;

  unique_jpointerArray(JNIEnv* const env, jlongArray orig)
    : wrapped(new native_jpointerArray<T>(env, orig))
  {
  }

  unique_jpointerArray(JNIEnv* const env, jlongArray orig, D const& del)
    : wrapped(new native_jpointerArray<T>(env, orig)), del(del)
  {
  }

  unique_jpointerArray(JNIEnv* const env, int len) : wrapped(new native_jpointerArray<T>(env, len))
  {
  }

  unique_jpointerArray(JNIEnv* const env, int len, D const& del)
    : wrapped(new native_jpointerArray<T>(env, len)), del(del)
  {
  }

  unique_jpointerArray(JNIEnv* const env, T* arr, int len)
    : wrapped(new native_jpointerArray<T>(env, arr, len))
  {
  }

  unique_jpointerArray(JNIEnv* const env, T* arr, int len, D const& del)
    : wrapped(new native_jpointerArray<T>(env, arr, len)), del(del)
  {
  }

  bool is_null() const noexcept { return wrapped == NULL || wrapped->is_null(); }

  int size() const noexcept { return wrapped == NULL ? 0 : wrapped->size(); }

  void reset(int index, T* new_ptr = NULL)
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    T* old = (*wrapped)[index];
    if (old != new_ptr) {
      (*wrapped)[index] = new_ptr;
      del(old);
    }
  }

  T* get(int index)
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    return (*wrapped)[index];
  }

  T* const* get()
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    return wrapped->data();
  }

  jlongArray release()
  {
    if (wrapped == NULL) { return NULL; }
    wrapped->commit();
    jlongArray ret = wrapped->get_jArray();
    wrapped.reset(NULL);
    return ret;
  }

  ~unique_jpointerArray()
  {
    if (wrapped != NULL) {
      for (int i = 0; i < wrapped->size(); i++) {
        reset(i, NULL);
      }
    }
  }
};

/**
 * @brief RAII for jstring to be sure it is handled correctly.
 */
class native_jstring {
 private:
  JNIEnv* env;
  jstring orig;
  mutable char const* cstr;
  mutable size_t cstr_length;

  void init_cstr() const
  {
    if (orig != NULL && cstr == NULL) {
      cstr_length = env->GetStringUTFLength(orig);
      cstr        = env->GetStringUTFChars(orig, 0);
      check_java_exception(env);
    }
  }

 public:
  native_jstring(native_jstring const&)            = delete;
  native_jstring& operator=(native_jstring const&) = delete;

  native_jstring(native_jstring&& other) noexcept
    : env(other.env), orig(other.orig), cstr(other.cstr), cstr_length(other.cstr_length)
  {
    other.cstr = NULL;
  }

  native_jstring(JNIEnv* const env, jstring orig) : env(env), orig(orig), cstr(NULL), cstr_length(0)
  {
  }

  native_jstring& operator=(native_jstring const&& other)
  {
    if (orig != NULL && cstr != NULL) { env->ReleaseStringUTFChars(orig, cstr); }
    this->env         = other.env;
    this->orig        = other.orig;
    this->cstr        = other.cstr;
    this->cstr_length = other.cstr_length;
    other.cstr        = NULL;
    return *this;
  }

  bool is_null() const noexcept { return orig == NULL; }

  char const* get() const
  {
    init_cstr();
    return cstr;
  }

  size_t size_bytes() const
  {
    init_cstr();
    return cstr_length;
  }

  bool is_empty() const
  {
    if (cstr != NULL) {
      return cstr_length <= 0;
    } else if (orig != NULL) {
      jsize len = env->GetStringLength(orig);
      check_java_exception(env);
      return len <= 0;
    }
    return true;
  }

  const jstring get_jstring() const { return orig; }

  ~native_jstring()
  {
    if (orig != NULL && cstr != NULL) { env->ReleaseStringUTFChars(orig, cstr); }
  }
};

/**
 * @brief jobjectArray wrapper to make accessing it more convenient.
 */
template <typename T>
class native_jobjectArray {
 private:
  JNIEnv* const env;
  jobjectArray orig;
  int len;

 public:
  native_jobjectArray(JNIEnv* const env, jobjectArray orig) : env(env), orig(orig), len(0)
  {
    if (orig != NULL) {
      len = env->GetArrayLength(orig);
      check_java_exception(env);
    }
  }

  bool is_null() const noexcept { return orig == NULL; }

  int size() const noexcept { return len; }

  T operator[](int index) const { return get(index); }

  T get(int index) const
  {
    if (orig == NULL) { throw_java_exception(env, NPE_CLASS, "jobjectArray pointer is NULL"); }
    T ret = static_cast<T>(env->GetObjectArrayElement(orig, index));
    check_java_exception(env);
    return ret;
  }

  void set(int index, T const& val)
  {
    if (orig == NULL) { throw_java_exception(env, NPE_CLASS, "jobjectArray pointer is NULL"); }
    env->SetObjectArrayElement(orig, index, val);
    check_java_exception(env);
  }

  jobjectArray wrapped() { return orig; }
};

/**
 * @brief jobjectArray wrapper to make accessing strings safe through RAII
 * and convenient.
 */
class native_jstringArray {
 private:
  JNIEnv* const env;
  native_jobjectArray<jstring> arr;
  mutable std::vector<native_jstring> cache;
  mutable std::vector<std::string> cpp_cache;
  mutable std::vector<char const*> c_cache;

  void init_cache() const
  {
    if (!arr.is_null() && cache.empty()) {
      int size = this->size();
      cache.reserve(size);
      for (int i = 0; i < size; i++) {
        cache.push_back(native_jstring(env, arr.get(i)));
      }
    }
  }

  void init_c_cache() const
  {
    if (!arr.is_null() && c_cache.empty()) {
      init_cache();
      int size = this->size();
      c_cache.reserve(size);
      for (int i = 0; i < size; i++) {
        c_cache.push_back(cache[i].get());
      }
    }
  }

  void init_cpp_cache() const
  {
    if (!arr.is_null() && cpp_cache.empty()) {
      init_cache();
      int size = this->size();
      cpp_cache.reserve(size);
      for (int i = 0; i < size; i++) {
        cpp_cache.push_back(cache[i].get());
      }
    }
  }

  void update_caches(int index, jstring val)
  {
    if (!cache.empty()) {
      cache[index] = native_jstring(env, val);
      if (!c_cache.empty()) { c_cache[index] = cache[index].get(); }

      if (!cpp_cache.empty()) { cpp_cache[index] = cache[index].get(); }
    } else if (!c_cache.empty() || !cpp_cache.empty()) {
      // Illegal state
      throw std::logic_error("CACHING IS MESSED UP");
    }
  }

 public:
  native_jstringArray(JNIEnv* const env, jobjectArray orig) : env(env), arr(env, orig) {}

  bool is_null() const noexcept { return arr.is_null(); }

  int size() const noexcept { return arr.size(); }

  native_jstring& operator[](int index) const { return get(index); }

  native_jstring& get(int index) const
  {
    if (arr.is_null()) {
      throw_java_exception(env, cudf::jni::NPE_CLASS, "jstringArray pointer is NULL");
    }
    init_cache();
    return cache[index];
  }

  char const** const as_c_array() const
  {
    init_c_cache();
    return c_cache.data();
  }

  const std::vector<std::string> as_cpp_vector() const
  {
    init_cpp_cache();
    return cpp_cache;
  }

  void set(int index, jstring val)
  {
    arr.set(index, val);
    update_caches(index, val);
  }

  void set(int index, native_jstring const& val)
  {
    arr.set(index, val.get_jstring());
    update_caches(index, val.get_jstring());
  }

  void set(int index, char const* val)
  {
    jstring str = env->NewStringUTF(val);
    check_java_exception(env);
    arr.set(index, str);
    update_caches(index, str);
  }
};

/**
 * @brief create a cuda exception from a given cudaError_t
 */
inline jthrowable cuda_exception(JNIEnv* const env, cudaError_t status, jthrowable cause = NULL)
{
  char const* ex_class_name;

  // Calls cudaGetLastError twice. It is nearly certain that a fatal error occurred if the second
  // call doesn't return with cudaSuccess.
  cudaGetLastError();
  auto const last = cudaGetLastError();
  // Call cudaDeviceSynchronize to ensure `last` did not result from an asynchronous error.
  // between two calls.
  if (status == last && last == cudaDeviceSynchronize()) {
    ex_class_name = cudf::jni::CUDA_FATAL_ERROR_CLASS;
  } else {
    ex_class_name = cudf::jni::CUDA_ERROR_CLASS;
  }

  jclass ex_class = env->FindClass(ex_class_name);
  if (ex_class == NULL) { return NULL; }
  jmethodID ctor_id =
    env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;ILjava/lang/Throwable;)V");
  if (ctor_id == NULL) { return NULL; }

  jstring msg = env->NewStringUTF(cudaGetErrorString(status));
  if (msg == NULL) { return NULL; }

  jint err_code = static_cast<jint>(status);

  jobject ret = env->NewObject(ex_class, ctor_id, msg, err_code, cause);
  return (jthrowable)ret;
}

inline void jni_cuda_check(JNIEnv* const env, cudaError_t cuda_status)
{
  if (cudaSuccess != cuda_status) {
    jthrowable jt = cuda_exception(env, cuda_status);
    if (jt != NULL) { env->Throw(jt); }
    throw jni_exception(std::string("CUDA ERROR: code ") +
                        std::to_string(static_cast<int>(cuda_status)));
  }
}

inline auto add_global_ref(JNIEnv* env, jobject jobj)
{
  auto new_global_ref = env->NewGlobalRef(jobj);
  if (new_global_ref == nullptr) { throw cudf::jni::jni_exception("global ref"); }
  return new_global_ref;
}

inline nullptr_t del_global_ref(JNIEnv* env, jobject jobj)
{
  if (jobj != nullptr) { env->DeleteGlobalRef(jobj); }
  return nullptr;
}

}  // namespace jni
}  // namespace cudf

#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val)    \
  {                                                   \
    if (env->ExceptionOccurred()) { return ret_val; } \
  }

#define JNI_THROW_NEW(env, class_name, message, ret_val) \
  {                                                      \
    jclass ex_class = env->FindClass(class_name);        \
    if (ex_class == NULL) { return ret_val; }            \
    env->ThrowNew(ex_class, message);                    \
    return ret_val;                                      \
  }

// Throw a new exception only if one is not pending then always return with the specified value
#define JNI_CHECK_THROW_CUDF_EXCEPTION(env, class_name, message, stacktrace, ret_val)           \
  {                                                                                             \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                 \
    auto const ex_class = env->FindClass(class_name);                                           \
    if (ex_class == nullptr) { return ret_val; }                                                \
    auto const ctor_id =                                                                        \
      env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V");          \
    if (ctor_id == nullptr) { return ret_val; }                                                 \
    auto const empty_str = std::string{""};                                                     \
    auto const jmessage  = env->NewStringUTF(message == nullptr ? empty_str.c_str() : message); \
    if (jmessage == nullptr) { return ret_val; }                                                \
    auto const jstacktrace =                                                                    \
      env->NewStringUTF(stacktrace == nullptr ? empty_str.c_str() : stacktrace);                \
    if (jstacktrace == nullptr) { return ret_val; }                                             \
    auto const jobj = env->NewObject(ex_class, ctor_id, jmessage, jstacktrace);                 \
    if (jobj == nullptr) { return ret_val; }                                                    \
    env->Throw(reinterpret_cast<jthrowable>(jobj));                                             \
    return ret_val;                                                                             \
  }

// Throw a new exception only if one is not pending then always return with the specified value
#define JNI_CHECK_THROW_CUDA_EXCEPTION(env, class_name, message, stacktrace, error_code, ret_val)   \
  {                                                                                                 \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                     \
    auto const ex_class = env->FindClass(class_name);                                               \
    if (ex_class == nullptr) { return ret_val; }                                                    \
    auto const ctor_id =                                                                            \
      env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;I)V");             \
    if (ctor_id == nullptr) { return ret_val; }                                                     \
    auto const empty_str = std::string{""};                                                         \
    auto const jmessage  = env->NewStringUTF(message == nullptr ? empty_str.c_str() : message);     \
    if (jmessage == nullptr) { return ret_val; }                                                    \
    auto const jstacktrace =                                                                        \
      env->NewStringUTF(stacktrace == nullptr ? empty_str.c_str() : stacktrace);                    \
    if (jstacktrace == nullptr) { return ret_val; }                                                 \
    auto const jerror_code = static_cast<jint>(error_code);                                         \
    auto const jobj        = env->NewObject(ex_class, ctor_id, jmessage, jstacktrace, jerror_code); \
    if (jobj == nullptr) { return ret_val; }                                                        \
    env->Throw(reinterpret_cast<jthrowable>(jobj));                                                 \
    return ret_val;                                                                                 \
  }

#define JNI_NULL_CHECK(env, obj, error_msg, ret_val)                                  \
  {                                                                                   \
    if ((obj) == 0) { JNI_THROW_NEW(env, cudf::jni::NPE_CLASS, error_msg, ret_val); } \
  }

#define JNI_ARG_CHECK(env, obj, error_msg, ret_val)                                       \
  {                                                                                       \
    if (!(obj)) { JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, error_msg, ret_val); } \
  }

#define CATCH_STD_CLASS(env, class_name, ret_val)                                                 \
  catch (const rmm::out_of_memory& e)                                                             \
  {                                                                                               \
    JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val);                                                   \
    auto const what =                                                                             \
      std::string("Could not allocate native memory: ") + (e.what() == nullptr ? "" : e.what());  \
    JNI_THROW_NEW(env, cudf::jni::OOM_CLASS, what.c_str(), ret_val);                              \
  }                                                                                               \
  catch (const cudf::fatal_cuda_error& e)                                                         \
  {                                                                                               \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                               \
      env, cudf::jni::CUDA_FATAL_ERROR_CLASS, e.what(), e.stacktrace(), e.error_code(), ret_val); \
  }                                                                                               \
  catch (const cudf::cuda_error& e)                                                               \
  {                                                                                               \
    JNI_CHECK_THROW_CUDA_EXCEPTION(                                                               \
      env, cudf::jni::CUDA_ERROR_CLASS, e.what(), e.stacktrace(), e.error_code(), ret_val);       \
  }                                                                                               \
  catch (const cudf::data_type_error& e)                                                          \
  {                                                                                               \
    JNI_CHECK_THROW_CUDF_EXCEPTION(                                                               \
      env, cudf::jni::CUDF_DTYPE_ERROR_CLASS, e.what(), e.stacktrace(), ret_val);                 \
  }                                                                                               \
  catch (std::overflow_error const& e)                                                            \
  {                                                                                               \
    JNI_CHECK_THROW_CUDF_EXCEPTION(env,                                                           \
                                   cudf::jni::CUDF_OVERFLOW_ERROR_CLASS,                          \
                                   e.what(),                                                      \
                                   "No native stacktrace is available.",                          \
                                   ret_val);                                                      \
  }                                                                                               \
  catch (const std::exception& e)                                                                 \
  {                                                                                               \
    char const* stacktrace = "No native stacktrace is available.";                                \
    if (auto const cudf_ex = dynamic_cast<cudf::logic_error const*>(&e); cudf_ex != nullptr) {    \
      stacktrace = cudf_ex->stacktrace();                                                         \
    }                                                                                             \
    /* Double check whether the thrown exception is unrecoverable CUDA error or not. */           \
    /* Like cudf::detail::throw_cuda_error, it is nearly certain that a fatal error  */           \
    /* occurred if the second call doesn't return with cudaSuccess. */                            \
    cudaGetLastError();                                                                           \
    auto const last = cudaFree(0);                                                                \
    if (cudaSuccess != last && last == cudaDeviceSynchronize()) {                                 \
      /* Throw CudaFatalException since the thrown exception is unrecoverable CUDA error */       \
      JNI_CHECK_THROW_CUDA_EXCEPTION(                                                             \
        env, cudf::jni::CUDA_FATAL_ERROR_CLASS, e.what(), stacktrace, last, ret_val);             \
    }                                                                                             \
    JNI_CHECK_THROW_CUDF_EXCEPTION(env, class_name, e.what(), stacktrace, ret_val);               \
  }

#define CATCH_STD(env, ret_val) CATCH_STD_CLASS(env, cudf::jni::CUDF_ERROR_CLASS, ret_val)
