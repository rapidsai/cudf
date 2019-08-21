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

#include "cudf/binaryop.hpp"
#include "cudf/filling.hpp"
#include "cudf/reduction.hpp"
#include "cudf/replace.hpp"
#include "cudf/stream_compaction.hpp"
#include "cudf/unary.hpp"

#include "jni_utils.hpp"

using unique_nvcat_ptr = std::unique_ptr<NVCategory, decltype(&NVCategory::destroy)>;
using unique_nvstr_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

namespace cudf {
namespace jni {

static const jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

static jclass scalar_jclass;
static jmethodID scalar_from_null;
static jmethodID scalar_timestamp_from_null;
static jmethodID scalar_from_bool;
static jmethodID scalar_from_byte;
static jmethodID scalar_from_short;
static jmethodID scalar_from_int;
static jmethodID scalar_date_from_int;
static jmethodID scalar_from_long;
static jmethodID scalar_date_from_long;
static jmethodID scalar_timestamp_from_long;
static jmethodID scalar_from_float;
static jmethodID scalar_from_double;

#define SCALAR_CLASS "ai/rapids/cudf/Scalar"
#define SCALAR_FACTORY_SIG(param_sig) "(" param_sig ")L" SCALAR_CLASS ";"

// Cache useful method IDs of the Scalar class along with a global reference
// to the class. This avoids redundant, dynamic class and method lookups later.
// Returns true if the class and method IDs were successfully cached or false
// if an error occurred.
static bool cache_scalar_jni(JNIEnv *env) {
  jclass cls = env->FindClass(SCALAR_CLASS);
  if (cls == nullptr) {
    return false;
  }

  scalar_from_null = env->GetStaticMethodID(cls, "fromNull", SCALAR_FACTORY_SIG("I"));
  if (scalar_from_null == nullptr) {
    return false;
  }
  scalar_timestamp_from_null =
      env->GetStaticMethodID(cls, "timestampFromNull", SCALAR_FACTORY_SIG("I"));
  if (scalar_timestamp_from_null == nullptr) {
    return false;
  }
  scalar_from_bool = env->GetStaticMethodID(cls, "fromBool", SCALAR_FACTORY_SIG("Z"));
  if (scalar_from_bool == nullptr) {
    return false;
  }
  scalar_from_byte = env->GetStaticMethodID(cls, "fromByte", SCALAR_FACTORY_SIG("B"));
  if (scalar_from_byte == nullptr) {
    return false;
  }
  scalar_from_short = env->GetStaticMethodID(cls, "fromShort", SCALAR_FACTORY_SIG("S"));
  if (scalar_from_short == nullptr) {
    return false;
  }
  scalar_from_int = env->GetStaticMethodID(cls, "fromInt", SCALAR_FACTORY_SIG("I"));
  if (scalar_from_int == nullptr) {
    return false;
  }
  scalar_date_from_int = env->GetStaticMethodID(cls, "dateFromInt", SCALAR_FACTORY_SIG("I"));
  if (scalar_date_from_int == nullptr) {
    return false;
  }
  scalar_from_long = env->GetStaticMethodID(cls, "fromLong", SCALAR_FACTORY_SIG("J"));
  if (scalar_from_long == nullptr) {
    return false;
  }
  scalar_date_from_long = env->GetStaticMethodID(cls, "dateFromLong", SCALAR_FACTORY_SIG("J"));
  if (scalar_date_from_long == nullptr) {
    return false;
  }
  scalar_timestamp_from_long =
      env->GetStaticMethodID(cls, "timestampFromLong", SCALAR_FACTORY_SIG("JI"));
  if (scalar_timestamp_from_long == nullptr) {
    return false;
  }
  scalar_from_float = env->GetStaticMethodID(cls, "fromFloat", SCALAR_FACTORY_SIG("F"));
  if (scalar_from_float == nullptr) {
    return false;
  }
  scalar_from_double = env->GetStaticMethodID(cls, "fromDouble", SCALAR_FACTORY_SIG("D"));
  if (scalar_from_double == nullptr) {
    return false;
  }

  // Convert local reference to global so it cannot be garbage collected.
  scalar_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (scalar_jclass == nullptr) {
    return false;
  }

  return true;
}

static void release_scalar_jni(JNIEnv *env) {
  if (scalar_jclass != nullptr) {
    env->DeleteGlobalRef(scalar_jclass);
    scalar_jclass = nullptr;
  }
}

jobject jscalar_from_scalar(JNIEnv *env, const gdf_scalar &scalar, gdf_time_unit time_unit) {
  jobject obj = nullptr;
  if (scalar.is_valid) {
    switch (scalar.dtype) {
      case GDF_INT8:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_byte, scalar.data.si08);
        break;
      case GDF_INT16:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_short, scalar.data.si16);
        break;
      case GDF_INT32:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_int, scalar.data.si32);
        break;
      case GDF_INT64:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_long, scalar.data.si64);
        break;
      case GDF_FLOAT32:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_float, scalar.data.fp32);
        break;
      case GDF_FLOAT64:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_double, scalar.data.fp64);
        break;
      case GDF_BOOL8:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_bool, scalar.data.b08);
        break;
      case GDF_DATE32:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_date_from_int, scalar.data.dt32);
        break;
      case GDF_DATE64:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_date_from_long, scalar.data.dt64);
        break;
      case GDF_TIMESTAMP:
        obj = env->CallStaticObjectMethod(scalar_jclass, scalar_timestamp_from_long,
                                          scalar.data.tmst, time_unit);
        break;
      default:
        throw_java_exception(env, "java/lang/UnsupportedOperationException",
                             "Unsupported native scalar type");
        break;
    }
  } else {
    if (scalar.dtype == GDF_TIMESTAMP) {
      obj = env->CallStaticObjectMethod(scalar_jclass, scalar_timestamp_from_null, time_unit);
    } else {
      obj = env->CallStaticObjectMethod(scalar_jclass, scalar_from_null, scalar.dtype);
    }
  }
  return obj;
}

static void gdf_scalar_init(gdf_scalar *scalar, jlong int_values, jfloat f_value, jdouble d_value,
                            jboolean is_valid, int dtype) {
  scalar->dtype = static_cast<gdf_dtype>(dtype);
  scalar->is_valid = is_valid;
  switch (scalar->dtype) {
    case GDF_INT8: scalar->data.si08 = static_cast<char>(int_values); break;
    case GDF_INT16: scalar->data.si16 = static_cast<short>(int_values); break;
    case GDF_INT32: scalar->data.si32 = static_cast<int>(int_values); break;
    case GDF_INT64: scalar->data.si64 = static_cast<long>(int_values); break;
    case GDF_DATE32: scalar->data.dt32 = static_cast<gdf_date32>(int_values); break;
    case GDF_DATE64: scalar->data.dt64 = static_cast<gdf_date64>(int_values); break;
    case GDF_TIMESTAMP: scalar->data.tmst = static_cast<gdf_timestamp>(int_values); break;
    case GDF_BOOL8: scalar->data.b08 = static_cast<char>(int_values); break;
    case GDF_FLOAT32: scalar->data.fp32 = f_value; break;
    case GDF_FLOAT64: scalar->data.fp64 = d_value; break;
    default: throw std::logic_error("Unsupported scalar type");
  }
}

static jni_rmm_unique_ptr<gdf_valid_type>
copy_validity(JNIEnv *env, gdf_size_type size, gdf_size_type null_count, gdf_valid_type *valid) {
  jni_rmm_unique_ptr<gdf_valid_type> ret{};
  if (null_count > 0) {
    gdf_size_type copy_size = ((size + 7) / 8);
    gdf_size_type alloc_size = gdf_valid_allocation_size(size);
    ret = jni_rmm_alloc<gdf_valid_type>(env, alloc_size);
    JNI_CUDA_TRY(env, 0, cudaMemcpy(ret.get(), valid, copy_size, cudaMemcpyDeviceToDevice));
  }
  return ret;
}

static jlong cast_string_cat_to(JNIEnv *env, NVCategory *cat, gdf_dtype target_type,
                                gdf_time_unit target_unit, gdf_size_type size,
                                gdf_size_type null_count, gdf_valid_type *valid) {
  switch (target_type) {
    case GDF_STRING: {
      if (size == 0) {
        gdf_column_wrapper output(size, target_type, null_count, nullptr,
                                nullptr);
        return reinterpret_cast<jlong>(output.release());
      }
      unique_nvstr_ptr str(cat->to_strings(), &NVStrings::destroy);

      jni_rmm_unique_ptr<gdf_valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, str.release(), valid_copy.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string_cat to");
  }
}

static jlong cast_string_to(JNIEnv *env, NVStrings *str, gdf_dtype target_type,
                            gdf_time_unit target_unit, gdf_size_type size, gdf_size_type null_count,
                            gdf_valid_type *valid) {
  switch (target_type) {
    case GDF_STRING_CATEGORY: {
      if (size == 0) {
        gdf_column_wrapper output(size, target_type, null_count, nullptr,
                                nullptr, nullptr);
        return reinterpret_cast<jlong>(output.release());
      }
      unique_nvcat_ptr cat(NVCategory::create_from_strings(*str), &NVCategory::destroy);
      auto cat_data = jni_rmm_alloc<int>(env, sizeof(int) * size);
      if (size != cat->get_values(cat_data.get(), true)) {
        JNI_THROW_NEW(env, "java/lang/IllegalStateException", "Internal Error copying str cat data",
                      0);
      }

      jni_rmm_unique_ptr<gdf_valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, cat_data.release(),
                                valid_copy.release(), cat.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string to");
  }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // cache some class/method/field lookups
  if (!cudf::jni::cache_scalar_jni(env)) {
    return JNI_ERR;
  }

  return cudf::jni::MINIMUM_JNI_VERSION;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return;
  }

  cudf::jni::release_scalar_jni(env);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfUnaryMath(JNIEnv *env, jclass, jlong input_ptr,
                                                              jint int_op, jint out_dtype) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    cudf::unary_op op = static_cast<cudf::unary_op>(int_op);
    std::unique_ptr<gdf_column, decltype(free) *> ret(
        static_cast<gdf_column *>(malloc(sizeof(gdf_column))), free);
    *ret.get() = cudf::unary_operation(*input, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVV(JNIEnv *env, jclass, jlong lhs_ptr,
                                                               jlong rhs_ptr, jint int_op,
                                                               jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhs_ptr);
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhs_ptr);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type,
                                      lhs->valid != nullptr || rhs->valid != nullptr);
    // Should be null count           lhs->null_count > 0 || rhs->null_count >
    // 0);

    cudf::binary_operation(ret.get(), lhs, rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpSV(
    JNIEnv *env, jclass, jlong lhs_int_values, jfloat lhs_f_value, jdouble lhs_d_value,
    jboolean lhs_is_valid, int lhs_dtype, jlong rhs_ptr, jint int_op, jint out_dtype) {
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    gdf_scalar lhs{};
    cudf::jni::gdf_scalar_init(&lhs, lhs_int_values, lhs_f_value, lhs_d_value, lhs_is_valid,
                               lhs_dtype);
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhs_ptr);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(rhs->size, out_type, !lhs.is_valid || rhs->valid != nullptr);
    // Should be null count           !lhs.is_valid || rhs->null_count > 0);

    cudf::binary_operation(ret.get(), &lhs, rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVS(
    JNIEnv *env, jclass, jlong lhs_ptr, jlong rhs_int_values, jfloat rhs_f_value,
    jdouble rhs_d_value, jboolean rhs_is_valid, int rhs_dtype, jint int_op, jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhs_ptr);
    gdf_scalar rhs{};
    cudf::jni::gdf_scalar_init(&rhs, rhs_int_values, rhs_f_value, rhs_d_value, rhs_is_valid,
                               rhs_dtype);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type, !rhs.is_valid || lhs->valid != nullptr);
    // Should be null count           !rhs.is_valid || lhs->null_count > 0);

    cudf::binary_operation(ret.get(), lhs, &rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeYear(JNIEnv *env, jclass,
                                                                        jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_year(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMonth(JNIEnv *env, jclass,
                                                                         jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_month(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeDay(JNIEnv *env, jclass,
                                                                       jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_day(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeHour(JNIEnv *env, jclass,
                                                                        jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_hour(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeMinute(JNIEnv *env, jclass,
                                                                          jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_minute(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfExtractDatetimeSecond(JNIEnv *env, jclass,
                                                                          jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    cudf::jni::gdf_column_wrapper output(input->size, GDF_INT16, input->null_count != 0);
    JNI_GDF_TRY(env, 0, gdf_extract_datetime_second(input, output.get()));
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfCast(JNIEnv *env, jclass, jlong input_ptr,
                                                         jint dtype, jint time_unit) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    gdf_dtype c_dtype = static_cast<gdf_dtype>(dtype);
    gdf_dtype_extra_info info{};
    gdf_time_unit c_time_unit = static_cast<gdf_time_unit>(time_unit);
    size_t size = input->size;
    if (input->dtype == GDF_STRING) {
      NVStrings *str = static_cast<NVStrings *>(input->data);
      return cudf::jni::cast_string_to(env, str, c_dtype, c_time_unit, size, input->null_count,
                                       input->valid);
    } else if (input->dtype == GDF_STRING_CATEGORY && c_dtype == GDF_STRING) {
      NVCategory *cat = static_cast<NVCategory *>(input->dtype_info.category);
      return cudf::jni::cast_string_cat_to(env, cat, c_dtype, c_time_unit, size, input->null_count,
                                           input->valid);
    } else {
      std::unique_ptr<gdf_column, decltype(free) *> ret(
              static_cast<gdf_column *>(malloc(sizeof(gdf_column))), free);
      info.time_unit = c_time_unit;
      *ret.get() = cudf::cast(*input, c_dtype, info);
      return reinterpret_cast<jlong>(ret.release());
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_replaceNulls(JNIEnv *env, jclass, jlong input_jcol,
                                                              jlong r_int_values, jfloat r_f_value,
                                                              jdouble r_d_value,
                                                              jboolean r_is_valid, int r_dtype) {
  JNI_NULL_CHECK(env, input_jcol, "input column is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_jcol);
    gdf_scalar replacement{};
    cudf::jni::gdf_scalar_init(&replacement, r_int_values, r_f_value, r_d_value, r_is_valid,
                               r_dtype);
    std::unique_ptr<gdf_column, decltype(free) *> result(
        static_cast<gdf_column *>(malloc(sizeof(gdf_column))), free);
    if (result.get() == nullptr) {
      cudf::jni::throw_java_exception(env, "java/lang/OutOfMemoryError",
                                      "Could not allocate native memory");
    }
    *result.get() = cudf::replace_nulls(*input, replacement);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cudf_fill(JNIEnv *env, jclass, jlong input_jcol,
                                                     jlong s_int_values, jfloat s_f_value,
                                                     jdouble s_d_value, jboolean s_is_valid,
                                                     int s_dtype) {
  JNI_NULL_CHECK(env, input_jcol, "input column is null", );
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_jcol);
    gdf_scalar fill_value{};
    cudf::jni::gdf_scalar_init(&fill_value, s_int_values, s_f_value, s_d_value, s_is_valid,
                               s_dtype);

    cudf::fill(input, fill_value, 0, input->size);
  }
  CATCH_STD(env, );
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cudf_reduce(JNIEnv *env, jclass, jlong jcol,
                                                          jint jop, jint jdtype) {
  JNI_NULL_CHECK(env, jcol, "input column is null", 0);
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    cudf::reduction::operators op = static_cast<cudf::reduction::operators>(jop);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_scalar scalar = cudf::reduce(col, op, dtype);
    return cudf::jni::jscalar_from_scalar(env, scalar, col->dtype_info.time_unit);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cudf_getCategoryIndex(JNIEnv *env, jclass, jlong jcol,
                                                                 jbyteArray jstr) {
  JNI_NULL_CHECK(env, jcol, "input column is null", -1);
  JNI_NULL_CHECK(env, jstr, "string data is null", -1);
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    if (col->size <= 0) {
      // it is empty so nothing is in there.
      return -1;
    }
    NVCategory *cat = static_cast<NVCategory *>(col->dtype_info.category);
    JNI_NULL_CHECK(env, cat, "category is null", -1);

    int len = env->GetArrayLength(jstr);
    cudf::jni::check_java_exception(env);
    std::unique_ptr<char[]> str(new char[len + 1]);
    env->GetByteArrayRegion(jstr, 0, len, reinterpret_cast<jbyte *>(str.get()));
    cudf::jni::check_java_exception(env);
    str[len] = '\0'; // NUL-terminate UTF-8 string

    return cat->get_value(str.get());
  }
  CATCH_STD(env, 0);
}
} // extern "C"
