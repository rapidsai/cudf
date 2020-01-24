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

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include "jni_utils.hpp"

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Scalar_closeScalar(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    cudf::scalar* s = reinterpret_cast<cudf::scalar*>(scalar_handle);
    delete s;
  }
  CATCH_STD(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Scalar_isScalarValid(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    cudf::scalar* s = reinterpret_cast<cudf::scalar*>(scalar_handle);
    return static_cast<jboolean>(s->is_valid());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jbyte JNICALL Java_ai_rapids_cudf_Scalar_getByte(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<int8_t>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jbyte>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jshort JNICALL Java_ai_rapids_cudf_Scalar_getShort(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<int16_t>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jshort>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Scalar_getInt(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<int32_t>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jint>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_getLong(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<int64_t>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jlong>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jfloat JNICALL Java_ai_rapids_cudf_Scalar_getFloat(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<float>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jfloat>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jdouble JNICALL Java_ai_rapids_cudf_Scalar_getDouble(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    using ScalarType = cudf::experimental::scalar_type_t<double>;
    auto s = reinterpret_cast<ScalarType*>(scalar_handle);
    return static_cast<jdouble>(s->value());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jbyteArray JNICALL Java_ai_rapids_cudf_Scalar_getUTF8(JNIEnv* env, jclass, jlong scalar_handle) {
  try {
    auto s = reinterpret_cast<cudf::string_scalar*>(scalar_handle);
    std::string val{s->to_string()};
    if (val.size() > 0x7FFFFFFF) {
      cudf::jni::throw_java_exception(env, "java/lang/IllegalArgumentException", "string scalar too large");
    }
    cudf::jni::native_jbyteArray jbytes{env, reinterpret_cast<jbyte const*>(val.data()), static_cast<int>(val.size())};
    return jbytes.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeBool8Scalar(JNIEnv* env, jclass,
    jboolean value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::BOOL8));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int8_t>;
      int8_t val = value ? 1 : 0;
      static_cast<ScalarType*>(s.get())->set_value(val);
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt8Scalar(JNIEnv* env, jclass,
    jbyte value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::INT8));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int8_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int8_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt16Scalar(JNIEnv* env, jclass,
    jshort value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::INT16));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int16_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int16_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt32Scalar(JNIEnv* env, jclass,
    jint value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeInt64Scalar(JNIEnv* env, jclass,
    jlong value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::INT64));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeFloat32Scalar(JNIEnv* env, jclass,
    jfloat value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::FLOAT32));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<float>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<float>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeFloat64Scalar(JNIEnv* env, jclass,
    jdouble value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_numeric_scalar(cudf::data_type(cudf::FLOAT64));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<double>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<double>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeStringScalar(JNIEnv* env, jclass,
    jbyteArray value, jboolean is_valid) {
  try {
    std::string strval;
    if (is_valid) {
      cudf::jni::native_jbyteArray jbytes{env, value};
      strval.assign(reinterpret_cast<char*>(jbytes.data()), jbytes.size());
    }

    auto s = new cudf::string_scalar{strval, is_valid};
    return reinterpret_cast<jlong>(s);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeTimestampDaysScalar(JNIEnv* env, jclass,
    jint value, jboolean is_valid) {
  try {
    std::unique_ptr<cudf::scalar> s = cudf::make_timestamp_scalar(cudf::data_type(cudf::TIMESTAMP_DAYS));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int32_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int32_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_makeTimestampTimeScalar(JNIEnv* env, jclass,
    jint jdtype_id, jlong value, jboolean is_valid) {
  try {
    auto dtype_id = static_cast<cudf::type_id>(jdtype_id);
    std::unique_ptr<cudf::scalar> s = cudf::make_timestamp_scalar(cudf::data_type(dtype_id));
    s->set_valid(is_valid);
    if (is_valid) {
      using ScalarType = cudf::experimental::scalar_type_t<int64_t>;
      static_cast<ScalarType*>(s.get())->set_value(static_cast<int64_t>(value));
    }
    return reinterpret_cast<jlong>(s.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Scalar_binaryOpSV(JNIEnv *env, jclass, jlong lhs_ptr,
                                                               jlong rhs_view, jint int_op,
                                                               jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_view, "rhs is null", 0);
  try {
    cudf::scalar *lhs = reinterpret_cast<cudf::scalar *>(lhs_ptr);
    auto rhs = reinterpret_cast<cudf::column_view *>(rhs_view);

    cudf::experimental::binary_operator op = static_cast<cudf::experimental::binary_operator>(int_op);
    std::unique_ptr<cudf::column> result = cudf::experimental::binary_operation(*lhs, *rhs, op, cudf::data_type(static_cast<cudf::type_id>(out_dtype)));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
