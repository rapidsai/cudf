/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <arrow/api.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_sequence(JNIEnv *env, jclass,
                                                                  jlong j_initial_val, jlong j_step,
                                                                  jint row_count) {
  JNI_NULL_CHECK(env, j_initial_val, "scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto initial_val = reinterpret_cast<cudf::scalar const *>(j_initial_val);
    auto step = reinterpret_cast<cudf::scalar const *>(j_step);
    std::unique_ptr<cudf::column> col;
    if (step) {
      col = cudf::sequence(row_count, *initial_val, *step);
    } else {
      col = cudf::sequence(row_count, *initial_val);
    }
    return reinterpret_cast<jlong>(col.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_sequences(JNIEnv *env, jclass,
                                                                   jlong j_start_handle,
                                                                   jlong j_size_handle,
                                                                   jlong j_step_handle) {
  JNI_NULL_CHECK(env, j_start_handle, "start is null", 0);
  JNI_NULL_CHECK(env, j_size_handle, "size is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto start = reinterpret_cast<cudf::column_view const *>(j_start_handle);
    auto size = reinterpret_cast<cudf::column_view const *>(j_size_handle);
    auto step = reinterpret_cast<cudf::column_view const *>(j_step_handle);
    std::unique_ptr<cudf::column> col;
    if (step) {
      col = cudf::lists::sequences(*start, *step, *size);
    } else {
      col = cudf::lists::sequences(*start, *size);
    }
    return reinterpret_cast<jlong>(col.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_fromArrow(
    JNIEnv *env, jclass, jint j_type, jlong j_col_length, jlong j_null_count, jobject j_data_obj,
    jobject j_validity_obj, jobject j_offsets_obj) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    // not all the buffers are used for all types
    void const *data_address = 0;
    int data_length = 0;
    if (j_data_obj != 0) {
      data_address = env->GetDirectBufferAddress(j_data_obj);
      data_length = env->GetDirectBufferCapacity(j_data_obj);
    }
    void const *validity_address = 0;
    int validity_length = 0;
    if (j_validity_obj != 0) {
      validity_address = env->GetDirectBufferAddress(j_validity_obj);
      validity_length = env->GetDirectBufferCapacity(j_validity_obj);
    }
    void const *offsets_address = 0;
    int offsets_length = 0;
    if (j_offsets_obj != 0) {
      offsets_address = env->GetDirectBufferAddress(j_offsets_obj);
      offsets_length = env->GetDirectBufferCapacity(j_offsets_obj);
    }
    auto data_buffer =
        arrow::Buffer::Wrap(static_cast<const char *>(data_address), static_cast<int>(data_length));
    auto null_buffer = arrow::Buffer::Wrap(static_cast<const char *>(validity_address),
                                           static_cast<int>(validity_length));
    auto offsets_buffer = arrow::Buffer::Wrap(static_cast<const char *>(offsets_address),
                                              static_cast<int>(offsets_length));

    std::shared_ptr<arrow::Array> arrow_array;
    switch (n_type) {
      case cudf::type_id::DECIMAL32:
        JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "Don't support converting DECIMAL32 yet",
                      0);
        break;
      case cudf::type_id::DECIMAL64:
        JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "Don't support converting DECIMAL64 yet",
                      0);
        break;
      case cudf::type_id::STRUCT:
        JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "Don't support converting STRUCT yet", 0);
        break;
      case cudf::type_id::LIST:
        JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "Don't support converting LIST yet", 0);
        break;
      case cudf::type_id::DICTIONARY32:
        JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS,
                      "Don't support converting DICTIONARY32 yet", 0);
        break;
      case cudf::type_id::STRING:
        arrow_array = std::make_shared<arrow::StringArray>(j_col_length, offsets_buffer,
                                                           data_buffer, null_buffer, j_null_count);
        break;
      default:
        // this handles the primitive types
        arrow_array = cudf::detail::to_arrow_array(n_type, j_col_length, data_buffer, null_buffer,
                                                   j_null_count);
    }
    auto name_and_type = arrow::field("col", arrow_array->type());
    std::vector<std::shared_ptr<arrow::Field>> fields = {name_and_type};
    std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);
    auto arrow_table =
        arrow::Table::Make(schema, std::vector<std::shared_ptr<arrow::Array>>{arrow_array});
    std::unique_ptr<cudf::table> table_result = cudf::from_arrow(*(arrow_table));
    std::vector<std::unique_ptr<cudf::column>> retCols = table_result->release();
    if (retCols.size() != 1) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Must result in one column", 0);
    }
    return reinterpret_cast<jlong>(retCols[0].release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringConcatenation(
    JNIEnv *env, jclass, jlongArray column_handles, jlong separator, jlong narep,
    jboolean separate_nulls) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  JNI_NULL_CHECK(env, separator, "separator string scalar object is null", 0);
  JNI_NULL_CHECK(env, narep, "narep string scalar object is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const auto &separator_scalar = *reinterpret_cast<cudf::string_scalar *>(separator);
    const auto &narep_scalar = *reinterpret_cast<cudf::string_scalar *>(narep);
    auto null_policy = separate_nulls ? cudf::strings::separator_on_nulls::YES :
                                        cudf::strings::separator_on_nulls::NO;

    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });

    std::unique_ptr<cudf::column> result = cudf::strings::concatenate(
        cudf::table_view(column_views), separator_scalar, narep_scalar, null_policy);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringConcatenationSepCol(
    JNIEnv *env, jclass, jlongArray column_handles, jlong sep_handle, jlong separator_narep,
    jlong col_narep, jboolean separate_nulls) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  JNI_NULL_CHECK(env, sep_handle, "separator column handle is null", 0);
  JNI_NULL_CHECK(env, separator_narep, "separator narep string scalar object is null", 0);
  JNI_NULL_CHECK(env, col_narep, "column narep string scalar object is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const auto &separator_narep_scalar = *reinterpret_cast<cudf::string_scalar *>(separator_narep);
    const auto &col_narep_scalar = *reinterpret_cast<cudf::string_scalar *>(col_narep);
    auto null_policy = separate_nulls ? cudf::strings::separator_on_nulls::YES :
                                        cudf::strings::separator_on_nulls::NO;

    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });

    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(sep_handle);
    cudf::strings_column_view strings_column(*column);
    std::unique_ptr<cudf::column> result =
        cudf::strings::concatenate(cudf::table_view(column_views), strings_column,
                                   separator_narep_scalar, col_narep_scalar, null_policy);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatListByRow(JNIEnv *env, jclass,
                                                                         jlongArray column_handles,
                                                                         jboolean ignore_null) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto null_policy = ignore_null ? cudf::lists::concatenate_null_policy::IGNORE :
                                     cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW;

    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });

    std::unique_ptr<cudf::column> result =
        cudf::lists::concatenate_rows(cudf::table_view(column_views), null_policy);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeList(JNIEnv *env, jobject j_object,
                                                                  jlongArray handles, jlong j_type,
                                                                  jint scale, jlong row_count) {
  using ScalarType = cudf::scalar_type_t<cudf::size_type>;
  JNI_NULL_CHECK(env, handles, "native view handles are null", 0)
  try {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::column> ret;
    cudf::jni::native_jpointerArray<cudf::column_view> children(env, handles);
    std::vector<cudf::column_view> children_vector(children.size());
    for (int i = 0; i < children.size(); i++) {
      children_vector[i] = *children[i];
    }
    auto zero = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
    zero->set_valid_async(true);
    static_cast<ScalarType *>(zero.get())->set_value(0);

    if (children.size() == 0) {
      // special case because cudf::interleave_columns does not support no columns
      auto offsets = cudf::make_column_from_scalar(*zero, row_count + 1);
      cudf::data_type n_data_type = cudf::jni::make_data_type(j_type, scale);
      auto empty_col = cudf::make_empty_column(n_data_type);
      ret = cudf::make_lists_column(row_count, std::move(offsets), std::move(empty_col), 0,
                                    rmm::device_buffer());
    } else {
      auto count = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
      count->set_valid_async(true);
      static_cast<ScalarType *>(count.get())->set_value(children.size());

      std::unique_ptr<cudf::column> offsets = cudf::sequence(row_count + 1, *zero, *count);
      auto data_col = cudf::interleave_columns(cudf::table_view(children_vector));
      ret = cudf::make_lists_column(row_count, std::move(offsets), std::move(data_col), 0,
                                    rmm::device_buffer());
    }

    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeListFromOffsets(
    JNIEnv *env, jobject j_object, jlong child_handle, jlong offsets_handle, jlong row_count) {
  JNI_NULL_CHECK(env, child_handle, "child_handle is null", 0)
  JNI_NULL_CHECK(env, offsets_handle, "offsets_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const *child_cv = reinterpret_cast<cudf::column_view const *>(child_handle);
    auto const *offsets_cv = reinterpret_cast<cudf::column_view const *>(offsets_handle);
    CUDF_EXPECTS(offsets_cv->type().id() == cudf::type_id::INT32,
                 "Input offsets does not have type INT32.");

    auto result = cudf::make_lists_column(static_cast<cudf::size_type>(row_count),
                                          std::make_unique<cudf::column>(*offsets_cv),
                                          std::make_unique<cudf::column>(*child_cv), 0, {});
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_fromScalar(JNIEnv *env, jclass,
                                                                    jlong j_scalar,
                                                                    jint row_count) {
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto scalar_val = reinterpret_cast<cudf::scalar const *>(j_scalar);
    std::unique_ptr<cudf::column> col;
    if (scalar_val->type().id() == cudf::type_id::STRING) {
      // Tests fail when using the cudf implementation, complaining no child for string column.
      // So here take care of the String type itself.
      // create a string column of all empty strings to fill (cheapest string column to create)
      auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, row_count + 1,
                                               cudf::mask_state::UNALLOCATED);
      auto data = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8});
      auto mask_buffer = cudf::create_null_mask(row_count, cudf::mask_state::UNALLOCATED);
      auto str_col = cudf::make_strings_column(row_count, std::move(offsets), std::move(data), 0,
                                               std::move(mask_buffer));

      col = cudf::fill(str_col->view(), 0, row_count, *scalar_val);
    } else if (scalar_val->type().id() == cudf::type_id::STRUCT && row_count == 0) {
      // Specialize the creation of empty struct column, since libcudf doesn't support it.
      auto struct_scalar = reinterpret_cast<cudf::struct_scalar const *>(j_scalar);
      auto children = cudf::empty_like(struct_scalar->view())->release();
      auto mask_buffer = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
      col = cudf::make_structs_column(0, std::move(children), 0, std::move(mask_buffer));
    } else {
      col = cudf::make_column_from_scalar(*scalar_val, row_count);
    }
    return reinterpret_cast<jlong>(col.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatenate(JNIEnv *env, jclass clazz,
                                                                     jlongArray column_handles) {
  JNI_NULL_CHECK(env, column_handles, "input columns are null", 0);
  using cudf::column;
  using cudf::column_view;
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<column_view> columns(env, column_handles);
    std::vector<column_view> columns_vector(columns.size());
    for (int i = 0; i < columns.size(); ++i) {
      JNI_NULL_CHECK(env, columns[i], "column to concat is null", 0);
      columns_vector[i] = *columns[i];
    }
    std::unique_ptr<column> result;
    if (columns_vector[0].type().id() == cudf::type_id::LIST) {
      result = cudf::lists::detail::concatenate(columns_vector);
    } else {
      result = cudf::concatenate(columns_vector);
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_hash(JNIEnv *env, jobject j_object,
                                                              jlongArray column_handles,
                                                              jint hash_function_id, jint seed) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  try {
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });
    cudf::table_view input_table{column_views};

    std::unique_ptr<cudf::column> result =
        cudf::hash(input_table, static_cast<cudf::hash_id>(hash_function_id), seed);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

////////
// Native methods specific to cudf::column. These either take or return a cudf::column
// instead of a cudf::column_view so they need to be used with caution. These should
// only be called from the CudfColumn child class.
////////

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_deleteCudfColumn(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "column handle is null", );
  try {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::column *>(handle);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_setNativeNullCountColumn(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jlong handle,
                                                                                 jint null_count) {
  JNI_NULL_CHECK(env, handle, "native handle is null", );
  try {
    cudf::jni::auto_set_device(env);
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    column->set_null_count(null_count);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeColumnView(JNIEnv *env,
                                                                             jobject j_object,
                                                                             jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    std::unique_ptr<cudf::column_view> view(new cudf::column_view());
    *view.get() = column->view();
    return reinterpret_cast<jlong>(view.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeEmptyCudfColumn(JNIEnv *env, jclass,
                                                                             jint j_type,
                                                                             jint scale) {

  try {
    cudf::jni::auto_set_device(env);
    cudf::data_type n_data_type = cudf::jni::make_data_type(j_type, scale);

    std::unique_ptr<cudf::column> column(cudf::make_empty_column(n_data_type));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeNullCountColumn(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}
} // extern "C"
