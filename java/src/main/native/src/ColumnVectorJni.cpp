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

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/rolling.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/substring.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>

#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_upperStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_upper(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_lowerStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_lower(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatenate(JNIEnv *env, jclass clazz,
                                                                     jlongArray column_handles) {
  JNI_NULL_CHECK(env, column_handles, "input columns are null", 0);
  using cudf::column;
  using cudf::column_view;
  try {
    cudf::jni::native_jpointerArray<column_view> columns(env, column_handles);
    std::vector<column_view> columns_vector(columns.size());
    for (int i = 0; i < columns.size(); ++i) {
      JNI_NULL_CHECK(env, columns[i], "column to concat is null", 0);
      columns_vector[i] = *columns[i];
    }
    std::unique_ptr<column> result = cudf::concatenate(columns_vector);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_fromScalar(JNIEnv *env, jclass,
    jlong j_scalar, jint row_count) {
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    auto scalar_val = reinterpret_cast<cudf::scalar const*>(j_scalar);
    auto dtype = scalar_val->type();
    cudf::mask_state mask_state = scalar_val->is_valid() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL;
    std::unique_ptr<cudf::column> col;
    if (row_count == 0) {
      col = cudf::make_empty_column(dtype);
    } else if (cudf::is_fixed_width(dtype)) {
      col = cudf::make_fixed_width_column(dtype, row_count, mask_state);
      auto mut_view = col->mutable_view();
      cudf::experimental::fill_in_place(mut_view, 0, row_count, *scalar_val);
    } else if (dtype.id() == cudf::type_id::STRING) {
      // create a string column of all empty strings to fill (cheapest string column to create)
      auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::INT32}, row_count + 1, cudf::mask_state::UNALLOCATED);
      auto data = cudf::make_empty_column(cudf::data_type{cudf::INT8});
      auto mask_buffer = cudf::create_null_mask(row_count, cudf::mask_state::UNALLOCATED);
      auto str_col = cudf::make_strings_column(row_count, std::move(offsets), std::move(data), 0, std::move(mask_buffer));

      col = cudf::experimental::fill(str_col->view(), 0, row_count, *scalar_val);
    } else {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
    }
    return reinterpret_cast<jlong>(col.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_replaceNulls(JNIEnv *env, jclass,
    jlong j_col, jlong j_scalar) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::column_view col = *reinterpret_cast<cudf::column_view*>(j_col);
    auto val = reinterpret_cast<cudf::scalar*>(j_scalar);
    std::unique_ptr<cudf::column> result = cudf::experimental::replace_nulls(col, *val);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_ifElseVV(JNIEnv *env, jclass,
    jlong j_pred_vec, jlong j_true_vec, jlong j_false_vec) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_vec, "false column is null", 0);
  try {
    auto pred_vec = reinterpret_cast<cudf::column_view*>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view*>(j_true_vec);
    auto false_vec = reinterpret_cast<cudf::column_view*>(j_false_vec);
    std::unique_ptr<cudf::column> result = cudf::experimental::copy_if_else(*true_vec, *false_vec, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_ifElseVS(JNIEnv *env, jclass,
    jlong j_pred_vec, jlong j_true_vec, jlong j_false_scalar) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_scalar, "false scalar is null", 0);
  try {
    auto pred_vec = reinterpret_cast<cudf::column_view*>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view*>(j_true_vec);
    auto false_scalar = reinterpret_cast<cudf::scalar*>(j_false_scalar);
    std::unique_ptr<cudf::column> result = cudf::experimental::copy_if_else(*true_vec, *false_scalar, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_ifElseSV(JNIEnv *env, jclass,
    jlong j_pred_vec, jlong j_true_scalar, jlong j_false_vec) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_scalar, "true scalar is null", 0);
  JNI_NULL_CHECK(env, j_false_vec, "false column is null", 0);
  try {
    auto pred_vec = reinterpret_cast<cudf::column_view*>(j_pred_vec);
    auto true_scalar = reinterpret_cast<cudf::scalar*>(j_true_scalar);
    auto false_vec = reinterpret_cast<cudf::column_view*>(j_false_vec);
    std::unique_ptr<cudf::column> result = cudf::experimental::copy_if_else(*true_scalar, *false_vec, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_ifElseSS(JNIEnv *env, jclass,
    jlong j_pred_vec, jlong j_true_scalar, jlong j_false_scalar) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_scalar, "true scalar is null", 0);
  JNI_NULL_CHECK(env, j_false_scalar, "false scalar is null", 0);
  try {
    auto pred_vec = reinterpret_cast<cudf::column_view*>(j_pred_vec);
    auto true_scalar = reinterpret_cast<cudf::scalar*>(j_true_scalar);
    auto false_scalar = reinterpret_cast<cudf::scalar*>(j_false_scalar);
    std::unique_ptr<cudf::column> result = cudf::experimental::copy_if_else(*true_scalar, *false_scalar, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_reduce(JNIEnv *env, jclass,
    jlong j_col_view, jint agg_type, jint j_dtype) {
  JNI_NULL_CHECK(env, j_col_view, "column view is null", 0);
  try {
    auto col = reinterpret_cast<cudf::column_view*>(j_col_view);
    auto agg = cudf::jni::map_jni_aggregation(agg_type);
    cudf::data_type out_dtype{static_cast<cudf::type_id>(j_dtype)};
    std::unique_ptr<cudf::scalar> result = cudf::experimental::reduce(*col, agg, out_dtype);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_quantile(JNIEnv *env, jclass clazz,
                                                                         jlong input_column,
                                                                         jint quantile_method,
                                                                         jdouble quantile) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    cudf::column_view *n_input_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::experimental::interpolation n_quantile_method =
        static_cast<cudf::experimental::interpolation>(quantile_method);
    std::unique_ptr<cudf::scalar> result =
        cudf::experimental::quantile(*n_input_column, quantile, n_quantile_method);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_rollingWindow(
    JNIEnv *env, jclass clazz, jlong input_col, jint min_periods, jint agg_type,
    jint preceding, jint following, jlong preceding_col, jlong following_col) {

  JNI_NULL_CHECK(env, input_col, "native handle is null", 0);
  try {
    cudf::column_view *n_input_col = reinterpret_cast<cudf::column_view *>(input_col);
    cudf::column_view *n_preceding_col = reinterpret_cast<cudf::column_view *>(preceding_col);
    cudf::column_view *n_following_col = reinterpret_cast<cudf::column_view *>(following_col);
    auto agg = cudf::jni::map_jni_aggregation(agg_type);

    std::unique_ptr<cudf::column> ret;
    if (n_preceding_col != nullptr && n_following_col != nullptr) {
      ret = cudf::experimental::rolling_window(*n_input_col, *n_preceding_col, *n_following_col,
              min_periods, agg);
    } else {
      ret = cudf::experimental::rolling_window(*n_input_col, preceding, following,
              min_periods, agg);
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_slice(JNIEnv *env, jclass clazz,
                                                                        jlong input_column,
                                                                        jintArray slice_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, slice_indices, "slice indices are null", 0);

  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::jni::native_jintArray n_slice_indices(env, slice_indices);

    std::vector<cudf::size_type> indices(n_slice_indices.size());
    for (int i = 0; i < n_slice_indices.size(); i++) {
      indices[i] = n_slice_indices[i];
    }

    std::vector<cudf::column_view> result = cudf::experimental::slice(*n_column, indices);
    cudf::jni::native_jlongArray n_result(env, result.size());
    std::vector<std::unique_ptr<cudf::column>> column_result(result.size());
    for (int i = 0; i < result.size(); i++) {
      column_result[i].reset(new cudf::column(result[i]));
      n_result[i] = reinterpret_cast<jlong>(column_result[i].get());
    }
    for (int i = 0; i < result.size(); i++) {
      column_result[i].release();
    }
    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_split(JNIEnv *env, jclass clazz,
                                                                    jlong input_column,
                                                                    jintArray split_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, split_indices, "split indices are null", 0);

  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.size());
    for (int i = 0; i < n_split_indices.size(); i++) {
      indices[i] = n_split_indices[i];
    }

    std::vector<cudf::column_view> result = cudf::experimental::split(*n_column, indices);
    cudf::jni::native_jlongArray n_result(env, result.size());
    std::vector<std::unique_ptr<cudf::column>> column_result(result.size());
    for (int i = 0; i < result.size(); i++) {
      column_result[i].reset(new cudf::column(result[i]));
      n_result[i] = reinterpret_cast<jlong>(column_result[i].get());
    }
    for (int i = 0; i < result.size(); i++) {
      column_result[i].release();
    }
    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_lengths(JNIEnv *env, jclass clazz,
                                                                 jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result = cudf::strings::count_characters(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_byteCount(JNIEnv *env, jclass clazz,
                                                                   jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result = cudf::strings::count_bytes(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_findAndReplaceAll(JNIEnv *env,
                                                                           jclass clazz,
                                                                           jlong old_values_handle,
                                                                           jlong new_values_handle,
                                                                           jlong input_handle) {
  JNI_NULL_CHECK(env, old_values_handle, "values column is null", 0);
  JNI_NULL_CHECK(env, new_values_handle, "replace column is null", 0);
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);

  using cudf::column_view;
  using cudf::column;

  try {
    column_view *input_column = reinterpret_cast<column_view *>(input_handle);
    column_view *old_values_column = reinterpret_cast<column_view *>(old_values_handle);
    column_view *new_values_column = reinterpret_cast<column_view *>(new_values_handle);

    std::unique_ptr<column> result =
        cudf::experimental::find_and_replace_all(*input_column, *old_values_column, *new_values_column);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNullNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_null(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNotNullNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_valid(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNanNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_nan(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNotNanNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_not_nan(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_unaryOperation(JNIEnv *env, jclass,
        jlong input_ptr, jint int_op) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    cudf::experimental::unary_op op = static_cast<cudf::experimental::unary_op>(int_op);
    std::unique_ptr<cudf::column> ret = cudf::experimental::unary_operation(*input, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_year(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_year(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_month(JNIEnv *env, jclass,
                                                               jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_month(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_day(JNIEnv *env, jclass,
                                                             jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_day(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_hour(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_hour(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_minute(JNIEnv *env, jclass,
                                                                jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_minute(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_second(JNIEnv *env, jclass,
                                                                jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_second(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_castTo(JNIEnv *env,
                                                                jobject j_object,
                                                                jlong handle, jint type) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type n_data_type(static_cast<cudf::type_id>(type));
    std::unique_ptr<cudf::column> result;
    if (n_data_type.id() == cudf::type_id::STRING) {
        switch (column->type().id()) {
        case cudf::type_id::BOOL8:
            result = cudf::strings::from_booleans(*column);
            break;
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64:
            result = cudf::strings::from_floats(*column);
            break;
        case cudf::type_id::INT8:
        case cudf::type_id::INT16:
        case cudf::type_id::INT32:
        case cudf::type_id::INT64:
            result = cudf::strings::from_integers(*column);
            break;
        default:
            JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
        }
    } else if (column->type().id() == cudf::type_id::STRING) {
        switch (n_data_type.id()) {
        case cudf::type_id::BOOL8:
            result = cudf::strings::to_booleans(*column);
            break;
        case cudf::type_id::FLOAT32:
            result = cudf::strings::to_floats(*column, cudf::data_type{cudf::type_id::FLOAT32});
            break;
        case cudf::type_id::FLOAT64:
            result = cudf::strings::to_floats(*column, cudf::data_type{cudf::type_id::FLOAT64});
            break;
        case cudf::type_id::INT8:
            result = cudf::strings::to_integers(*column, cudf::data_type{cudf::type_id::INT8});
            break;
        case cudf::type_id::INT16:
            result = cudf::strings::to_integers(*column, cudf::data_type{cudf::type_id::INT16});
            break;
        case cudf::type_id::INT32:
            result = cudf::strings::to_integers(*column, cudf::data_type{cudf::type_id::INT32});
            break;
        case cudf::type_id::INT64:
            result = cudf::strings::to_integers(*column, cudf::data_type{cudf::type_id::INT64});
            break;
        default:
            JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
        }
    } else {
        result = cudf::experimental::cast(*column, n_data_type);
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringTimestampToTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jint time_unit, jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, formatObj, "format is null", 0);

  try {
    cudf::jni::native_jstring format(env, formatObj);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_timestamps(strings_column, cudf::data_type(static_cast<cudf::type_id>(time_unit)), format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_timestampToStringTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jstring j_format) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, j_format, "format is null", 0);

  try {
    cudf::jni::native_jstring format(env, j_format);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);

    std::unique_ptr<cudf::column> result = cudf::strings::from_timestamps(*column, format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ColumnVector_containsScalar(JNIEnv *env, jobject j_object,
                                                                 jlong j_view_handle, jlong j_scalar_handle) {
  JNI_NULL_CHECK(env, j_view_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_scalar_handle, "scalar needle is null", false);
  try {
    cudf::column_view* column_view = reinterpret_cast<cudf::column_view*>(j_view_handle);
    cudf::scalar* scalar = reinterpret_cast<cudf::scalar*>(j_scalar_handle);

    return cudf::experimental::contains(*column_view, *scalar);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_containsVector(JNIEnv *env, jobject j_object,
                                                                 jlong j_haystack_handle, jlong j_needle_handle) {
  JNI_NULL_CHECK(env, j_haystack_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_needle_handle, "needle vector is null", false);
  try {
    cudf::column_view* haystack = reinterpret_cast<cudf::column_view*>(j_haystack_handle);
    cudf::column_view* needle = reinterpret_cast<cudf::column_view*>(j_needle_handle);

    std::unique_ptr<cudf::column> result = std::move(cudf::experimental::contains(*haystack, *needle));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_transform(JNIEnv *env, jobject j_object,
                                                                 jlong handle, jstring j_udf,
                                                                 jboolean j_is_ptx) {
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::jni::native_jstring n_j_udf(env, j_udf);
    std::string n_udf(n_j_udf.get());
    std::unique_ptr<cudf::column> result = cudf::experimental::transform(
        *column, n_udf, cudf::data_type(cudf::INT32), j_is_ptx);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringStartWith(JNIEnv *env, jobject j_object,
                                                                    jlong j_view_handle, jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::column_view* column_view = reinterpret_cast<cudf::column_view*>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar* comp_scalar = reinterpret_cast<cudf::string_scalar*>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::starts_with(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringEndWith(JNIEnv *env, jobject j_object,
                                                                    jlong j_view_handle, jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::column_view* column_view = reinterpret_cast<cudf::column_view*>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar* comp_scalar = reinterpret_cast<cudf::string_scalar*>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::ends_with(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringContains(JNIEnv *env, jobject j_object,
                                                                    jlong j_view_handle, jlong comp_string) {
    JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
    JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::column_view* column_view = reinterpret_cast<cudf::column_view*>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar* comp_scalar = reinterpret_cast<cudf::string_scalar*>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::contains(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringConcatenation(JNIEnv *env, jobject j_object,
                                                                  jlongArray column_handles, jlong separator,
                                                                  jlong narep) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  JNI_NULL_CHECK(env, separator, "separator string scalar object is null", 0);
  JNI_NULL_CHECK(env, narep, "narep string scalar object is null", 0);
  try {
    cudf::string_scalar* separator_scalar = reinterpret_cast<cudf::string_scalar*>(separator);
    cudf::string_scalar* narep_scalar = reinterpret_cast<cudf::string_scalar*>(narep);
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(), std::back_inserter(column_views),
      [](auto const& p_column) { return *p_column;}
    );
    cudf::table_view* string_columns = new cudf::table_view(column_views);

    std::unique_ptr<cudf::column> result = cudf::strings::concatenate(*string_columns, *separator_scalar, *narep_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_binaryOpVV(JNIEnv *env, jclass, jlong lhs_view,
                                                               jlong rhs_view, jint int_op,
                                                               jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_view, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_view, "rhs is null", 0);
  try {
    auto lhs = reinterpret_cast<cudf::column_view *>(lhs_view);
    auto rhs = reinterpret_cast<cudf::column_view *>(rhs_view);

    cudf::experimental::binary_operator op = static_cast<cudf::experimental::binary_operator>(int_op);
    std::unique_ptr<cudf::column> result = cudf::experimental::binary_operation(*lhs, *rhs, op, cudf::data_type(static_cast<cudf::type_id>(out_dtype)));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_binaryOpVS(JNIEnv *env, jclass, jlong lhs_view,
                                                               jlong rhs_ptr, jint int_op,
                                                               jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_view, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    auto lhs = reinterpret_cast<cudf::column_view *>(lhs_view);
    cudf::scalar *rhs = reinterpret_cast<cudf::scalar *>(rhs_ptr);

    cudf::experimental::binary_operator op = static_cast<cudf::experimental::binary_operator>(int_op);
    std::unique_ptr<cudf::column> result = cudf::experimental::binary_operation(*lhs, *rhs, op, cudf::data_type(static_cast<cudf::type_id>(out_dtype)));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_substring(JNIEnv *env, jclass, jlong column_view,
                                                                jint start, jint end) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::column_view* cv = reinterpret_cast<cudf::column_view*>(column_view);
    cudf::strings_column_view scv(*cv);

    std::unique_ptr<cudf::column> result = (end == -1 ? cudf::strings::slice_strings(scv, start): cudf::strings::slice_strings(scv, start, end));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_substringColumn(JNIEnv *env, jclass, jlong column_view,
                                                                      jlong start_column, jlong end_column) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, start_column, "column is null", 0);
  JNI_NULL_CHECK(env, end_column, "column is null", 0);
  try {
    cudf::column_view* cv = reinterpret_cast<cudf::column_view*>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::column_view *sc = reinterpret_cast<cudf::column_view *>(start_column);
    cudf::column_view *ec = reinterpret_cast<cudf::column_view *>(end_column);

    std::unique_ptr<cudf::column> result = cudf::strings::slice_strings(scv, *sc, *ec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_substringLocate(JNIEnv *env, jclass, jlong column_view,
                                                                      jlong substring, jint start, jint end) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, substring, "target string scalar is null", 0);
  try {
    cudf::column_view* cv = reinterpret_cast<cudf::column_view*>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar* ss_scalar = reinterpret_cast<cudf::string_scalar*>(substring);
    
    std::unique_ptr<cudf::column> result = cudf::strings::find(scv, *ss_scalar, start, end);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

////////
// Native cudf::column_view life cycle and metadata access methods. Life cycle methods
// should typically only be called from the CudfColumn inner class.
////////


JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeCudfColumnView(
    JNIEnv *env, jobject j_object, jint j_type, jlong j_data, jlong j_data_size,
    jlong j_offset, jlong j_valid, jint j_null_count, jint size) {

  JNI_ARG_CHECK(env, (size != 0), "size is 0", 0);

  cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
  cudf::data_type n_data_type(n_type);

  std::unique_ptr<cudf::column_view> ret;
  void * data = reinterpret_cast<void *>(j_data);
  cudf::bitmask_type * valid = reinterpret_cast<cudf::bitmask_type *>(j_valid);
  if (valid == nullptr) {
    j_null_count = 0;
  }

  if (n_type == cudf::STRING) {
    JNI_NULL_CHECK(env, j_offset, "offset is null", 0);
    // This must be kept in sync with how string columns are created
    // offsets are always the first child
    // data is the second child

    cudf::size_type * offsets = reinterpret_cast<cudf::size_type *>(j_offset);
    cudf::column_view offsets_column(cudf::data_type{cudf::INT32}, size + 1, offsets);
    cudf::column_view data_column(cudf::data_type{cudf::INT8}, j_data_size, data);
    ret.reset(new cudf::column_view(cudf::data_type{cudf::STRING}, size, nullptr,
                valid, j_null_count, 0, {offsets_column, data_column}));
  } else {
    ret.reset(new cudf::column_view(n_data_type, size, data, valid, j_null_count));
  }

  return reinterpret_cast<jlong>(ret.release());
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeTypeId(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return column->type().id();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeRowCount(JNIEnv *env,
                                                                        jobject j_object,
                                                                        jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->size());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeNullCount(JNIEnv *env,
                                                                             jobject j_object,
                                                                             jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_deleteColumnView(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong handle) {
  try {
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    delete view;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeDataPointer(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jlong handle) {
  try {
    cudf::jni::native_jlongArray ret(env, 2);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view data_view = view.chars();
        ret[0] = reinterpret_cast<jlong>(data_view.data<char>());
        ret[1] = data_view.size();
      } else {
        ret[0] = 0;
        ret[1] = 0;
      }
    } else {
      ret[0] = reinterpret_cast<jlong>(column->data<char>());
      ret[1] = cudf::size_of(column->type()) * column->size();
    }
    return ret.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeOffsetsPointer(JNIEnv *env,
                                                                                    jobject j_object,
                                                                                    jlong handle) {
  try {
    cudf::jni::native_jlongArray ret(env, 2);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        ret[0] = reinterpret_cast<jlong>(offsets_view.data<char>());
        ret[1] = sizeof(int) * offsets_view.size();
      } else {
        ret[0] = 0;
        ret[1] = 0;
      }
    } else {
      ret[0] = 0;
      ret[1] = 0;
    }
    return ret.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeValidPointer(JNIEnv *env,
                                                                                  jobject j_object,
                                                                                  jlong handle) {
  try {
    cudf::jni::native_jlongArray ret(env, 2);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    ret[0] = reinterpret_cast<jlong>(column->null_mask());
    if (ret[0] != 0) {
      ret[1] = cudf::bitmask_allocation_size_bytes(column->size());
    } else {
      ret[1] = 0;
    }
    return ret.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeValidPointerSize(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jint size) {
  try {
    return static_cast<jlong>(cudf::bitmask_allocation_size_bytes(size));
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
  delete reinterpret_cast<cudf::column *>(handle);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_setNativeNullCountColumn(JNIEnv *env,
                                                                               jobject j_object,
                                                                               jlong handle,
                                                                               jint null_count) {
  JNI_NULL_CHECK(env, handle, "native handle is null", );
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    column->set_null_count(null_count);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeColumnView(JNIEnv *env,
                                                                           jobject j_object,
                                                                           jlong handle) {
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    std::unique_ptr<cudf::column_view> view(new cudf::column_view());
    *view.get() = column->view();
    return reinterpret_cast<jlong>(view.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeEmptyCudfColumn(
    JNIEnv *env, jobject j_object, jint j_type) {

  try {
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    cudf::data_type n_data_type(n_type);
    std::unique_ptr<cudf::column> column(
        cudf::make_empty_column(n_data_type));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeNumericCudfColumn(
    JNIEnv *env, jobject j_object, jint j_type, jint j_size, jint j_mask_state) {

  JNI_ARG_CHECK(env, (j_size != 0), "size is 0", 0);

  try {
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    cudf::data_type n_data_type(n_type);
    cudf::size_type n_size = static_cast<cudf::size_type>(j_size);
    cudf::mask_state n_mask_state = static_cast<cudf::mask_state>(j_mask_state);
    std::unique_ptr<cudf::column> column(
        cudf::make_numeric_column(n_data_type, n_size, n_mask_state));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeTimestampCudfColumn(
    JNIEnv *env, jobject j_object, jint j_type, jint j_size, jint j_mask_state) {

  JNI_NULL_CHECK(env, j_type, "type id is null", 0);
  JNI_NULL_CHECK(env, j_size, "size is null", 0);

  try {
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    std::unique_ptr<cudf::data_type> n_data_type(new cudf::data_type(n_type));
    cudf::size_type n_size = static_cast<cudf::size_type>(j_size);
    cudf::mask_state n_mask_state = static_cast<cudf::mask_state>(j_mask_state);
    std::unique_ptr<cudf::column> column(
        cudf::make_timestamp_column(*n_data_type.get(), n_size, n_mask_state));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeStringCudfColumnHostSide(
    JNIEnv *env, jobject j_object, jlong j_char_data, jlong j_offset_data, jlong j_valid_data,
    jint j_null_count, jint size) {

  JNI_ARG_CHECK(env, (size != 0), "size is 0", 0);
  JNI_NULL_CHECK(env, j_char_data, "char data is null", 0);
  JNI_NULL_CHECK(env, j_offset_data, "offset is null", 0);

  try {
    cudf::size_type *host_offsets = reinterpret_cast<cudf::size_type *>(j_offset_data);
    char *n_char_data = reinterpret_cast<char *>(j_char_data);
    cudf::size_type n_data_size = host_offsets[size];
    cudf::bitmask_type *n_validity = reinterpret_cast<cudf::bitmask_type *>(j_valid_data);

    if (n_validity == nullptr) {
      j_null_count = 0;
    }

    std::unique_ptr<cudf::column> offsets = cudf::make_numeric_column(
            cudf::data_type{cudf::INT32},
            size + 1,
            cudf::mask_state::UNALLOCATED);
    auto offsets_view = offsets->mutable_view();
    JNI_CUDA_TRY(env, 0, cudaMemcpyAsync(
                offsets_view.data<int32_t>(),
                host_offsets,
                (size + 1) * sizeof(int32_t),
                cudaMemcpyHostToDevice));

    std::unique_ptr<cudf::column> data = cudf::make_numeric_column(
            cudf::data_type{cudf::INT8},
            n_data_size,
            cudf::mask_state::UNALLOCATED);
    auto data_view = data->mutable_view();
    JNI_CUDA_TRY(env, 0, cudaMemcpyAsync(
                data_view.data<int8_t>(),
                n_char_data,
                n_data_size,
                cudaMemcpyHostToDevice));

    std::unique_ptr<cudf::column> column;
    if (j_null_count == 0) {
      column = cudf::make_strings_column(size, std::move(offsets), std::move(data), j_null_count, {});
    } else {
      cudf::size_type bytes = (cudf::word_index(size) + 1) * sizeof(cudf::bitmask_type);
      rmm::device_buffer dev_validity(bytes);
      JNI_CUDA_TRY(env, 0, cudaMemcpyAsync(
                  dev_validity.data(),
                  n_validity,
                  bytes,
                  cudaMemcpyHostToDevice));

      column = cudf::make_strings_column(size, std::move(offsets), std::move(data),
              j_null_count, std::move(dev_validity));
    }

    JNI_CUDA_TRY(env, 0, cudaStreamSynchronize(0));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNativeNullCountColumn(JNIEnv *env,
                                                                               jobject j_object,
                                                                               jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
