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

#include <numeric>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/reshape.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/rolling.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/substring.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <map_lookup.hpp>

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

namespace {

std::size_t calc_device_memory_size(cudf::column_view const &view) {
  std::size_t total = 0;
  auto row_count = view.size();

  if (view.nullable()) {
    total += cudf::bitmask_allocation_size_bytes(row_count);
  }

  auto dtype = view.type();
  if (cudf::is_fixed_width(dtype)) {
    total += cudf::size_of(dtype) * view.size();
  }

  return std::accumulate(view.child_begin(), view.child_end(), total,
                         [](std::size_t t, cudf::column_view const &v) {
                           return t + calc_device_memory_size(v);
                         });
}

} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_upperStrings(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_upper(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_lowerStrings(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_lower(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceNulls(JNIEnv *env, jclass,
                                                                    jlong j_col, jlong j_scalar) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view col = *reinterpret_cast<cudf::column_view *>(j_col);
    auto val = reinterpret_cast<cudf::scalar *>(j_scalar);
    std::unique_ptr<cudf::column> result = cudf::replace_nulls(col, *val);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseVV(JNIEnv *env, jclass,
                                                                jlong j_pred_vec,
                                                                jlong j_true_vec,
                                                                jlong j_false_vec) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_vec, "false column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view *>(j_true_vec);
    auto false_vec = reinterpret_cast<cudf::column_view *>(j_false_vec);
    std::unique_ptr<cudf::column> result = cudf::copy_if_else(*true_vec, *false_vec, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseVS(JNIEnv *env, jclass,
                                                                jlong j_pred_vec,
                                                                jlong j_true_vec,
                                                                jlong j_false_scalar) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_scalar, "false scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view *>(j_true_vec);
    auto false_scalar = reinterpret_cast<cudf::scalar *>(j_false_scalar);
    std::unique_ptr<cudf::column> result = cudf::copy_if_else(*true_vec, *false_scalar, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseSV(JNIEnv *env, jclass,
                                                                jlong j_pred_vec,
                                                                jlong j_true_scalar,
                                                                jlong j_false_vec) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_scalar, "true scalar is null", 0);
  JNI_NULL_CHECK(env, j_false_vec, "false column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_scalar = reinterpret_cast<cudf::scalar *>(j_true_scalar);
    auto false_vec = reinterpret_cast<cudf::column_view *>(j_false_vec);
    std::unique_ptr<cudf::column> result = cudf::copy_if_else(*true_scalar, *false_vec, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseSS(JNIEnv *env, jclass,
                                                                jlong j_pred_vec,
                                                                jlong j_true_scalar,
                                                                jlong j_false_scalar) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_scalar, "true scalar is null", 0);
  JNI_NULL_CHECK(env, j_false_scalar, "false scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_scalar = reinterpret_cast<cudf::scalar *>(j_true_scalar);
    auto false_scalar = reinterpret_cast<cudf::scalar *>(j_false_scalar);
    std::unique_ptr<cudf::column> result =
        cudf::copy_if_else(*true_scalar, *false_scalar, *pred_vec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_reduce(JNIEnv *env, jclass,
                                                              jlong j_col_view,
                                                              jlong j_agg,
                                                              jint j_dtype, jint scale) {
  JNI_NULL_CHECK(env, j_col_view, "column view is null", 0);
  JNI_NULL_CHECK(env, j_agg, "aggregation is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto col = reinterpret_cast<cudf::column_view *>(j_col_view);
    auto agg = reinterpret_cast<cudf::aggregation *>(j_agg);
    cudf::data_type out_dtype = cudf::jni::make_data_type(j_dtype, scale);

    std::unique_ptr<cudf::scalar> result = cudf::reduce(*col, agg->clone(), out_dtype);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_quantile(JNIEnv *env, jclass clazz,
                                                                jlong input_column,
                                                                jint quantile_method,
                                                                jdoubleArray jquantiles) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jdoubleArray native_quantiles(env, jquantiles);
    std::vector<double> quantiles(native_quantiles.data(),
                                  native_quantiles.data() + native_quantiles.size());
    cudf::column_view *n_input_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::interpolation n_quantile_method = static_cast<cudf::interpolation>(quantile_method);
    std::unique_ptr<cudf::column> result =
        cudf::quantile(*n_input_column, quantiles, n_quantile_method);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_rollingWindow(
    JNIEnv *env, jclass clazz, jlong input_col, jlong default_output_col, 
    jint min_periods, jlong agg_ptr, jint preceding,
    jint following, jlong preceding_col, jlong following_col) {

  JNI_NULL_CHECK(env, input_col, "native handle is null", 0);
  JNI_NULL_CHECK(env, agg_ptr, "aggregation handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_input_col = reinterpret_cast<cudf::column_view *>(input_col);
    cudf::column_view *n_default_output_col =
        reinterpret_cast<cudf::column_view *>(default_output_col);
    cudf::column_view *n_preceding_col = reinterpret_cast<cudf::column_view *>(preceding_col);
    cudf::column_view *n_following_col = reinterpret_cast<cudf::column_view *>(following_col);
    cudf::aggregation * agg = reinterpret_cast<cudf::aggregation *>(agg_ptr);

    std::unique_ptr<cudf::column> ret;
    if (n_default_output_col != nullptr) {
      if (n_preceding_col != nullptr && n_following_col != nullptr) {
        CUDF_FAIL("A default output column is not currently supported with variable length preceding and following");
        //ret = cudf::rolling_window(*n_input_col, *n_default_output_col, 
        //        *n_preceding_col, *n_following_col, min_periods, agg->clone());
      } else {
        ret = cudf::rolling_window(*n_input_col, *n_default_output_col,
                preceding, following, min_periods, agg->clone());
      }

    } else {
      if (n_preceding_col != nullptr && n_following_col != nullptr) {
        ret = cudf::rolling_window(*n_input_col, *n_preceding_col, *n_following_col,
                min_periods, agg->clone());
      } else {
        ret = cudf::rolling_window(*n_input_col, preceding, following, min_periods,
                agg->clone());
      }
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_slice(JNIEnv *env, jclass clazz,
                                                                  jlong input_column,
                                                                  jintArray slice_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, slice_indices, "slice indices are null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::jni::native_jintArray n_slice_indices(env, slice_indices);

    std::vector<cudf::size_type> indices(n_slice_indices.size());
    for (int i = 0; i < n_slice_indices.size(); i++) {
      indices[i] = n_slice_indices[i];
    }

    std::vector<cudf::column_view> result = cudf::slice(*n_column, indices);
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_extractListElement(JNIEnv *env, jclass,
                                                                          jlong column_view,
                                                                          jint index) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::lists_column_view lcv(*cv);

    std::unique_ptr<cudf::column> ret = cudf::lists::extract_list_element(lcv, index);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_stringSplit(JNIEnv *env, jclass,
                                                                        jlong column_view,
                                                                        jlong delimiter) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, delimiter, "string scalar delimiter is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_scalar = reinterpret_cast<cudf::string_scalar *>(delimiter);

    std::unique_ptr<cudf::table> table_result = cudf::strings::split(scv, *ss_scalar);
    return cudf::jni::convert_table_for_return(env, table_result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringSplitRecord(JNIEnv *env, jclass,
                                                                         jlong column_view,
                                                                         jlong delimiter,
                                                                         jint max_split) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, delimiter, "string scalar delimiter is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_scalar = reinterpret_cast<cudf::string_scalar *>(delimiter);

    std::unique_ptr<cudf::column> ret = cudf::strings::split_record(scv, *ss_scalar, max_split);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_split(JNIEnv *env, jclass clazz,
                                                                  jlong input_column,
                                                                  jintArray split_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, split_indices, "split indices are null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.size());
    for (int i = 0; i < n_split_indices.size(); i++) {
      indices[i] = n_split_indices[i];
    }

    std::vector<cudf::column_view> result = cudf::split(*n_column, indices);
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_charLengths(JNIEnv *env, jclass clazz,
                                                                   jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result =
        cudf::strings::count_characters(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_byteCount(JNIEnv *env, jclass clazz,
                                                                 jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result =
        cudf::strings::count_bytes(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_findAndReplaceAll(JNIEnv *env,
                                                                         jclass clazz,
                                                                         jlong old_values_handle,
                                                                         jlong new_values_handle,
                                                                         jlong input_handle) {
  JNI_NULL_CHECK(env, old_values_handle, "values column is null", 0);
  JNI_NULL_CHECK(env, new_values_handle, "replace column is null", 0);
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);

  using cudf::column;
  using cudf::column_view;

  try {
    cudf::jni::auto_set_device(env);
    column_view *input_column = reinterpret_cast<column_view *>(input_handle);
    column_view *old_values_column = reinterpret_cast<column_view *>(old_values_handle);
    column_view *new_values_column = reinterpret_cast<column_view *>(new_values_handle);

    std::unique_ptr<column> result =
        cudf::find_and_replace_all(*input_column, *old_values_column, *new_values_column);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNullNative(JNIEnv *env, jclass,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::is_null(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNotNullNative(JNIEnv *env, jclass,
                                                                       jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::is_valid(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNanNative(JNIEnv *env, jclass,
                                                                   jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::is_nan(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNotNanNative(JNIEnv *env, jclass,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::is_not_nan(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_unaryOperation(JNIEnv *env, jclass,
                                                                      jlong input_ptr,
                                                                      jint int_op) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    cudf::unary_operator op = static_cast<cudf::unary_operator>(int_op);
    std::unique_ptr<cudf::column> ret = cudf::unary_operation(*input, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}


JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_round(JNIEnv *env, jclass,
                                                               jlong input_ptr, jint decimal_places,
                                                               jint rounding_method) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
   try {
     cudf::jni::auto_set_device(env);
     cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
     cudf::rounding_method method = static_cast<cudf::rounding_method>(rounding_method);
     std::unique_ptr<cudf::column> ret = cudf::round(*input, decimal_places, method);
     return reinterpret_cast<jlong>(ret.release());
   }
   CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_year(JNIEnv *env, jclass,
                                                            jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_year(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_month(JNIEnv *env, jclass,
                                                             jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_month(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_day(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_day(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_hour(JNIEnv *env, jclass,
                                                            jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_hour(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_minute(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_minute(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_second(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_second(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_weekDay(JNIEnv *env, jclass,
                                                               jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_weekday(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_lastDayOfMonth(JNIEnv *env, jclass,
                                                                      jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::last_day_of_month(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_dayOfYear(JNIEnv *env, jclass,
                                                                 jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::day_of_year(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_castTo(JNIEnv *env, jclass,
                                                              jlong handle, jint type,
                                                              jint scale) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type n_data_type = cudf::jni::make_data_type(type, scale);
    std::unique_ptr<cudf::column> result;
    if (n_data_type.id() == column->type().id()) {
      std::unique_ptr<cudf::column> copy(new cudf::column(*column));
      return reinterpret_cast<jlong>(copy.release());
    }
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
        case cudf::type_id::UINT8:
        case cudf::type_id::INT16:
        case cudf::type_id::UINT16:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
          result = cudf::strings::from_integers(*column);
          break;
        default: JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
      }
    } else if (column->type().id() == cudf::type_id::STRING) {
      switch (n_data_type.id()) {
        case cudf::type_id::BOOL8:
          result = cudf::strings::to_booleans(*column);
          break;
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64:
          result = cudf::strings::to_floats(*column, n_data_type);
          break;
        case cudf::type_id::INT8:
        case cudf::type_id::UINT8:
        case cudf::type_id::INT16:
        case cudf::type_id::UINT16:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
          result = cudf::strings::to_integers(*column, n_data_type);
          break;
        default: JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
      }
    } else if (cudf::is_timestamp(n_data_type) && cudf::is_numeric(column->type())) {
      // This is a temporary workaround to allow Java to cast from integral types into a timestamp
      // without forcing an intermediate duration column to be manifested.  Ultimately this style of
      // "reinterpret" casting will be supported via https://github.com/rapidsai/cudf/pull/5358
      if (n_data_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
        if (column->type().id() != cudf::type_id::INT32) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Numeric cast to TIMESTAMP_DAYS requires INT32", 0);
        }
      } else {
        if (column->type().id() != cudf::type_id::INT64) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Numeric cast to non-day timestamp requires INT64", 0);
        }
      }
      cudf::data_type duration_type = cudf::jni::timestamp_to_duration(n_data_type);
      cudf::column_view duration_view = cudf::column_view(duration_type,
                                                          column->size(),
                                                          column->head(),
                                                          column->null_mask(),
                                                          column->null_count());
      result = cudf::cast(duration_view, n_data_type);
    } else if (cudf::is_timestamp(column->type()) && cudf::is_numeric(n_data_type)) {
      // This is a temporary workaround to allow Java to cast from timestamp types to integral types
      // without forcing an intermediate duration column to be manifested.  Ultimately this style of
      // "reinterpret" casting will be supported via https://github.com/rapidsai/cudf/pull/5358
      cudf::data_type duration_type = cudf::jni::timestamp_to_duration(column->type());
      cudf::column_view duration_view = cudf::column_view(duration_type,
                                                          column->size(),
                                                          column->head(),
                                                          column->null_mask(),
                                                          column->null_count());
      result = cudf::cast(duration_view, n_data_type);
    } else {
      result = cudf::cast(*column, n_data_type);
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_byteListCast(JNIEnv *env, jobject j_object,
                                                                    jlong handle,
                                                                    jboolean endianness_config) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::flip_endianness config(static_cast<cudf::flip_endianness>(endianness_config));
    std::unique_ptr<cudf::column> result = byte_cast(*column, config);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringTimestampToTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jint time_unit, jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, formatObj, "format is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring format(env, formatObj);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_timestamps(
        strings_column, cudf::data_type(static_cast<cudf::type_id>(time_unit)), format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isTimestamp(JNIEnv *env, jclass,
                                                                   jlong handle, jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, formatObj, "format is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring format(env, formatObj);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::is_timestamp(
        strings_column, format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_timestampToStringTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jstring j_format) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, j_format, "format is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring format(env, j_format);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);

    std::unique_ptr<cudf::column> result = cudf::strings::from_timestamps(*column, format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ColumnView_containsScalar(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong j_view_handle,
                                                                         jlong j_scalar_handle) {
  JNI_NULL_CHECK(env, j_view_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_scalar_handle, "scalar needle is null", false);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::scalar *scalar = reinterpret_cast<cudf::scalar *>(j_scalar_handle);

    return cudf::contains(*column_view, *scalar);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_containsVector(JNIEnv *env,
                                                                      jobject j_object,
                                                                      jlong j_haystack_handle,
                                                                      jlong j_needle_handle) {
  JNI_NULL_CHECK(env, j_haystack_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_needle_handle, "needle vector is null", false);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *haystack = reinterpret_cast<cudf::column_view *>(j_haystack_handle);
    cudf::column_view *needle = reinterpret_cast<cudf::column_view *>(j_needle_handle);

    std::unique_ptr<cudf::column> result = std::move(cudf::contains(*haystack, *needle));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_transform(JNIEnv *env, jobject j_object,
                                                                 jlong handle, jstring j_udf,
                                                                 jboolean j_is_ptx) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::jni::native_jstring n_j_udf(env, j_udf);
    std::string n_udf(n_j_udf.get());
    std::unique_ptr<cudf::column> result =
        cudf::transform(*column, n_udf, cudf::data_type(cudf::type_id::INT32), j_is_ptx);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringStartWith(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong j_view_handle,
                                                                       jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar *comp_scalar = reinterpret_cast<cudf::string_scalar *>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::starts_with(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringEndWith(JNIEnv *env,
                                                                     jobject j_object,
                                                                     jlong j_view_handle,
                                                                     jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar *comp_scalar = reinterpret_cast<cudf::string_scalar *>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::ends_with(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringContains(JNIEnv *env,
                                                                      jobject j_object,
                                                                      jlong j_view_handle,
                                                                      jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar *comp_scalar = reinterpret_cast<cudf::string_scalar *>(comp_string);

    std::unique_ptr<cudf::column> result = cudf::strings::contains(strings_column, *comp_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_matchesRe(JNIEnv *env, jobject j_object,
                                                                 jlong j_view_handle,
                                                                 jstring patternObj) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, patternObj, "pattern is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::jni::native_jstring pattern(env, patternObj);

    std::unique_ptr<cudf::column> result = cudf::strings::matches_re(strings_column, pattern.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_containsRe(JNIEnv *env, jobject j_object,
                                                                  jlong j_view_handle,
                                                                  jstring patternObj) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, patternObj, "pattern is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::jni::native_jstring pattern(env, patternObj);

    std::unique_ptr<cudf::column> result =
        cudf::strings::contains_re(strings_column, pattern.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringConcatenation(
    JNIEnv *env, jobject j_object, jlongArray column_handles, jlong separator, jlong narep) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  JNI_NULL_CHECK(env, separator, "separator string scalar object is null", 0);
  JNI_NULL_CHECK(env, narep, "narep string scalar object is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::string_scalar *separator_scalar = reinterpret_cast<cudf::string_scalar *>(separator);
    cudf::string_scalar *narep_scalar = reinterpret_cast<cudf::string_scalar *>(narep);
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });
    cudf::table_view *string_columns = new cudf::table_view(column_views);

    std::unique_ptr<cudf::column> result =
        cudf::strings::concatenate(*string_columns, *separator_scalar, *narep_scalar);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_binaryOpVV(JNIEnv *env, jclass,
                                                                  jlong lhs_view, jlong rhs_view,
                                                                  jint int_op, jint out_dtype,
                                                                  jint scale) {
  JNI_NULL_CHECK(env, lhs_view, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_view, "rhs is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto lhs = reinterpret_cast<cudf::column_view *>(lhs_view);
    auto rhs = reinterpret_cast<cudf::column_view *>(rhs_view);

    cudf::data_type n_data_type = cudf::jni::make_data_type(out_dtype, scale);
    cudf::binary_operator op = static_cast<cudf::binary_operator>(int_op);
    std::unique_ptr<cudf::column> result = cudf::binary_operation(
        *lhs, *rhs, op, n_data_type);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_binaryOpVS(JNIEnv *env, jclass,
                                                                  jlong lhs_view, jlong rhs_ptr,
                                                                  jint int_op, jint out_dtype,
                                                                  jint scale) {
  JNI_NULL_CHECK(env, lhs_view, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto lhs = reinterpret_cast<cudf::column_view *>(lhs_view);
    cudf::scalar *rhs = reinterpret_cast<cudf::scalar *>(rhs_ptr);
    cudf::data_type n_data_type = cudf::jni::make_data_type(out_dtype, scale);

    cudf::binary_operator op = static_cast<cudf::binary_operator>(int_op);
    std::unique_ptr<cudf::column> result = cudf::binary_operation(
        *lhs, *rhs, op, n_data_type);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_substring(JNIEnv *env, jclass,
                                                                 jlong column_view, jint start,
                                                                 jint end) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);

    std::unique_ptr<cudf::column> result =
        (end == -1 ? cudf::strings::slice_strings(scv, start) :
                     cudf::strings::slice_strings(scv, start, end));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_substringColumn(JNIEnv *env, jclass,
                                                                       jlong column_view,
                                                                       jlong start_column,
                                                                       jlong end_column) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, start_column, "column is null", 0);
  JNI_NULL_CHECK(env, end_column, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::column_view *sc = reinterpret_cast<cudf::column_view *>(start_column);
    cudf::column_view *ec = reinterpret_cast<cudf::column_view *>(end_column);

    std::unique_ptr<cudf::column> result = cudf::strings::slice_strings(scv, *sc, *ec);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_substringLocate(JNIEnv *env, jclass,
                                                                       jlong column_view,
                                                                       jlong substring,
                                                                       jint start, jint end) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, substring, "target string scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_scalar = reinterpret_cast<cudf::string_scalar *>(substring);

    std::unique_ptr<cudf::column> result = cudf::strings::find(scv, *ss_scalar, start, end);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringReplace(JNIEnv *env, jclass,
                                                                     jlong column_view,
                                                                     jlong target,
                                                                     jlong replace) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, target, "target string scalar is null", 0);
  JNI_NULL_CHECK(env, replace, "replace string scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_target = reinterpret_cast<cudf::string_scalar *>(target);
    cudf::string_scalar *ss_replace = reinterpret_cast<cudf::string_scalar *>(replace);

    std::unique_ptr<cudf::column> result = cudf::strings::replace(scv, *ss_target, *ss_replace);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_mapLookup(JNIEnv *env, jclass,
                                                                 jlong map_column_view,
                                                                 jlong lookup_key) {
  JNI_NULL_CHECK(env, map_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key, "target string scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(map_column_view);
    cudf::string_scalar *ss_key = reinterpret_cast<cudf::string_scalar *>(lookup_key);

    std::unique_ptr<cudf::column> result = cudf::jni::map_lookup(*cv, *ss_key);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringReplaceWithBackrefs(JNIEnv *env,
                                                                                 jclass,
                                                                                 jlong column_view,
                                                                                 jstring patternObj,
                                                                                 jstring replaceObj) {

  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, patternObj, "pattern string is null", 0);
  JNI_NULL_CHECK(env, replaceObj, "replace string is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::jni::native_jstring ss_pattern(env, patternObj);
    cudf::jni::native_jstring ss_replace(env, replaceObj);

    std::unique_ptr<cudf::column> result = cudf::strings::replace_with_backrefs(
        scv, ss_pattern.get(), ss_replace.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_zfill(JNIEnv *env, jclass, jlong column_view,
                                                             jint j_width) {

  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::size_type width = reinterpret_cast<cudf::size_type>(j_width);

    std::unique_ptr<cudf::column> result = cudf::strings::zfill(scv, width);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_pad(JNIEnv *env,
                                                           jclass,
                                                           jlong column_view,
                                                           jint j_width,
                                                           jint j_side,
                                                           jstring fill_char) {

  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, fill_char, "fill_char is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::size_type width = reinterpret_cast<cudf::size_type>(j_width);
    cudf::strings::pad_side side = static_cast<cudf::strings::pad_side>(j_side);
    cudf::jni::native_jstring ss_fill(env, fill_char);

    std::unique_ptr<cudf::column> result = cudf::strings::pad(scv, width, side, ss_fill.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringStrip(JNIEnv *env, jclass,
                                                                   jlong column_view,
                                                                   jint strip_type,
                                                                   jlong to_strip) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, to_strip, "to_strip scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::strings::strip_type s_striptype = static_cast<cudf::strings::strip_type>(strip_type);
    cudf::string_scalar *ss_tostrip = reinterpret_cast<cudf::string_scalar *>(to_strip);

    std::unique_ptr<cudf::column> result = cudf::strings::strip(scv, s_striptype, *ss_tostrip);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_extractRe(JNIEnv *env,
                                                                      jobject j_object,
                                                                      jlong j_view_handle,
                                                                      jstring patternObj) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", nullptr);
  JNI_NULL_CHECK(env, patternObj, "pattern is null", nullptr);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::jni::native_jstring pattern(env, patternObj);

    std::unique_ptr<cudf::table> table_result =
        cudf::strings::extract(strings_column, pattern.get());
    return cudf::jni::convert_table_for_return(env, table_result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_normalizeNANsAndZeros(JNIEnv *env,
                                                                             jclass clazz,
                                                                             jlong input_column) {
  using cudf::column_view;

  JNI_NULL_CHECK(env, input_column, "Input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    return reinterpret_cast<jlong>(
        cudf::normalize_nans_and_zeros(*reinterpret_cast<column_view *>(input_column)).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_bitwiseMergeAndSetValidity(JNIEnv *env,
                                                                                  jobject j_object,
                                                                                  jlong base_column,
                                                                                  jlongArray column_handles,
                                                                                  jint bin_op) {
  JNI_NULL_CHECK(env, base_column, "base column native handle is null", 0);
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  try {
    cudf::column_view *original_column = reinterpret_cast<cudf::column_view *>(base_column);
    std::unique_ptr<cudf::column> copy(new cudf::column(*original_column));
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);

    if (n_cudf_columns.size() == 0) {
      rmm::device_buffer null_mask{0};
      copy->set_null_mask(null_mask);
      return reinterpret_cast<jlong>(copy.release());
    }

    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });
    cudf::table_view *input_table = new cudf::table_view(column_views);

    cudf::binary_operator op = static_cast<cudf::binary_operator>(bin_op);
    if(op == cudf::binary_operator::BITWISE_AND) {
      copy->set_null_mask(cudf::bitmask_and(*input_table));
    }

    return reinterpret_cast<jlong>(copy.release());
  }
  CATCH_STD(env, 0);
}

////////
// Native cudf::column_view life cycle and metadata access methods. Life cycle methods
// should typically only be called from the CudfColumn inner class.
////////

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_makeCudfColumnView(JNIEnv *env,
                                                                          jclass, jint j_type,
                                                                          jint scale, jlong j_data,
                                                                          jlong j_data_size,
                                                                          jlong j_offset,
    jlong j_valid, jint j_null_count, jint size, jlongArray j_children) {

  try {
    using cudf::column_view;
    cudf::jni::auto_set_device(env);
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    cudf::data_type n_data_type = cudf::jni::make_data_type(j_type, scale);

    std::unique_ptr<cudf::column_view> ret;
    void *data = reinterpret_cast<void *>(j_data);
    cudf::bitmask_type *valid = reinterpret_cast<cudf::bitmask_type *>(j_valid);
    if (valid == nullptr) {
      j_null_count = 0;
    }

    if (n_type == cudf::type_id::STRING) {
      if (size == 0) {
        ret.reset(new cudf::column_view(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0));
      } else {
        JNI_NULL_CHECK(env, j_offset, "offset is null", 0);
        // This must be kept in sync with how string columns are created
        // offsets are always the first child
        // data is the second child

        cudf::size_type *offsets = reinterpret_cast<cudf::size_type *>(j_offset);
        cudf::column_view offsets_column(cudf::data_type{cudf::type_id::INT32}, size + 1, offsets);
        cudf::column_view data_column(cudf::data_type{cudf::type_id::INT8}, j_data_size, data);
        ret.reset(new cudf::column_view(cudf::data_type{cudf::type_id::STRING}, size, nullptr,
                                        valid, j_null_count, 0, {offsets_column, data_column}));
      }
    } else if (n_type == cudf::type_id::LIST) {
      JNI_NULL_CHECK(env, j_children, "children of a list are null", 0);
      cudf::jni::native_jpointerArray<cudf::column_view> children(env, j_children);
      JNI_ARG_CHECK(env, (children.size() == 1), "LIST children size is not 1", 0);
      cudf::size_type offsets_size = 0;
      cudf::size_type *offsets = nullptr;
      if (size != 0) {
        JNI_NULL_CHECK(env, j_offset, "offset is null", 0);
        offsets_size = size + 1;
        offsets = reinterpret_cast<cudf::size_type *>(j_offset);
      }
      cudf::column_view offsets_column(cudf::data_type{cudf::type_id::INT32}, offsets_size, offsets);
      ret.reset(new cudf::column_view(cudf::data_type{cudf::type_id::LIST}, size, nullptr, valid,
        j_null_count, 0, {offsets_column, *children[0]}));
   } else if (n_type == cudf::type_id::STRUCT) {
     JNI_NULL_CHECK(env, j_children, "children of a list are null", 0);
     cudf::jni::native_jpointerArray<cudf::column_view> children(env, j_children);
     std::vector<column_view> children_vector(children.size());
     for (int i = 0; i < children.size(); i++) {
       children_vector[i] = *children[i];
     }
     ret.reset(new cudf::column_view(cudf::data_type{cudf::type_id::STRUCT}, size, nullptr, valid,
       j_null_count, 0, children_vector));
   } else {
     ret.reset(new cudf::column_view(n_data_type, size, data, valid, j_null_count));
    }

    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeTypeId(JNIEnv *env,
                                                                      jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->type().id());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeTypeScale(JNIEnv *env,
                                                                         jclass,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return column->type().scale();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeRowCount(JNIEnv *env,
                                                                        jclass,
                                                                        jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->size());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeNullCount(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnView_deleteColumnView(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong handle) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    delete view;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeDataAddress(JNIEnv *env, jclass,
                                                                            jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    jlong result = 0;
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::type_id::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view data_view = view.chars();
        result = reinterpret_cast<jlong>(data_view.data<char>());
      }
    } else if(column->type().id() != cudf::type_id::LIST && column->type().id() != cudf::type_id::STRUCT) {
      result = reinterpret_cast<jlong>(column->data<char>());
    }
    return result;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeDataLength(JNIEnv *env, jclass,
                                                                           jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    jlong result = 0;
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::type_id::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view data_view = view.chars();
        result = data_view.size();
      }
    } else if(column->type().id() != cudf::type_id::LIST && column->type().id() != cudf::type_id::STRUCT) {
      result = cudf::size_of(column->type()) * column->size();
    }
    return result;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeNumChildren(JNIEnv *env,
                                                                           jobject j_object,
                                                                           jlong handle) {

    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    try {
      cudf::jni::auto_set_device(env);
      cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
      // Strings has children(offsets and chars) but not a nested child() we care about here.
      if (column->type().id() == cudf::type_id::STRING) {
        return 0;
      } else if (column->type().id() == cudf::type_id::LIST) {
        // first child is always offsets in lists which we do not want to count here
        return static_cast<jint>(column->num_children() - 1);
      } else if (column->type().id() == cudf::type_id::STRUCT) {
        return static_cast<jint>(column->num_children());
      } else {
        return 0;
      }
    }
    CATCH_STD(env, 0);

}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getChildCvPointer(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle, jint child_index) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    try {
      cudf::jni::auto_set_device(env);
      cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
      if (column->type().id() == cudf::type_id::LIST) {
        std::unique_ptr<cudf::lists_column_view> view = std::make_unique<cudf::lists_column_view>(*column);
        // first child is always offsets which we do not want to get from this call
        std::unique_ptr<cudf::column_view> next_view = std::make_unique<cudf::column_view>(column->child(1 + child_index));
        return reinterpret_cast<jlong>(next_view.release());
      } else {
        std::unique_ptr<cudf::structs_column_view> view = std::make_unique<cudf::structs_column_view>(*column);
        std::unique_ptr<cudf::column_view> next_view = std::make_unique<cudf::column_view>(column->child(child_index));
        return reinterpret_cast<jlong>(next_view.release());
      }
    }
    CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeOffsetsAddress(JNIEnv *env, jclass,
                                                                               jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    jlong result = 0;
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::type_id::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        result = reinterpret_cast<jlong>(offsets_view.data<char>());
      }
    } else if (column->type().id() == cudf::type_id::LIST) {
      if (column->size() > 0) {
        cudf::lists_column_view view = cudf::lists_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        result = reinterpret_cast<jlong>(offsets_view.data<char>());
      }
    }
    return result;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeOffsetsLength(JNIEnv *env, jclass,
                                                                              jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    jlong result = 0;
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    if (column->type().id() == cudf::type_id::STRING) {
      if (column->size() > 0) {
        cudf::strings_column_view view = cudf::strings_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        result = sizeof(int) * offsets_view.size();
      }
    } else if (column->type().id() == cudf::type_id::LIST) {
      if (column->size() > 0) {
        cudf::lists_column_view view = cudf::lists_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        result = sizeof(int) * offsets_view.size();
      }
    }
    return result;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeValidityAddress(JNIEnv *env,
                                                                                jclass,
                                                                                jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return reinterpret_cast<jlong>(column->null_mask());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeValidityLength(JNIEnv *env,
                                                                               jclass,
                                                                               jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    jlong result = 0;
    if (column->null_mask() != nullptr) {
      result = cudf::bitmask_allocation_size_bytes(column->size());
    }
    return result;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeValidPointerSize(JNIEnv *env,
                                                                                 jobject j_object,
                                                                                 jint size) {
  try {
    cudf::jni::auto_set_device(env);
    return static_cast<jlong>(cudf::bitmask_allocation_size_bytes(size));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getDeviceMemorySize(JNIEnv *env, jclass,
                                                                           jlong handle) {
    JNI_NULL_CHECK(env, handle, "native handle is null", 0);
    try {
      cudf::jni::auto_set_device(env);
      auto view = reinterpret_cast<cudf::column_view const *>(handle);
      return calc_device_memory_size(*view);
    }
    CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_clamper(JNIEnv *env, jobject j_object,
                                                               jlong handle, jlong j_lo_scalar,
                                                               jlong j_lo_replace_scalar,
                                                               jlong j_hi_scalar,
                                                               jlong j_hi_replace_scalar) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)
  JNI_NULL_CHECK(env, j_lo_scalar, "lo scalar is null", 0)
  JNI_NULL_CHECK(env, j_lo_replace_scalar, "lo scalar replace value is null", 0)
  JNI_NULL_CHECK(env, j_hi_scalar, "lo scalar is null", 0)
  JNI_NULL_CHECK(env, j_hi_replace_scalar, "lo scalar replace value is null", 0)
  using cudf::clamp;
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(handle);
    cudf::scalar *lo_scalar = reinterpret_cast<cudf::scalar *>(j_lo_scalar);
    cudf::scalar *lo_replace_scalar = reinterpret_cast<cudf::scalar *>(j_lo_replace_scalar);
    cudf::scalar *hi_scalar = reinterpret_cast<cudf::scalar *>(j_hi_scalar);
    cudf::scalar *hi_replace_scalar = reinterpret_cast<cudf::scalar *>(j_hi_replace_scalar);

    std::unique_ptr<cudf::column> result =
        clamp(*column_view, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_title(JNIEnv *env, jobject j_object,
                                                             jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> result = cudf::strings::title(*view);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_nansToNulls(JNIEnv *env, jobject j_object,
                                                                   jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    // get a new null mask by setting all the nans to null
    std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> pair =
        cudf::nans_to_nulls(*view);
    // create a column_view which is a no-copy wrapper around the original column without the null
    // mask
    std::unique_ptr<cudf::column_view> copy_view(
        new cudf::column_view(view->type(), view->size(), view->data<char>()));
    // create a column by deep copying the copy_view
    std::unique_ptr<cudf::column> copy(new cudf::column(*copy_view));
    // set the null mask with nans set to null
    copy->set_null_mask(std::move(*pair.first), pair.second);
    return reinterpret_cast<jlong>(copy.release());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isFloat(JNIEnv *env, jobject j_object,
                                                               jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> result = cudf::strings::is_float(*view);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isInteger(JNIEnv *env, jobject j_object,
                                                                 jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> result = cudf::strings::is_integer(*view);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_copyColumnViewToCV(JNIEnv *env, jobject j_object,
                                                                          jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::column> ret;
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    ret.reset(reinterpret_cast<cudf::column *>(new cudf::column(*view)));
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0)
}
} // extern "C"
