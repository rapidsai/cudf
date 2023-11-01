/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "ColumnViewJni.hpp"
#include <numeric>

#include <jni.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/rolling.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"
#include "maps_column_view.hpp"

using cudf::jni::ptr_as_jlong;
using cudf::jni::release_as_jlong;

namespace {

std::size_t pad_size(std::size_t size, bool const should_pad_for_cpu) {
  if (should_pad_for_cpu) {
    constexpr std::size_t ALIGN = sizeof(std::max_align_t);
    return (size + (ALIGN - 1)) & ~(ALIGN - 1);
  } else {
    return size;
  }
}

std::size_t calc_device_memory_size(cudf::column_view const &view, bool const pad_for_cpu) {
  std::size_t total = 0;
  auto row_count = view.size();

  if (view.nullable()) {
    total += pad_size(cudf::bitmask_allocation_size_bytes(row_count), pad_for_cpu);
  }

  auto dtype = view.type();
  if (cudf::is_fixed_width(dtype)) {
    total += pad_size(cudf::size_of(dtype) * view.size(), pad_for_cpu);
  }

  return std::accumulate(view.child_begin(), view.child_end(), total,
                         [pad_for_cpu](std::size_t t, cudf::column_view const &v) {
                           return t + calc_device_memory_size(v, pad_for_cpu);
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
    return release_as_jlong(cudf::strings::to_upper(strings_column));
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
    return release_as_jlong(cudf::strings::to_lower(strings_column));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceNullsScalar(JNIEnv *env, jclass,
                                                                          jlong j_col,
                                                                          jlong j_scalar) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view col = *reinterpret_cast<cudf::column_view *>(j_col);
    auto val = reinterpret_cast<cudf::scalar *>(j_scalar);
    return release_as_jlong(cudf::replace_nulls(col, *val));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceNullsColumn(JNIEnv *env, jclass,
                                                                          jlong j_col,
                                                                          jlong j_replace_col) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  JNI_NULL_CHECK(env, j_replace_col, "replacement column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto col = reinterpret_cast<cudf::column_view *>(j_col);
    auto replacements = reinterpret_cast<cudf::column_view *>(j_replace_col);
    return release_as_jlong(cudf::replace_nulls(*col, *replacements));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceNullsPolicy(JNIEnv *env, jclass,
                                                                          jlong j_col,
                                                                          jboolean is_preceding) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view col = *reinterpret_cast<cudf::column_view *>(j_col);
    return release_as_jlong(cudf::replace_nulls(
        col, is_preceding ? cudf::replace_policy::PRECEDING : cudf::replace_policy::FOLLOWING));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_distinctCount(JNIEnv *env, jclass,
                                                                    jlong j_col,
                                                                    jboolean nulls_included) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view col = *reinterpret_cast<cudf::column_view *>(j_col);

    return cudf::distinct_count(
        col, nulls_included ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
        cudf::nan_policy::NAN_IS_VALID);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseVV(JNIEnv *env, jclass,
                                                                jlong j_pred_vec, jlong j_true_vec,
                                                                jlong j_false_vec) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_vec, "false column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view *>(j_true_vec);
    auto false_vec = reinterpret_cast<cudf::column_view *>(j_false_vec);
    return release_as_jlong(cudf::copy_if_else(*true_vec, *false_vec, *pred_vec));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_ifElseVS(JNIEnv *env, jclass,
                                                                jlong j_pred_vec, jlong j_true_vec,
                                                                jlong j_false_scalar) {
  JNI_NULL_CHECK(env, j_pred_vec, "predicate column is null", 0);
  JNI_NULL_CHECK(env, j_true_vec, "true column is null", 0);
  JNI_NULL_CHECK(env, j_false_scalar, "false scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto pred_vec = reinterpret_cast<cudf::column_view *>(j_pred_vec);
    auto true_vec = reinterpret_cast<cudf::column_view *>(j_true_vec);
    auto false_scalar = reinterpret_cast<cudf::scalar *>(j_false_scalar);
    return release_as_jlong(cudf::copy_if_else(*true_vec, *false_scalar, *pred_vec));
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
    return release_as_jlong(cudf::copy_if_else(*true_scalar, *false_vec, *pred_vec));
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
    return release_as_jlong(cudf::copy_if_else(*true_scalar, *false_scalar, *pred_vec));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getElement(JNIEnv *env, jclass, jlong from,
                                                                  jint index) {
  JNI_NULL_CHECK(env, from, "from column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto from_vec = reinterpret_cast<cudf::column_view *>(from);
    return release_as_jlong(cudf::get_element(*from_vec, index));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_reduce(JNIEnv *env, jclass, jlong j_col_view,
                                                              jlong j_agg, jint j_dtype,
                                                              jint scale) {
  JNI_NULL_CHECK(env, j_col_view, "column view is null", 0);
  JNI_NULL_CHECK(env, j_agg, "aggregation is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto col = reinterpret_cast<cudf::column_view *>(j_col_view);
    auto agg = reinterpret_cast<cudf::aggregation *>(j_agg);
    cudf::data_type out_dtype = cudf::jni::make_data_type(j_dtype, scale);
    return release_as_jlong(
        cudf::reduce(*col, *dynamic_cast<cudf::reduce_aggregation *>(agg), out_dtype));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_segmentedReduce(
    JNIEnv *env, jclass, jlong j_data_view, jlong j_offsets_view, jlong j_agg,
    jboolean include_nulls, jint j_dtype, jint scale) {
  JNI_NULL_CHECK(env, j_data_view, "data column view is null", 0);
  JNI_NULL_CHECK(env, j_offsets_view, "offsets column view is null", 0);
  JNI_NULL_CHECK(env, j_agg, "aggregation is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto data = reinterpret_cast<cudf::column_view *>(j_data_view);
    auto offsets = reinterpret_cast<cudf::column_view *>(j_offsets_view);
    auto agg = reinterpret_cast<cudf::aggregation *>(j_agg);
    auto s_agg = dynamic_cast<cudf::segmented_reduce_aggregation *>(agg);
    JNI_ARG_CHECK(env, s_agg != nullptr, "agg is not a cudf::segmented_reduce_aggregation", 0)
    auto null_policy = include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    cudf::data_type out_dtype = cudf::jni::make_data_type(j_dtype, scale);
    return release_as_jlong(
        cudf::segmented_reduce(*data, *offsets, *s_agg, out_dtype, null_policy));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_segmentedGather(
    JNIEnv *env, jclass, jlong source_column, jlong gather_map_list, jboolean nullify_out_bounds) {
  JNI_NULL_CHECK(env, source_column, "source column view is null", 0);
  JNI_NULL_CHECK(env, gather_map_list, "gather map is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const &src_col =
        cudf::lists_column_view(*reinterpret_cast<cudf::column_view *>(source_column));
    auto const &gather_map =
        cudf::lists_column_view(*reinterpret_cast<cudf::column_view *>(gather_map_list));
    auto out_bounds_policy = nullify_out_bounds ? cudf::out_of_bounds_policy::NULLIFY :
                                                  cudf::out_of_bounds_policy::DONT_CHECK;
    return release_as_jlong(cudf::lists::segmented_gather(src_col, gather_map, out_bounds_policy));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_scan(JNIEnv *env, jclass, jlong j_col_view,
                                                            jlong j_agg, jboolean is_inclusive,
                                                            jboolean include_nulls) {
  JNI_NULL_CHECK(env, j_col_view, "column view is null", 0);
  JNI_NULL_CHECK(env, j_agg, "aggregation is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto col = reinterpret_cast<cudf::column_view *>(j_col_view);
    auto agg = reinterpret_cast<cudf::aggregation *>(j_agg);
    auto scan_type = is_inclusive ? cudf::scan_type::INCLUSIVE : cudf::scan_type::EXCLUSIVE;
    auto null_policy = include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    return release_as_jlong(
        cudf::scan(*col, *dynamic_cast<cudf::scan_aggregation *>(agg), scan_type, null_policy));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_approxPercentile(JNIEnv *env, jclass clazz,
                                                                        jlong input_column,
                                                                        jlong percentiles_column) {
  JNI_NULL_CHECK(env, input_column, "input_column native handle is null", 0);
  JNI_NULL_CHECK(env, percentiles_column, "percentiles_column native handle is null", 0);
  try {
    using namespace cudf;
    using tdigest_column_view = cudf::tdigest::tdigest_column_view;
    jni::auto_set_device(env);
    auto const tdigest_view =
        tdigest_column_view{structs_column_view{*reinterpret_cast<column_view *>(input_column)}};
    auto const p_percentiles = reinterpret_cast<column_view *>(percentiles_column);
    return release_as_jlong(percentile_approx(tdigest_view, *p_percentiles));
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
    return release_as_jlong(cudf::quantile(*n_input_column, quantiles, n_quantile_method));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_rollingWindow(
    JNIEnv *env, jclass clazz, jlong input_col, jlong default_output_col, jint min_periods,
    jlong agg_ptr, jint preceding, jint following, jlong preceding_col, jlong following_col) {

  JNI_NULL_CHECK(env, input_col, "native handle is null", 0);
  JNI_NULL_CHECK(env, agg_ptr, "aggregation handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_input_col = reinterpret_cast<cudf::column_view *>(input_col);
    cudf::column_view *n_default_output_col =
        reinterpret_cast<cudf::column_view *>(default_output_col);
    cudf::column_view *n_preceding_col = reinterpret_cast<cudf::column_view *>(preceding_col);
    cudf::column_view *n_following_col = reinterpret_cast<cudf::column_view *>(following_col);
    cudf::rolling_aggregation *agg =
        dynamic_cast<cudf::rolling_aggregation *>(reinterpret_cast<cudf::aggregation *>(agg_ptr));
    JNI_ARG_CHECK(env, agg != nullptr, "aggregation is not an instance of rolling_aggregation", 0);

    std::unique_ptr<cudf::column> ret;
    if (n_default_output_col != nullptr) {
      if (n_preceding_col != nullptr && n_following_col != nullptr) {
        CUDF_FAIL("A default output column is not currently supported with variable length "
                  "preceding and following");
        // ret = cudf::rolling_window(*n_input_col, *n_default_output_col,
        //        *n_preceding_col, *n_following_col, min_periods, agg);
      } else {
        ret = cudf::rolling_window(*n_input_col, *n_default_output_col, preceding, following,
                                   min_periods, *agg);
      }

    } else {
      if (n_preceding_col != nullptr && n_following_col != nullptr) {
        ret = cudf::rolling_window(*n_input_col, *n_preceding_col, *n_following_col, min_periods,
                                   *agg);
      } else {
        ret = cudf::rolling_window(*n_input_col, preceding, following, min_periods, *agg);
      }
    }
    return release_as_jlong(ret);
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
    std::vector<cudf::size_type> indices(n_slice_indices.begin(), n_slice_indices.end());

    std::vector<cudf::column_view> result = cudf::slice(*n_column, indices);
    cudf::jni::native_jlongArray n_result(env, result.size());

    std::transform(result.begin(), result.end(), n_result.begin(),
                   [](cudf::column_view const &result_col) {
                     return ptr_as_jlong(new cudf::column{result_col});
                   });

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
    return release_as_jlong(cudf::lists::extract_list_element(lcv, index));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_extractListElementV(JNIEnv *env, jclass,
                                                                           jlong column_view,
                                                                           jlong indices_view) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, indices_view, "indices is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *indices = reinterpret_cast<cudf::column_view *>(indices_view);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::lists_column_view lcv(*cv);
    return release_as_jlong(cudf::lists::extract_list_element(lcv, *indices));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_dropListDuplicates(JNIEnv *env, jclass,
                                                                          jlong column_view) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const *>(column_view);
    return release_as_jlong(cudf::lists::distinct(cudf::lists_column_view{*input_cv}));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_dropListDuplicatesWithKeysValues(
    JNIEnv *env, jclass, jlong keys_vals_handle) {
  JNI_NULL_CHECK(env, keys_vals_handle, "keys_vals_handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const *>(keys_vals_handle);
    JNI_ARG_CHECK(env, input_cv->type().id() == cudf::type_id::LIST,
                  "Input column is not a lists column.", 0);

    auto const lists_keys_vals = cudf::lists_column_view(*input_cv);
    auto const keys_vals = lists_keys_vals.child();
    JNI_ARG_CHECK(env, keys_vals.type().id() == cudf::type_id::STRUCT,
                  "Input column has child that is not a structs column.", 0);
    JNI_ARG_CHECK(env, keys_vals.num_children() == 2,
                  "Input column has child that does not have 2 children.", 0);

    return release_as_jlong(
        cudf::jni::lists_distinct_by_key(lists_keys_vals, cudf::get_default_stream()));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_flattenLists(JNIEnv *env, jclass,
                                                                    jlong input_handle,
                                                                    jboolean ignore_null) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const null_policy = ignore_null ? cudf::lists::concatenate_null_policy::IGNORE :
                                           cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW;
    auto const input_cv = reinterpret_cast<cudf::column_view const *>(input_handle);
    return release_as_jlong(cudf::lists::concatenate_list_elements(*input_cv, null_policy));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listContains(JNIEnv *env, jclass,
                                                                    jlong column_view,
                                                                    jlong lookup_key) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key, "lookup scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::lists_column_view lcv(*cv);
    cudf::scalar *lookup_scalar = reinterpret_cast<cudf::scalar *>(lookup_key);
    return release_as_jlong(cudf::lists::contains(lcv, *lookup_scalar));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listContainsNulls(JNIEnv *env, jclass,
                                                                         jlong column_view) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto cv = reinterpret_cast<cudf::column_view *>(column_view);
    auto lcv = cudf::lists_column_view{*cv};
    return release_as_jlong(cudf::lists::contains_nulls(lcv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listContainsColumn(JNIEnv *env, jclass,
                                                                          jlong column_view,
                                                                          jlong lookup_key_cv) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key_cv, "lookup column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::lists_column_view lcv(*cv);
    cudf::column_view *lookup_cv = reinterpret_cast<cudf::column_view *>(lookup_key_cv);
    return release_as_jlong(cudf::lists::contains(lcv, *lookup_cv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listIndexOfScalar(JNIEnv *env, jclass,
                                                                         jlong column_view,
                                                                         jlong lookup_key,
                                                                         jboolean is_find_first) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key, "lookup scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(column_view);
    auto const lcv = cudf::lists_column_view{*cv};
    auto const lookup_key_scalar = reinterpret_cast<cudf::scalar const *>(lookup_key);
    auto const find_option = is_find_first ? cudf::lists::duplicate_find_option::FIND_FIRST :
                                             cudf::lists::duplicate_find_option::FIND_LAST;
    return release_as_jlong(cudf::lists::index_of(lcv, *lookup_key_scalar, find_option));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listIndexOfColumn(JNIEnv *env, jclass,
                                                                         jlong column_view,
                                                                         jlong lookup_keys,
                                                                         jboolean is_find_first) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_keys, "lookup key column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(column_view);
    auto const lcv = cudf::lists_column_view{*cv};
    auto const lookup_key_column = reinterpret_cast<cudf::column_view const *>(lookup_keys);
    auto const find_option = is_find_first ? cudf::lists::duplicate_find_option::FIND_FIRST :
                                             cudf::lists::duplicate_find_option::FIND_LAST;
    return release_as_jlong(cudf::lists::index_of(lcv, *lookup_key_column, find_option));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listSortRows(JNIEnv *env, jclass,
                                                                    jlong column_view,
                                                                    jboolean is_descending,
                                                                    jboolean is_null_smallest) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto sort_order = is_descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    auto null_order = is_null_smallest ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    auto *cv = reinterpret_cast<cudf::column_view *>(column_view);
    return release_as_jlong(
        cudf::lists::sort_lists(cudf::lists_column_view(*cv), sort_order, null_order));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_generateListOffsets(JNIEnv *env, jclass,
                                                                           jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(handle);
    return release_as_jlong(cudf::jni::generate_list_offsets(*cv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listsHaveOverlap(JNIEnv *env, jclass,
                                                                        jlong lhs_handle,
                                                                        jlong rhs_handle) {
  JNI_NULL_CHECK(env, lhs_handle, "lhs_handle is null", 0)
  JNI_NULL_CHECK(env, rhs_handle, "rhs_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const lhs = reinterpret_cast<cudf::column_view const *>(lhs_handle);
    auto const rhs = reinterpret_cast<cudf::column_view const *>(rhs_handle);
    auto overlap_result =
        cudf::lists::have_overlap(cudf::lists_column_view{*lhs}, cudf::lists_column_view{*rhs},
                                  cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);
    cudf::jni::post_process_list_overlap(*lhs, *rhs, overlap_result);
    return release_as_jlong(overlap_result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listsIntersectDistinct(JNIEnv *env, jclass,
                                                                              jlong lhs_handle,
                                                                              jlong rhs_handle) {
  JNI_NULL_CHECK(env, lhs_handle, "lhs_handle is null", 0)
  JNI_NULL_CHECK(env, rhs_handle, "rhs_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const lhs = reinterpret_cast<cudf::column_view const *>(lhs_handle);
    auto const rhs = reinterpret_cast<cudf::column_view const *>(rhs_handle);
    return release_as_jlong(cudf::lists::intersect_distinct(
        cudf::lists_column_view{*lhs}, cudf::lists_column_view{*rhs}, cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listsUnionDistinct(JNIEnv *env, jclass,
                                                                          jlong lhs_handle,
                                                                          jlong rhs_handle) {
  JNI_NULL_CHECK(env, lhs_handle, "lhs_handle is null", 0)
  JNI_NULL_CHECK(env, rhs_handle, "rhs_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const lhs = reinterpret_cast<cudf::column_view const *>(lhs_handle);
    auto const rhs = reinterpret_cast<cudf::column_view const *>(rhs_handle);
    return release_as_jlong(
        cudf::lists::union_distinct(cudf::lists_column_view{*lhs}, cudf::lists_column_view{*rhs},
                                    cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_listsDifferenceDistinct(JNIEnv *env, jclass,
                                                                               jlong lhs_handle,
                                                                               jlong rhs_handle) {
  JNI_NULL_CHECK(env, lhs_handle, "lhs_handle is null", 0)
  JNI_NULL_CHECK(env, rhs_handle, "rhs_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto const lhs = reinterpret_cast<cudf::column_view const *>(lhs_handle);
    auto const rhs = reinterpret_cast<cudf::column_view const *>(rhs_handle);
    return release_as_jlong(cudf::lists::difference_distinct(
        cudf::lists_column_view{*lhs}, cudf::lists_column_view{*rhs}, cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_reverseStringsOrLists(JNIEnv *env, jclass,
                                                                             jlong input_handle) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0)
  try {
    cudf::jni::auto_set_device(env);

    auto const input = reinterpret_cast<cudf::column_view const *>(input_handle);
    switch (input->type().id()) {
      case cudf::type_id::STRING:
        return release_as_jlong(cudf::strings::reverse(cudf::strings_column_view{*input}));
      case cudf::type_id::LIST:
        return release_as_jlong(cudf::lists::reverse(cudf::lists_column_view{*input}));
      default:
        JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                      "A column of type string or list is required for reverse()", 0);
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_stringSplit(JNIEnv *env, jclass,
                                                                        jlong input_handle,
                                                                        jstring delimiter_obj,
                                                                        jint limit) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);

  if (limit == 0 || limit == 1) {
    // Cannot achieve the results of splitting with limit == 0 or limit == 1.
    // This is because cudf operates on a different parameter (`max_split`) which is converted from
    // limit. When limit == 0 or limit == 1, max_split will be non-positive and will result in an
    // unlimited split.
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "limit == 0 and limit == 1 are not supported", 0);
  }

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const *>(input_handle);
    auto const strings_column = cudf::strings_column_view{*input};
    auto const delimiter_jstr = cudf::jni::native_jstring(env, delimiter_obj);
    auto const delimiter = std::string(delimiter_jstr.get(), delimiter_jstr.size_bytes());
    auto const max_split = limit > 1 ? limit - 1 : limit;
    auto result = cudf::strings::split(strings_column, cudf::string_scalar{delimiter}, max_split);
    return cudf::jni::convert_table_for_return(env, std::move(result));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_stringSplitRe(
    JNIEnv *env, jclass, jlong input_handle, jstring pattern_obj, jint regex_flags,
    jint capture_groups, jint limit) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);

  if (limit == 0 || limit == 1) {
    // Cannot achieve the results of splitting with limit == 0 or limit == 1.
    // This is because cudf operates on a different parameter (`max_split`) which is converted from
    // limit. When limit == 0 or limit == 1, max_split will be non-positive and will result in an
    // unlimited split.
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "limit == 0 and limit == 1 are not supported", 0);
  }

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const *>(input_handle);
    auto const strings_column = cudf::strings_column_view{*input};
    auto const pattern_jstr = cudf::jni::native_jstring(env, pattern_obj);
    auto const pattern = std::string(pattern_jstr.get(), pattern_jstr.size_bytes());
    auto const max_split = limit > 1 ? limit - 1 : limit;
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern, flags, groups);
    auto result = cudf::strings::split_re(strings_column, *regex_prog, max_split);
    return cudf::jni::convert_table_for_return(env, std::move(result));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringSplitRecord(JNIEnv *env, jclass,
                                                                         jlong input_handle,
                                                                         jstring delimiter_obj,
                                                                         jint limit) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);

  if (limit == 0 || limit == 1) {
    // Cannot achieve the results of splitting with limit == 0 or limit == 1.
    // This is because cudf operates on a different parameter (`max_split`) which is converted from
    // limit. When limit == 0 or limit == 1, max_split will be non-positive and will result in an
    // unlimited split.
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "limit == 0 and limit == 1 are not supported", 0);
  }

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const *>(input_handle);
    auto const strings_column = cudf::strings_column_view{*input};
    auto const delimiter_jstr = cudf::jni::native_jstring(env, delimiter_obj);
    auto const delimiter = std::string(delimiter_jstr.get(), delimiter_jstr.size_bytes());
    auto const max_split = limit > 1 ? limit - 1 : limit;
    auto result =
        cudf::strings::split_record(strings_column, cudf::string_scalar{delimiter}, max_split);
    return release_as_jlong(result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringSplitRecordRe(
    JNIEnv *env, jclass, jlong input_handle, jstring pattern_obj, jint regex_flags,
    jint capture_groups, jint limit) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);

  if (limit == 0 || limit == 1) {
    // Cannot achieve the results of splitting with limit == 0 or limit == 1.
    // This is because cudf operates on a different parameter (`max_split`) which is converted from
    // limit. When limit == 0 or limit == 1, max_split will be non-positive and will result in an
    // unlimited split.
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "limit == 0 and limit == 1 are not supported", 0);
  }

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const *>(input_handle);
    auto const strings_column = cudf::strings_column_view{*input};
    auto const pattern_jstr = cudf::jni::native_jstring(env, pattern_obj);
    auto const pattern = std::string(pattern_jstr.get(), pattern_jstr.size_bytes());
    auto const max_split = limit > 1 ? limit - 1 : limit;
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern, flags, groups);
    auto result = cudf::strings::split_record_re(strings_column, *regex_prog, max_split);
    return release_as_jlong(result);
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
    std::vector<cudf::size_type> indices(n_split_indices.begin(), n_split_indices.end());

    std::vector<cudf::column_view> result = cudf::split(*n_column, indices);
    cudf::jni::native_jlongArray n_result(env, result.size());

    std::transform(result.begin(), result.end(), n_result.begin(),
                   [](cudf::column_view const &result_col) {
                     return ptr_as_jlong(new cudf::column_view{result_col});
                   });

    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_countElements(JNIEnv *env, jclass clazz,
                                                                     jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    return release_as_jlong(cudf::lists::count_elements(cudf::lists_column_view(*n_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_charLengths(JNIEnv *env, jclass clazz,
                                                                   jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    return release_as_jlong(cudf::strings::count_characters(cudf::strings_column_view(*n_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_byteCount(JNIEnv *env, jclass clazz,
                                                                 jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    return release_as_jlong(cudf::strings::count_bytes(cudf::strings_column_view(*n_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_findAndReplaceAll(JNIEnv *env, jclass clazz,
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
    return release_as_jlong(
        cudf::find_and_replace_all(*input_column, *old_values_column, *new_values_column));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNullNative(JNIEnv *env, jclass,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::is_null(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNotNullNative(JNIEnv *env, jclass,
                                                                       jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::is_valid(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNanNative(JNIEnv *env, jclass,
                                                                   jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::is_nan(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isNotNanNative(JNIEnv *env, jclass,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::is_not_nan(*input));
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
    return release_as_jlong(cudf::unary_operation(*input, op));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_round(JNIEnv *env, jclass, jlong input_ptr,
                                                             jint decimal_places,
                                                             jint rounding_method) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    cudf::rounding_method method = static_cast<cudf::rounding_method>(rounding_method);
    return release_as_jlong(cudf::round(*input, decimal_places, method));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_year(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_year(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_month(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_month(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_day(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_day(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_hour(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_hour(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_minute(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_minute(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_second(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_second(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_weekDay(JNIEnv *env, jclass,
                                                               jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_weekday(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_lastDayOfMonth(JNIEnv *env, jclass,
                                                                      jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::last_day_of_month(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_dayOfYear(JNIEnv *env, jclass,
                                                                 jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::day_of_year(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_quarterOfYear(JNIEnv *env, jclass,
                                                                     jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::extract_quarter(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_addCalendricalMonths(JNIEnv *env, jclass,
                                                                            jlong ts_ptr,
                                                                            jlong months_ptr) {
  JNI_NULL_CHECK(env, ts_ptr, "ts is null", 0);
  JNI_NULL_CHECK(env, months_ptr, "months is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *ts = reinterpret_cast<cudf::column_view *>(ts_ptr);
    const cudf::column_view *months = reinterpret_cast<cudf::column_view *>(months_ptr);
    return release_as_jlong(cudf::datetime::add_calendrical_months(*ts, *months));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isLeapYear(JNIEnv *env, jclass,
                                                                  jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::datetime::is_leap_year(*input));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_castTo(JNIEnv *env, jclass, jlong handle,
                                                              jint type, jint scale) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type n_data_type = cudf::jni::make_data_type(type, scale);
    if (n_data_type == column->type()) {
      return ptr_as_jlong(new cudf::column(*column));
    }
    if (n_data_type.id() == cudf::type_id::STRING) {
      switch (column->type().id()) {
        case cudf::type_id::BOOL8: {
          auto const true_scalar = cudf::string_scalar("true");
          auto const false_scalar = cudf::string_scalar("false");
          return release_as_jlong(cudf::strings::from_booleans(*column, true_scalar, false_scalar));
        }
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64: return release_as_jlong(cudf::strings::from_floats(*column));
        case cudf::type_id::INT8:
        case cudf::type_id::UINT8:
        case cudf::type_id::INT16:
        case cudf::type_id::UINT16:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64: return release_as_jlong(cudf::strings::from_integers(*column));
        case cudf::type_id::DECIMAL32:
        case cudf::type_id::DECIMAL64:
        case cudf::type_id::DECIMAL128:
          return release_as_jlong(cudf::strings::from_fixed_point(*column));
        default: JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
      }
    } else if (column->type().id() == cudf::type_id::STRING) {
      switch (n_data_type.id()) {
        case cudf::type_id::BOOL8: {
          auto const true_scalar = cudf::string_scalar("true");
          return release_as_jlong(cudf::strings::to_booleans(*column, true_scalar));
        }
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64:
          return release_as_jlong(cudf::strings::to_floats(*column, n_data_type));
        case cudf::type_id::INT8:
        case cudf::type_id::UINT8:
        case cudf::type_id::INT16:
        case cudf::type_id::UINT16:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
          return release_as_jlong(cudf::strings::to_integers(*column, n_data_type));
        case cudf::type_id::DECIMAL32:
        case cudf::type_id::DECIMAL64:
        case cudf::type_id::DECIMAL128:
          return release_as_jlong(cudf::strings::to_fixed_point(*column, n_data_type));
        default: JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
      }
    } else if (cudf::is_timestamp(n_data_type) && cudf::is_numeric(column->type())) {
      // This is a temporary workaround to allow Java to cast from integral types into a timestamp
      // without forcing an intermediate duration column to be manifested.  Ultimately this style of
      // "reinterpret" casting will be supported via https://github.com/rapidsai/cudf/pull/5358
      if (n_data_type.id() == cudf::type_id::TIMESTAMP_DAYS) {
        if (column->type().id() != cudf::type_id::INT32) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                        "Numeric cast to TIMESTAMP_DAYS requires INT32", 0);
        }
      } else {
        if (column->type().id() != cudf::type_id::INT64) {
          JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                        "Numeric cast to non-day timestamp requires INT64", 0);
        }
      }
      cudf::data_type duration_type = cudf::jni::timestamp_to_duration(n_data_type);
      cudf::column_view duration_view = cudf::column_view(
          duration_type, column->size(), column->head(), column->null_mask(), column->null_count());
      return release_as_jlong(cudf::cast(duration_view, n_data_type));
    } else if (cudf::is_timestamp(column->type()) && cudf::is_numeric(n_data_type)) {
      // This is a temporary workaround to allow Java to cast from timestamp types to integral types
      // without forcing an intermediate duration column to be manifested.  Ultimately this style of
      // "reinterpret" casting will be supported via https://github.com/rapidsai/cudf/pull/5358
      cudf::data_type duration_type = cudf::jni::timestamp_to_duration(column->type());
      cudf::column_view duration_view = cudf::column_view(
          duration_type, column->size(), column->head(), column->null_mask(), column->null_count());
      return release_as_jlong(cudf::cast(duration_view, n_data_type));
    } else {
      return release_as_jlong(cudf::cast(*column, n_data_type));
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_bitCastTo(JNIEnv *env, jclass, jlong handle,
                                                                 jint type, jint scale) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type n_data_type = cudf::jni::make_data_type(type, scale);
    return ptr_as_jlong(new cudf::column_view{cudf::bit_cast(*column, n_data_type)});
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
    return release_as_jlong(byte_cast(*column, config));
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

    return release_as_jlong(cudf::strings::to_timestamps(
        strings_column, cudf::data_type(static_cast<cudf::type_id>(time_unit)), format.get()));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isTimestamp(JNIEnv *env, jclass,
                                                                   jlong handle,
                                                                   jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, formatObj, "format is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring format(env, formatObj);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);
    return release_as_jlong(cudf::strings::is_timestamp(strings_column, format.get()));
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
    return release_as_jlong(cudf::strings::from_timestamps(*column, format.get()));
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_containsVector(JNIEnv *env, jobject j_object,
                                                                      jlong j_values_handle,
                                                                      jlong j_search_space_handle) {
  JNI_NULL_CHECK(env, j_values_handle, "values vector is null", false);
  JNI_NULL_CHECK(env, j_search_space_handle, "search_space vector is null", false);
  try {
    cudf::jni::auto_set_device(env);
    auto const search_space_ptr =
        reinterpret_cast<cudf::column_view const *>(j_search_space_handle);
    auto const values_ptr = reinterpret_cast<cudf::column_view const *>(j_values_handle);

    // The C++ API `cudf::contains` requires that the search space is the first parameter.
    return release_as_jlong(cudf::contains(*search_space_ptr, *values_ptr));
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
    return release_as_jlong(
        cudf::transform(*column, n_udf, cudf::data_type(cudf::type_id::INT32), j_is_ptx));
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
    return release_as_jlong(cudf::strings::starts_with(strings_column, *comp_scalar));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringEndWith(JNIEnv *env, jobject j_object,
                                                                     jlong j_view_handle,
                                                                     jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar *comp_scalar = reinterpret_cast<cudf::string_scalar *>(comp_string);
    return release_as_jlong(cudf::strings::ends_with(strings_column, *comp_scalar));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringContains(JNIEnv *env, jobject j_object,
                                                                      jlong j_view_handle,
                                                                      jlong comp_string) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, comp_string, "comparison string scalar is null", false);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_column(*column_view);
    cudf::string_scalar *comp_scalar = reinterpret_cast<cudf::string_scalar *>(comp_string);
    return release_as_jlong(cudf::strings::contains(strings_column, *comp_scalar));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_matchesRe(JNIEnv *env, jobject j_object,
                                                                 jlong j_view_handle,
                                                                 jstring pattern_obj,
                                                                 jint regex_flags,
                                                                 jint capture_groups) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, pattern_obj, "pattern is null", false);

  try {
    cudf::jni::auto_set_device(env);
    auto const column_view = reinterpret_cast<cudf::column_view const *>(j_view_handle);
    auto const strings_column = cudf::strings_column_view{*column_view};
    auto const pattern = cudf::jni::native_jstring(env, pattern_obj);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, groups);
    return release_as_jlong(cudf::strings::matches_re(strings_column, *regex_prog));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_containsRe(JNIEnv *env, jobject j_object,
                                                                  jlong j_view_handle,
                                                                  jstring pattern_obj,
                                                                  jint regex_flags,
                                                                  jint capture_groups) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, pattern_obj, "pattern is null", false);

  try {
    cudf::jni::auto_set_device(env);
    auto const column_view = reinterpret_cast<cudf::column_view const *>(j_view_handle);
    auto const strings_column = cudf::strings_column_view{*column_view};
    auto const pattern = cudf::jni::native_jstring(env, pattern_obj);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const capture = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, capture);
    return release_as_jlong(cudf::strings::contains_re(strings_column, *regex_prog));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_like(JNIEnv *env, jobject j_object,
                                                            jlong j_view_handle, jlong pattern,
                                                            jlong escapeChar) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", false);
  JNI_NULL_CHECK(env, pattern, "pattern is null", false);
  JNI_NULL_CHECK(env, escapeChar, "escape character is null", false);

  try {
    cudf::jni::auto_set_device(env);
    auto const column_view = reinterpret_cast<cudf::column_view const *>(j_view_handle);
    auto const strings_column = cudf::strings_column_view{*column_view};
    auto const pattern_scalar = reinterpret_cast<cudf::string_scalar const *>(pattern);
    auto const escape_scalar = reinterpret_cast<cudf::string_scalar const *>(escapeChar);
    return release_as_jlong(cudf::strings::like(strings_column, *pattern_scalar, *escape_scalar));
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

    if (lhs->type().id() == cudf::type_id::STRUCT) {
      auto out = make_fixed_width_column(n_data_type, lhs->size(), cudf::mask_state::UNALLOCATED);

      if (op == cudf::binary_operator::NULL_EQUALS) {
        out->set_null_mask(rmm::device_buffer{}, 0);
      } else {
        auto [new_mask, null_count] = cudf::bitmask_and(cudf::table_view{{*lhs, *rhs}});
        out->set_null_mask(std::move(new_mask), null_count);
      }

      auto out_view = out->mutable_view();
      cudf::binops::compiled::detail::apply_sorting_struct_binary_op(
          out_view, *lhs, *rhs, false, false, op, cudf::get_default_stream());
      return release_as_jlong(out);
    }

    return release_as_jlong(cudf::binary_operation(*lhs, *rhs, op, n_data_type));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_fixedPointOutputScale(JNIEnv *env, jclass,
                                                                            jint int_op,
                                                                            jint lhs_scale,
                                                                            jint rhs_scale) {
  try {
    // we just return the scale as the types will be the same as the lhs input
    return cudf::binary_operation_fixed_point_scale(static_cast<cudf::binary_operator>(int_op),
                                                    lhs_scale, rhs_scale);
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

    if (lhs->type().id() == cudf::type_id::STRUCT) {
      auto out = make_fixed_width_column(n_data_type, lhs->size(), cudf::mask_state::UNALLOCATED);

      if (op == cudf::binary_operator::NULL_EQUALS) {
        out->set_null_mask(rmm::device_buffer{}, 0);
      } else {
        auto [new_mask, new_null_count] = cudf::binops::scalar_col_valid_mask_and(*lhs, *rhs);
        out->set_null_mask(std::move(new_mask), new_null_count);
      }

      auto rhsv = cudf::make_column_from_scalar(*rhs, 1);
      auto out_view = out->mutable_view();
      cudf::binops::compiled::detail::apply_sorting_struct_binary_op(
          out_view, *lhs, rhsv->view(), false, true, op, cudf::get_default_stream());
      return release_as_jlong(out);
    }

    return release_as_jlong(cudf::binary_operation(*lhs, *rhs, op, n_data_type));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_substringS(JNIEnv *env, jclass,
                                                                  jlong cv_handle, jint start) {
  JNI_NULL_CHECK(env, cv_handle, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(cv_handle);
    auto const scv = cudf::strings_column_view{*cv};
    return release_as_jlong(cudf::strings::slice_strings(scv, start));
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
    return release_as_jlong(cudf::strings::slice_strings(scv, start, end));
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
    return release_as_jlong(cudf::strings::slice_strings(scv, *sc, *ec));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_substringLocate(JNIEnv *env, jclass,
                                                                       jlong column_view,
                                                                       jlong substring, jint start,
                                                                       jint end) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, substring, "target string scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_scalar = reinterpret_cast<cudf::string_scalar *>(substring);
    return release_as_jlong(cudf::strings::find(scv, *ss_scalar, start, end));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringReplace(JNIEnv *env, jclass,
                                                                     jlong column_view,
                                                                     jlong target, jlong replace) {
  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, target, "target string scalar is null", 0);
  JNI_NULL_CHECK(env, replace, "replace string scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar *ss_target = reinterpret_cast<cudf::string_scalar *>(target);
    cudf::string_scalar *ss_replace = reinterpret_cast<cudf::string_scalar *>(replace);
    return release_as_jlong(cudf::strings::replace(scv, *ss_target, *ss_replace));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringReplaceMulti(JNIEnv *env, jclass,
                                                                          jlong inputs_cv,
                                                                          jlong targets_cv,
                                                                          jlong repls_cv) {
  JNI_NULL_CHECK(env, inputs_cv, "column is null", 0);
  JNI_NULL_CHECK(env, targets_cv, "targets string column view is null", 0);
  JNI_NULL_CHECK(env, repls_cv, "repls string column view is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(inputs_cv);
    cudf::strings_column_view scv(*cv);
    cudf::column_view *cvtargets = reinterpret_cast<cudf::column_view *>(targets_cv);
    cudf::strings_column_view scvtargets(*cvtargets);
    cudf::column_view *cvrepls = reinterpret_cast<cudf::column_view *>(repls_cv);
    cudf::strings_column_view scvrepls(*cvrepls);
    return release_as_jlong(cudf::strings::replace(scv, scvtargets, scvrepls));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_mapLookupForKeys(JNIEnv *env, jclass,
                                                                        jlong map_column_view,
                                                                        jlong lookup_keys) {
  JNI_NULL_CHECK(env, map_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_keys, "lookup key is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view *>(map_column_view);
    auto const *column_keys = reinterpret_cast<cudf::column_view *>(lookup_keys);
    auto const maps_view = cudf::jni::maps_column_view{*cv};
    return release_as_jlong(maps_view.get_values_for(*column_keys));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_mapLookup(JNIEnv *env, jclass,
                                                                 jlong map_column_view,
                                                                 jlong lookup_key) {
  JNI_NULL_CHECK(env, map_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key, "lookup key is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view *>(map_column_view);
    auto const *scalar_key = reinterpret_cast<cudf::scalar *>(lookup_key);
    auto const maps_view = cudf::jni::maps_column_view{*cv};
    return release_as_jlong(maps_view.get_values_for(*scalar_key));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_mapContainsKeys(JNIEnv *env, jclass,
                                                                       jlong map_column_view,
                                                                       jlong lookup_keys) {
  JNI_NULL_CHECK(env, map_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_keys, "lookup key is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view *>(map_column_view);
    auto const *column_key = reinterpret_cast<cudf::column_view *>(lookup_keys);
    auto const maps_view = cudf::jni::maps_column_view{*cv};
    return release_as_jlong(maps_view.contains(*column_key));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_mapContains(JNIEnv *env, jclass,
                                                                   jlong map_column_view,
                                                                   jlong lookup_key) {
  JNI_NULL_CHECK(env, map_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, lookup_key, "lookup key is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view *>(map_column_view);
    auto const *scalar_key = reinterpret_cast<cudf::scalar *>(lookup_key);
    auto const maps_view = cudf::jni::maps_column_view{*cv};
    return release_as_jlong(maps_view.contains(*scalar_key));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceRegex(
    JNIEnv *env, jclass, jlong j_column_view, jstring j_pattern, jint regex_flags,
    jint capture_groups, jlong j_repl, jlong j_maxrepl) {

  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, j_pattern, "pattern string is null", 0);
  JNI_NULL_CHECK(env, j_repl, "replace scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(j_column_view);
    auto const strings_column = cudf::strings_column_view{*cv};
    auto const pattern = cudf::jni::native_jstring(env, j_pattern);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, groups);
    auto const repl = reinterpret_cast<cudf::string_scalar const *>(j_repl);
    return release_as_jlong(
        cudf::strings::replace_re(strings_column, *regex_prog, *repl, j_maxrepl));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_replaceMultiRegex(JNIEnv *env, jclass,
                                                                         jlong j_column_view,
                                                                         jobjectArray j_patterns,
                                                                         jlong j_repls) {

  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, j_patterns, "patterns is null", 0);
  JNI_NULL_CHECK(env, j_repls, "repls is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto cv = reinterpret_cast<cudf::column_view const *>(j_column_view);
    cudf::strings_column_view scv(*cv);
    cudf::jni::native_jstringArray patterns(env, j_patterns);
    auto repl_cv = reinterpret_cast<cudf::column_view const *>(j_repls);
    cudf::strings_column_view repl_scv(*repl_cv);
    return release_as_jlong(cudf::strings::replace_re(scv, patterns.as_cpp_vector(), repl_scv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringReplaceWithBackrefs(
    JNIEnv *env, jclass, jlong j_column_view, jstring pattern_obj, jint regex_flags,
    jint capture_groups, jstring replace_obj) {

  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  JNI_NULL_CHECK(env, pattern_obj, "pattern string is null", 0);
  JNI_NULL_CHECK(env, replace_obj, "replace string is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = reinterpret_cast<cudf::column_view const *>(j_column_view);
    auto const strings_column = cudf::strings_column_view{*cv};
    auto const pattern = cudf::jni::native_jstring(env, pattern_obj);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, groups);
    cudf::jni::native_jstring ss_replace(env, replace_obj);
    return release_as_jlong(
        cudf::strings::replace_with_backrefs(strings_column, *regex_prog, ss_replace.get()));
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
    return release_as_jlong(cudf::strings::zfill(scv, width));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_pad(JNIEnv *env, jclass, jlong column_view,
                                                           jint j_width, jint j_side,
                                                           jstring fill_char) {

  JNI_NULL_CHECK(env, column_view, "column is null", 0);
  JNI_NULL_CHECK(env, fill_char, "fill_char is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::size_type width = reinterpret_cast<cudf::size_type>(j_width);
    cudf::strings::side_type side = static_cast<cudf::strings::side_type>(j_side);
    cudf::jni::native_jstring ss_fill(env, fill_char);
    return release_as_jlong(cudf::strings::pad(scv, width, side, ss_fill.get()));
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
    cudf::strings::side_type s_striptype = static_cast<cudf::strings::side_type>(strip_type);
    cudf::string_scalar *ss_tostrip = reinterpret_cast<cudf::string_scalar *>(to_strip);
    return release_as_jlong(cudf::strings::strip(scv, s_striptype, *ss_tostrip));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnView_extractRe(JNIEnv *env, jclass,
                                                                      jlong j_view_handle,
                                                                      jstring pattern_obj,
                                                                      jint regex_flags,
                                                                      jint capture_groups) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", nullptr);
  JNI_NULL_CHECK(env, pattern_obj, "pattern is null", nullptr);

  try {
    cudf::jni::auto_set_device(env);
    auto const column_view = reinterpret_cast<cudf::column_view const *>(j_view_handle);
    auto const strings_column = cudf::strings_column_view{*column_view};
    auto const pattern = cudf::jni::native_jstring(env, pattern_obj);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, groups);
    return cudf::jni::convert_table_for_return(env,
                                               cudf::strings::extract(strings_column, *regex_prog));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_extractAllRecord(
    JNIEnv *env, jclass, jlong j_view_handle, jstring pattern_obj, jint regex_flags,
    jint capture_groups, jint idx) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", 0);
  JNI_NULL_CHECK(env, pattern_obj, "pattern is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const column_view = reinterpret_cast<cudf::column_view const *>(j_view_handle);
    auto const strings_column = cudf::strings_column_view{*column_view};
    auto const pattern = cudf::jni::native_jstring(env, pattern_obj);
    auto const flags = static_cast<cudf::strings::regex_flags>(regex_flags);
    auto const groups = static_cast<cudf::strings::capture_groups>(capture_groups);
    auto const regex_prog = cudf::strings::regex_program::create(pattern.get(), flags, groups);
    auto result = (idx == 0) ? cudf::strings::findall(strings_column, *regex_prog) :
                               cudf::strings::extract_all_record(strings_column, *regex_prog);
    return release_as_jlong(result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_urlDecode(JNIEnv *env, jclass,
                                                                 jlong j_view_handle) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto view_ptr = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_view(*view_ptr);
    return release_as_jlong(cudf::strings::url_decode(strings_view));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_urlEncode(JNIEnv *env, jclass,
                                                                 jlong j_view_handle) {
  JNI_NULL_CHECK(env, j_view_handle, "column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto view_ptr = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view strings_view(*view_ptr);
    return release_as_jlong(cudf::strings::url_encode(strings_view));
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
    return release_as_jlong(
        cudf::normalize_nans_and_zeros(*reinterpret_cast<column_view *>(input_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_bitwiseMergeAndSetValidity(
    JNIEnv *env, jobject j_object, jlong base_column, jlongArray column_handles, jint bin_op) {
  JNI_NULL_CHECK(env, base_column, "base column native handle is null", 0);
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *original_column = reinterpret_cast<cudf::column_view *>(base_column);
    std::unique_ptr<cudf::column> copy(new cudf::column(*original_column));
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);

    if (n_cudf_columns.size() == 0) {
      copy->set_null_mask({}, 0);
      return release_as_jlong(copy);
    }

    cudf::binary_operator op = static_cast<cudf::binary_operator>(bin_op);
    switch (op) {
      case cudf::binary_operator::BITWISE_AND: {
        auto cols = n_cudf_columns.get_dereferenced();
        cols.push_back(copy->view());
        auto table_view = cudf::table_view{cols};
        auto [new_bitmask, null_count] = cudf::bitmask_and(table_view);
        copy->set_null_mask(std::move(new_bitmask), null_count);
        break;
      }
      case cudf::binary_operator::BITWISE_OR: {
        auto input_table = cudf::table_view{n_cudf_columns.get_dereferenced()};
        auto [tmp_new_bitmask, tmp_null_count] = cudf::bitmask_or(input_table);
        copy->set_null_mask(std::move(tmp_new_bitmask), tmp_null_count);
        // and the bitmask with the original column
        cudf::table_view table_view{std::vector<cudf::column_view>{copy->view(), *original_column}};
        auto [new_bitmask, null_count] = cudf::bitmask_and(table_view);
        copy->set_null_mask(std::move(new_bitmask), null_count);
        break;
      }
      default: JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "Unsupported merge operation", 0);
    }
    auto const copy_cv = copy->view();
    if (cudf::has_nonempty_nulls(copy_cv)) {
      copy = cudf::purge_nonempty_nulls(copy_cv);
    }

    return release_as_jlong(copy);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_copyWithBooleanColumnAsValidity(
    JNIEnv *env, jobject j_object, jlong exemplar_handle, jlong validity_column_handle) {
  JNI_NULL_CHECK(env, exemplar_handle, "ColumnView handle is null", 0);
  JNI_NULL_CHECK(env, validity_column_handle, "Validity column handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const exemplar = *reinterpret_cast<cudf::column_view *>(exemplar_handle);
    auto const validity = *reinterpret_cast<cudf::column_view *>(validity_column_handle);
    return release_as_jlong(
        cudf::jni::new_column_with_boolean_column_as_validity(exemplar, validity));
  }
  CATCH_STD(env, 0);
}

////////
// Native cudf::column_view life cycle and metadata access methods. Life cycle methods
// should typically only be called from the CudfColumn inner class.
////////

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_makeCudfColumnView(
    JNIEnv *env, jclass, jint j_type, jint scale, jlong j_data, jlong j_data_size, jlong j_offset,
    jlong j_valid, jint j_null_count, jint size, jlongArray j_children) {

  try {
    using cudf::column_view;
    cudf::jni::auto_set_device(env);
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    cudf::data_type n_data_type = cudf::jni::make_data_type(j_type, scale);

    void *data = reinterpret_cast<void *>(j_data);
    cudf::bitmask_type *valid = reinterpret_cast<cudf::bitmask_type *>(j_valid);
    if (valid == nullptr) {
      j_null_count = 0;
    }

    if (j_null_count < 0) { // Check for unknown null count.
      // Calculate concrete null count.
      j_null_count = cudf::null_count(valid, 0, size);
    }

    if (n_type == cudf::type_id::STRING) {
      if (size == 0) {
        return ptr_as_jlong(
            new cudf::column_view(cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0));
      } else {
        JNI_NULL_CHECK(env, j_offset, "offset is null", 0);
        // This must be kept in sync with how string columns are created
        // offsets are always the first child
        // data is the second child

        cudf::size_type *offsets = reinterpret_cast<cudf::size_type *>(j_offset);
        cudf::column_view offsets_column(cudf::data_type{cudf::type_id::INT32}, size + 1, offsets,
                                         nullptr, 0);
        cudf::column_view data_column(cudf::data_type{cudf::type_id::INT8}, j_data_size, data,
                                      nullptr, 0);
        return ptr_as_jlong(new cudf::column_view(cudf::data_type{cudf::type_id::STRING}, size,
                                                  nullptr, valid, j_null_count, 0,
                                                  {offsets_column, data_column}));
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
      cudf::column_view offsets_column(cudf::data_type{cudf::type_id::INT32}, offsets_size, offsets,
                                       nullptr, 0);
      return ptr_as_jlong(new cudf::column_view(cudf::data_type{cudf::type_id::LIST}, size, nullptr,
                                                valid, j_null_count, 0,
                                                {offsets_column, *children[0]}));
    } else if (n_type == cudf::type_id::STRUCT) {
      JNI_NULL_CHECK(env, j_children, "children of a struct are null", 0);
      cudf::jni::native_jpointerArray<cudf::column_view> children(env, j_children);
      std::vector<column_view> children_vector = children.get_dereferenced();
      return ptr_as_jlong(new cudf::column_view(cudf::data_type{cudf::type_id::STRUCT}, size,
                                                nullptr, valid, j_null_count, 0, children_vector));
    } else {
      return ptr_as_jlong(new cudf::column_view(n_data_type, size, data, valid, j_null_count));
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeTypeId(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return static_cast<jint>(column->type().id());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeTypeScale(JNIEnv *env, jclass,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return column->type().scale();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnView_getNativeRowCount(JNIEnv *env, jclass,
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
    } else if (column->type().id() != cudf::type_id::LIST &&
               column->type().id() != cudf::type_id::STRUCT) {
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
    } else if (column->type().id() != cudf::type_id::LIST &&
               column->type().id() != cudf::type_id::STRUCT) {
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
                                                                         jlong handle,
                                                                         jint child_index) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    auto const is_list = column->type().id() == cudf::type_id::LIST;
    auto const child = column->child(child_index + (is_list ? 1 : 0));
    return ptr_as_jlong(new cudf::column_view(child));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getListOffsetCvPointer(JNIEnv *env,
                                                                              jobject j_object,
                                                                              jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::lists_column_view view = cudf::lists_column_view(*column);
    cudf::column_view offsets_view = view.offsets();
    return ptr_as_jlong(new cudf::column_view(offsets_view));
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
        result = ptr_as_jlong(offsets_view.data<char>());
      }
    } else if (column->type().id() == cudf::type_id::LIST) {
      if (column->size() > 0) {
        cudf::lists_column_view view = cudf::lists_column_view(*column);
        cudf::column_view offsets_view = view.offsets();
        result = ptr_as_jlong(offsets_view.data<char>());
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeValidityAddress(JNIEnv *env, jclass,
                                                                                jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    return ptr_as_jlong(column->null_mask());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getNativeValidityLength(JNIEnv *env, jclass,
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getDeviceMemorySize(JNIEnv *env, jclass,
                                                                           jlong handle,
                                                                           jboolean pad_for_cpu) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto view = reinterpret_cast<cudf::column_view const *>(handle);
    return calc_device_memory_size(*view, pad_for_cpu);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_hostPaddingSizeInBytes(JNIEnv *env, jclass) {
  return sizeof(std::max_align_t);
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

    return release_as_jlong(
        clamp(*column_view, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_title(JNIEnv *env, jobject j_object,
                                                             jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::strings::title(*view));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_capitalize(JNIEnv *env, jobject j_object,
                                                                  jlong strs_handle,
                                                                  jlong delimiters_handle) {

  JNI_NULL_CHECK(env, strs_handle, "native view handle is null", 0)
  JNI_NULL_CHECK(env, delimiters_handle, "delimiters scalar handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(strs_handle);
    cudf::string_scalar *deli = reinterpret_cast<cudf::string_scalar *>(delimiters_handle);
    return release_as_jlong(cudf::strings::capitalize(*view, *deli));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_joinStrings(JNIEnv *env, jobject j_object,
                                                                   jlong strs_handle,
                                                                   jlong separator_handle,
                                                                   jlong narep_handle) {

  JNI_NULL_CHECK(env, strs_handle, "native view handle is null", 0)
  JNI_NULL_CHECK(env, separator_handle, "separator scalar handle is null", 0)
  JNI_NULL_CHECK(env, narep_handle, "narep scalar handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(strs_handle);
    cudf::string_scalar *sep = reinterpret_cast<cudf::string_scalar *>(separator_handle);
    cudf::string_scalar *narep = reinterpret_cast<cudf::string_scalar *>(narep_handle);
    return release_as_jlong(cudf::strings::join_strings(*view, *sep, *narep));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_makeStructView(JNIEnv *env, jobject j_object,
                                                                      jlongArray handles,
                                                                      jlong row_count) {

  JNI_NULL_CHECK(env, handles, "native view handles are null", 0)
  try {
    cudf::jni::auto_set_device(env);
    auto children = cudf::jni::native_jpointerArray<cudf::column_view>{env, handles};
    auto children_vector = children.get_dereferenced();
    return ptr_as_jlong(new cudf::column_view(cudf::data_type{cudf::type_id::STRUCT}, row_count,
                                              nullptr, nullptr, 0, 0, children_vector));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_nansToNulls(JNIEnv *env, jobject j_object,
                                                                   jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view *>(handle);
    // get a new null mask by setting all the nans to null
    auto [new_nullmask, new_null_count] = cudf::nans_to_nulls(input);
    // create a column_view which is a no-copy wrapper around the original column without the null
    // mask
    auto const input_without_nullmask = cudf::column_view(
        input.type(), input.size(), input.head<void>(), nullptr, 0, input.offset(),
        std::vector<cudf::column_view>{input.child_begin(), input.child_end()});
    // create a column by deep copying `input_without_nullmask`.
    auto deep_copy = std::make_unique<cudf::column>(input_without_nullmask);
    deep_copy->set_null_mask(std::move(*new_nullmask), new_null_count);
    return release_as_jlong(deep_copy);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isFloat(JNIEnv *env, jobject j_object,
                                                               jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::strings::is_float(*view));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isInteger(JNIEnv *env, jobject j_object,
                                                                 jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    return release_as_jlong(cudf::strings::is_integer(*view));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isFixedPoint(JNIEnv *env, jobject,
                                                                    jlong handle, jint j_dtype,
                                                                    jint scale) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type fp_dtype = cudf::jni::make_data_type(j_dtype, scale);
    return release_as_jlong(cudf::strings::is_fixed_point(*view, fp_dtype));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_isIntegerWithType(JNIEnv *env, jobject,
                                                                         jlong handle, jint j_dtype,
                                                                         jint scale) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type int_dtype = cudf::jni::make_data_type(j_dtype, scale);
    return release_as_jlong(cudf::strings::is_integer(*view, int_dtype));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_copyColumnViewToCV(JNIEnv *env,
                                                                          jobject j_object,
                                                                          jlong handle) {

  JNI_NULL_CHECK(env, handle, "native view handle is null", 0)

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *view = reinterpret_cast<cudf::column_view *>(handle);
    return ptr_as_jlong(new cudf::column(*view));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_getJSONObject(JNIEnv *env, jclass,
                                                                     jlong j_view_handle,
                                                                     jlong j_scalar_handle) {

  JNI_NULL_CHECK(env, j_view_handle, "view cannot be null", 0);
  JNI_NULL_CHECK(env, j_scalar_handle, "path cannot be null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *n_column_view = reinterpret_cast<cudf::column_view *>(j_view_handle);
    cudf::strings_column_view n_strings_col_view(*n_column_view);
    cudf::string_scalar *n_scalar_path = reinterpret_cast<cudf::string_scalar *>(j_scalar_handle);
    return release_as_jlong(cudf::get_json_object(n_strings_col_view, *n_scalar_path));
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringConcatenationListElementsSepCol(
    JNIEnv *env, jclass, jlong column_handle, jlong sep_handle, jlong separator_narep,
    jlong col_narep, jboolean separate_nulls, jboolean empty_string_output_if_empty_list) {
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_NULL_CHECK(env, sep_handle, "separator column handle is null", 0);
  JNI_NULL_CHECK(env, separator_narep, "separator narep string scalar object is null", 0);
  JNI_NULL_CHECK(env, col_narep, "column narep string scalar object is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const auto &separator_narep_scalar = *reinterpret_cast<cudf::string_scalar *>(separator_narep);
    const auto &col_narep_scalar = *reinterpret_cast<cudf::string_scalar *>(col_narep);
    auto null_policy = separate_nulls ? cudf::strings::separator_on_nulls::YES :
                                        cudf::strings::separator_on_nulls::NO;
    auto empty_list_output = empty_string_output_if_empty_list ?
                                 cudf::strings::output_if_empty_list::EMPTY_STRING :
                                 cudf::strings::output_if_empty_list::NULL_ELEMENT;

    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(sep_handle);
    cudf::strings_column_view strings_column(*column);
    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_handle);
    cudf::lists_column_view lcv(*cv);
    return release_as_jlong(
        cudf::strings::join_list_elements(lcv, strings_column, separator_narep_scalar,
                                          col_narep_scalar, null_policy, empty_list_output));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_stringConcatenationListElements(
    JNIEnv *env, jclass, jlong column_handle, jlong separator, jlong narep, jboolean separate_nulls,
    jboolean empty_string_output_if_empty_list) {
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_NULL_CHECK(env, separator, "separator string scalar object is null", 0);
  JNI_NULL_CHECK(env, narep, "separator narep string scalar object is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const auto &separator_scalar = *reinterpret_cast<cudf::string_scalar *>(separator);
    const auto &narep_scalar = *reinterpret_cast<cudf::string_scalar *>(narep);
    auto null_policy = separate_nulls ? cudf::strings::separator_on_nulls::YES :
                                        cudf::strings::separator_on_nulls::NO;
    auto empty_list_output = empty_string_output_if_empty_list ?
                                 cudf::strings::output_if_empty_list::EMPTY_STRING :
                                 cudf::strings::output_if_empty_list::NULL_ELEMENT;

    cudf::column_view *cv = reinterpret_cast<cudf::column_view *>(column_handle);
    cudf::lists_column_view lcv(*cv);
    return release_as_jlong(cudf::strings::join_list_elements(lcv, separator_scalar, narep_scalar,
                                                              null_policy, empty_list_output));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_repeatStrings(JNIEnv *env, jclass,
                                                                     jlong strings_handle,
                                                                     jint repeat_times) {
  JNI_NULL_CHECK(env, strings_handle, "strings_handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const cv = *reinterpret_cast<cudf::column_view *>(strings_handle);
    auto const strs_col = cudf::strings_column_view(cv);
    return release_as_jlong(cudf::strings::repeat_strings(strs_col, repeat_times));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_repeatStringsWithColumnRepeatTimes(
    JNIEnv *env, jclass, jlong strings_handle, jlong repeat_times_handle) {
  JNI_NULL_CHECK(env, strings_handle, "strings_handle is null", 0);
  JNI_NULL_CHECK(env, repeat_times_handle, "repeat_times_handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const strings_cv = *reinterpret_cast<cudf::column_view *>(strings_handle);
    auto const strs_col = cudf::strings_column_view(strings_cv);
    auto const repeat_times_cv = *reinterpret_cast<cudf::column_view *>(repeat_times_handle);
    return release_as_jlong(cudf::strings::repeat_strings(strs_col, repeat_times_cv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_applyBooleanMask(
    JNIEnv *env, jclass, jlong list_column_handle, jlong boolean_mask_list_column_handle) {
  JNI_NULL_CHECK(env, list_column_handle, "list handle is null", 0);
  JNI_NULL_CHECK(env, boolean_mask_list_column_handle, "boolean mask handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view const *list_column =
        reinterpret_cast<cudf::column_view const *>(list_column_handle);
    cudf::lists_column_view const list_view = cudf::lists_column_view(*list_column);

    cudf::column_view const *boolean_mask_list_column =
        reinterpret_cast<cudf::column_view const *>(boolean_mask_list_column_handle);
    cudf::lists_column_view const boolean_mask_list_view =
        cudf::lists_column_view(*boolean_mask_list_column);

    return release_as_jlong(cudf::lists::apply_boolean_mask(list_view, boolean_mask_list_view));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL
Java_ai_rapids_cudf_ColumnView_hasNonEmptyNulls(JNIEnv *env, jclass, jlong column_view_handle) {
  JNI_NULL_CHECK(env, column_view_handle, "column_view handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view const *>(column_view_handle);
    return cudf::has_nonempty_nulls(*cv);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_ColumnView_purgeNonEmptyNulls(JNIEnv *env, jclass, jlong column_view_handle) {
  JNI_NULL_CHECK(env, column_view_handle, "column_view handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const *cv = reinterpret_cast<cudf::column_view const *>(column_view_handle);
    return release_as_jlong(cudf::purge_nonempty_nulls(*cv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnView_toHex(JNIEnv *env, jclass, jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    return release_as_jlong(cudf::strings::integers_to_hex(*input));
  }
  CATCH_STD(env, 0);
}
} // extern "C"
