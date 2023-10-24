/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Aggregation_close(JNIEnv *env, jclass class_object,
                                                             jlong ptr) {
  try {
    cudf::jni::auto_set_device(env);
    auto to_del = reinterpret_cast<cudf::aggregation *>(ptr);
    delete to_del;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createNoParamAgg(JNIEnv *env,
                                                                         jclass class_object,
                                                                         jint kind) {
  try {
    cudf::jni::auto_set_device(env);
    auto ret = [&] {
      // These numbers come from Aggregation.java and must stay in sync
      switch (kind) {
        case 0: // SUM
          return cudf::make_sum_aggregation();
        case 1: // PRODUCT
          return cudf::make_product_aggregation();
        case 2: // MIN
          return cudf::make_min_aggregation();
        case 3: // MAX
          return cudf::make_max_aggregation();
        // case 4 COUNT
        case 5: // ANY
          return cudf::make_any_aggregation();
        case 6: // ALL
          return cudf::make_all_aggregation();
        case 7: // SUM_OF_SQUARES
          return cudf::make_sum_of_squares_aggregation();
        case 8: // MEAN
          return cudf::make_mean_aggregation();
        // case 9: VARIANCE
        // case 10: STD
        case 11: // MEDIAN
          return cudf::make_median_aggregation();
        // case 12: QUANTILE
        case 13: // ARGMAX
          return cudf::make_argmax_aggregation();
        case 14: // ARGMIN
          return cudf::make_argmin_aggregation();
        // case 15: NUNIQUE
        // case 16: NTH_ELEMENT
        case 17: // ROW_NUMBER
          return cudf::make_row_number_aggregation();
        // case 18: COLLECT_LIST
        // case 19: COLLECT_SET
        case 20: // MERGE_LISTS
          return cudf::make_merge_lists_aggregation();
        // case 21: MERGE_SETS
        // case 22: LEAD
        // case 23: LAG
        // case 24: PTX
        // case 25: CUDA
        case 26: // M2
          return cudf::make_m2_aggregation();
        case 27: // MERGE_M2
          return cudf::make_merge_m2_aggregation();
        case 28: // RANK
          return cudf::make_rank_aggregation(cudf::rank_method::MIN, {},
                                             cudf::null_policy::INCLUDE);
        case 29: // DENSE_RANK
          return cudf::make_rank_aggregation(cudf::rank_method::DENSE, {},
                                             cudf::null_policy::INCLUDE);
        case 30: // ANSI SQL PERCENT_RANK
          return cudf::make_rank_aggregation(cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE,
                                             {}, cudf::rank_percentage::ONE_NORMALIZED);
        case 33: // HISTOGRAM
          return cudf::make_histogram_aggregation();
        case 34: // MERGE_HISTOGRAM
          return cudf::make_merge_histogram_aggregation();

        default: throw std::logic_error("Unsupported No Parameter Aggregation Operation");
      }
    }();

    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createNthAgg(JNIEnv *env,
                                                                     jclass class_object,
                                                                     jint offset,
                                                                     jboolean include_nulls) {
  try {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret = cudf::make_nth_element_aggregation(
        offset, include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createDdofAgg(JNIEnv *env,
                                                                      jclass class_object,
                                                                      jint kind, jint ddof) {
  try {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 9: // VARIANCE
        ret = cudf::make_variance_aggregation(ddof);
        break;
      case 10: // STD
        ret = cudf::make_std_aggregation(ddof);
        break;
      default: throw std::logic_error("Unsupported DDOF Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createTDigestAgg(JNIEnv *env,
                                                                         jclass class_object,
                                                                         jint kind, jint delta) {
  try {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 31: // TDIGEST
        ret = cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta);
        break;
      case 32: // MERGE_TDIGEST
        ret = cudf::make_merge_tdigest_aggregation<cudf::groupby_aggregation>(delta);
        break;
      default: throw std::logic_error("Unsupported TDigest Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCountLikeAgg(JNIEnv *env,
                                                                           jclass class_object,
                                                                           jint kind,
                                                                           jboolean include_nulls) {
  try {
    cudf::jni::auto_set_device(env);

    cudf::null_policy policy =
        include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 4: // COUNT
        ret = cudf::make_count_aggregation(policy);
        break;
      case 15: // NUNIQUE
        ret = cudf::make_nunique_aggregation(policy);
        break;
      default: throw std::logic_error("Unsupported Count Like Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createQuantAgg(JNIEnv *env,
                                                                       jclass class_object,
                                                                       jint j_method,
                                                                       jdoubleArray j_quantiles) {
  JNI_NULL_CHECK(env, j_quantiles, "quantiles are null", 0);
  try {
    cudf::jni::auto_set_device(env);

    const cudf::jni::native_jdoubleArray quantiles(env, j_quantiles);

    std::vector<double> quants(quantiles.data(), quantiles.data() + quantiles.size());
    cudf::interpolation interp = static_cast<cudf::interpolation>(j_method);

    std::unique_ptr<cudf::aggregation> ret = cudf::make_quantile_aggregation(quants, interp);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createLeadLagAgg(JNIEnv *env,
                                                                         jclass class_object,
                                                                         jint kind, jint offset) {
  try {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 22: // LEAD
        ret = cudf::make_lead_aggregation(offset);
        break;
      case 23: // LAG
        ret = cudf::make_lag_aggregation(offset);
        break;
      default: throw std::logic_error("Unsupported Lead/Lag Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCollectListAgg(
    JNIEnv *env, jclass class_object, jboolean include_nulls) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::null_policy policy =
        include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    std::unique_ptr<cudf::aggregation> ret = cudf::make_collect_list_aggregation(policy);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createCollectSetAgg(JNIEnv *env,
                                                                            jclass class_object,
                                                                            jboolean include_nulls,
                                                                            jboolean nulls_equal,
                                                                            jboolean nans_equal) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::null_policy null_policy =
        include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    cudf::null_equality null_equality =
        nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    cudf::nan_equality nan_equality =
        nans_equal ? cudf::nan_equality::ALL_EQUAL : cudf::nan_equality::UNEQUAL;
    std::unique_ptr<cudf::aggregation> ret =
        cudf::make_collect_set_aggregation(null_policy, null_equality, nan_equality);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createMergeSetsAgg(JNIEnv *env,
                                                                           jclass class_object,
                                                                           jboolean nulls_equal,
                                                                           jboolean nans_equal) {
  try {
    cudf::jni::auto_set_device(env);
    cudf::null_equality null_equality =
        nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    cudf::nan_equality nan_equality =
        nans_equal ? cudf::nan_equality::ALL_EQUAL : cudf::nan_equality::UNEQUAL;
    std::unique_ptr<cudf::aggregation> ret =
        cudf::make_merge_sets_aggregation(null_equality, nan_equality);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
