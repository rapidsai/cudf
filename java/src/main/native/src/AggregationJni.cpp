/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Aggregation_close(JNIEnv *env,
                                                             jclass class_object,
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
    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 0: // SUM
        ret = cudf::make_sum_aggregation();
        break;
      case 1: // PRODUCT
        ret = cudf::make_product_aggregation();
        break;
      case 2: // MIN
        ret = cudf::make_min_aggregation();
        break;
      case 3: // MAX
        ret = cudf::make_max_aggregation();
        break;
      //case 4 COUNT
      case 5: // ANY
        ret = cudf::make_any_aggregation();
        break;
      case 6: // ALL
        ret = cudf::make_all_aggregation();
        break;
      case 7: // SUM_OF_SQUARES
        ret = cudf::make_sum_of_squares_aggregation();
        break;
      case 8: // MEAN
        ret = cudf::make_mean_aggregation();
        break;
      // case 9: VARIANCE
      // case 10: STD
      case 11: // MEDIAN
        ret = cudf::make_median_aggregation();
        break;
      // case 12: QUANTILE
      case 13: // ARGMAX
        ret = cudf::make_argmax_aggregation();
        break;
      case 14: // ARGMIN
        ret = cudf::make_argmin_aggregation();
        break;
      // case 15: NUNIQUE
      // case 16: NTH_ELEMENT
      case 17: // ROW_NUMBER
        ret = cudf::make_row_number_aggregation();
        break;
      case 18: // COLLECT
        ret = cudf::make_collect_aggregation();
        break;
      // case 19: LEAD
      // case 20: LAG
      // case 21: PTX
      // case 22: CUDA
      default: throw std::logic_error("Unsupported No Parameter Aggregation Operation");
    }

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

    std::unique_ptr<cudf::aggregation> ret = 
        cudf::make_nth_element_aggregation(offset,
                include_nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation_createDdofAgg(JNIEnv *env,
                                                                      jclass class_object,
                                                                      jint kind,
                                                                      jint ddof) {
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
                                                                         jint kind,
                                                                         jint offset) {
  try {
    cudf::jni::auto_set_device(env);

    std::unique_ptr<cudf::aggregation> ret;
    // These numbers come from Aggregation.java and must stay in sync
    switch (kind) {
      case 19: // LEAD
        ret = cudf::make_lead_aggregation(offset);
        break;
      case 20: // LAG
        ret = cudf::make_lag_aggregation(offset);
        break;
      default: throw std::logic_error("Unsupported Lead/Lag Aggregation Operation");
    }
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
