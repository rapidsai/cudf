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
#include "cudf/copying.hpp"
#include "cudf/quantiles.hpp"
#include "cudf/replace.hpp"
#include "cudf/rolling.hpp"

#include "jni_utils.hpp"

using unique_nvcat_ptr = std::unique_ptr<NVCategory, decltype(&NVCategory::destroy)>;
using unique_nvstr_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

namespace cudf {
namespace jni {
static jlongArray put_strings_on_host(JNIEnv *env, NVStrings *nvstr) {
  cudf::jni::native_jlongArray ret(env, 4);
  unsigned int numstrs = nvstr->size();
  size_t strdata_size = nvstr->memsize();
  size_t offset_size = sizeof(int) * (numstrs + 1);
  std::unique_ptr<char, decltype(free) *> strdata(
      static_cast<char *>(malloc(sizeof(char) * strdata_size)), free);
  std::unique_ptr<int, decltype(free) *> offsetdata(
      static_cast<int *>(malloc(sizeof(int) * (numstrs + 1))), free);
  nvstr->create_offsets(strdata.get(), offsetdata.get(), nullptr, false);
  ret[0] = reinterpret_cast<jlong>(strdata.get());
  ret[1] = strdata_size;
  ret[2] = reinterpret_cast<jlong>(offsetdata.get());
  ret[3] = offset_size;
  strdata.release();
  offsetdata.release();
  return ret.get_jArray();
}
} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_allocateCudfColumn(JNIEnv *env,
                                                                            jobject j_object) {
  try {
    return reinterpret_cast<jlong>(calloc(1, sizeof(gdf_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_freeCudfColumn(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong handle,
                                                                       jboolean deep_clean) {
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  if (column != NULL) {
    if (deep_clean) {
      gdf_column_free(column);
    } else if (column->dtype == GDF_STRING) {
      NVStrings::destroy(static_cast<NVStrings *>(column->data));
    } else if (column->dtype == GDF_STRING_CATEGORY) {
      NVCategory::destroy(static_cast<NVCategory *>(column->dtype_info.category));
    }
    free(column->col_name);
  }
  free(column);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getDataPtr(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return reinterpret_cast<jlong>(column->data);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getValidPtr(JNIEnv *env, jobject j_object,
                                                                     jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return reinterpret_cast<jlong>(column->valid);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getRowCount(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return static_cast<jint>(column->size);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNullCount(JNIEnv *env, jobject j_object,
                                                                     jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->null_count;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getDTypeInternal(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->dtype;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getTimeUnitInternal(JNIEnv *env,
                                                                            jobject j_object,
                                                                            jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->dtype_info.time_unit;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_cudfColumnViewAugmented(
    JNIEnv *env, jobject, jlong handle, jlong data_ptr, jlong j_valid, jint size, jint dtype,
    jint null_count, jint time_unit) {
  JNI_NULL_CHECK(env, handle, "column is null", );
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  void *data = reinterpret_cast<void *>(data_ptr);
  gdf_valid_type *valid = reinterpret_cast<gdf_valid_type *>(j_valid);
  gdf_dtype c_dtype = static_cast<gdf_dtype>(dtype);
  gdf_dtype_extra_info info{};
  info.time_unit = static_cast<gdf_time_unit>(time_unit);
  JNI_GDF_TRY(env, ,
              gdf_column_view_augmented(column, data, valid, size, c_dtype, null_count, info));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_cudfColumnViewStrings(
    JNIEnv *env, jobject, jlong handle, jlong data_ptr, jboolean data_ptr_on_host,
    jlong host_offsets_ptr, jboolean reset_offsets_to_zero, jlong device_valid_ptr,
    jlong device_output_data_ptr, jint size, jint jdtype, jint null_count) {
  JNI_NULL_CHECK(env, handle, "column is null", );
  JNI_NULL_CHECK(env, data_ptr, "string data is null", );
  JNI_NULL_CHECK(env, host_offsets_ptr, "host offsets is null", );

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    char *data = reinterpret_cast<char *>(data_ptr);
    uint32_t *host_offsets = reinterpret_cast<uint32_t *>(host_offsets_ptr);

    uint32_t data_size = host_offsets[size];
    if (reset_offsets_to_zero) {
      data_size -= host_offsets[0];
    }

    gdf_valid_type *valid = reinterpret_cast<gdf_valid_type *>(device_valid_ptr);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_dtype_extra_info info{};

    // NOTE: Even though the caller API is tailor-made to use
    // NVCategory::create_from_offsets or NVStrings::create_from_offsets, it's much faster to
    // use create_from_index, block-transferring the host string data to the device first.

    char *device_data = nullptr;
    cudf::jni::jni_rmm_unique_ptr<char> dev_string_data_holder;
    if (data_ptr_on_host) {
      dev_string_data_holder = cudf::jni::jni_rmm_alloc<char>(env, data_size);
      JNI_CUDA_TRY(
          env, ,
          cudaMemcpyAsync(dev_string_data_holder.get(), data, data_size, cudaMemcpyHostToDevice));
      device_data = dev_string_data_holder.get();
    } else {
      device_data = data;
    }

    uint32_t offset_amount_to_subtract = 0;
    if (reset_offsets_to_zero) {
      offset_amount_to_subtract = host_offsets[0];
    }
    std::vector<std::pair<const char *, size_t>> index{};
    index.reserve(size);
    for (int i = 0; i < size; i++) {
      index[i].first = device_data + host_offsets[i] - offset_amount_to_subtract;
      index[i].second = host_offsets[i + 1] - host_offsets[i];
    }

    if (dtype == GDF_STRING) {
      unique_nvstr_ptr strings(NVStrings::create_from_index(index.data(), size, false),
                               &NVStrings::destroy);
      JNI_GDF_TRY(
          env, ,
          gdf_column_view_augmented(column, strings.get(), valid, size, dtype, null_count, info));
      strings.release();
    } else if (dtype == GDF_STRING_CATEGORY) {
      JNI_NULL_CHECK(env, device_output_data_ptr, "device data pointer is null", );
      int *cat_data = reinterpret_cast<int *>(device_output_data_ptr);
      unique_nvcat_ptr cat(NVCategory::create_from_index(index.data(), size, false),
                           &NVCategory::destroy);
      info.category = cat.get();
      if (size != cat->get_values(cat_data, true)) {
        JNI_THROW_NEW(env, "java/lang/IllegalStateException",
                      "Internal Error copying str cat data", );
      }
      JNI_GDF_TRY(
          env, , gdf_column_view_augmented(column, cat_data, valid, size, dtype, null_count, info));
      cat.release();
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, );
}

NVStrings::timestamp_units translateTimestampUnit(gdf_time_unit time_unit) {
  switch (time_unit) {
    case TIME_UNIT_s: return NVStrings::seconds;
    case TIME_UNIT_ms: return NVStrings::ms;
    case TIME_UNIT_us: return NVStrings::us;
    case TIME_UNIT_ns: return NVStrings::ns;
  }
  throw std::logic_error("UNSUPPORTED COLUMN VECTOR TIMESTAMP UNIT");
}

// Resolve the mutated dictionary with the original index values
// gathering column metadata from the most relevant sources
cudf::jni::gdf_column_wrapper gather_mutated_category(gdf_column *dict_result, gdf_column *column) {
  std::vector<gdf_column*> vec {dict_result};
  cudf::table tmp_table(vec);

  cudf::jni::gdf_column_wrapper result(column->size, dict_result->dtype, column->null_count != 0);
  gdf_column * result_ptr = result.get();
  std::vector<gdf_column*> out_vec {result_ptr};
  cudf::table output_table(out_vec);

  gather(&tmp_table, static_cast<gdf_index_type *>(column->data), &output_table);
  if (column->null_count > 0) {
    CUDA_TRY(cudaMemcpy(result_ptr->valid, column->valid,
                        gdf_num_bitmask_elements(column->size), cudaMemcpyDeviceToDevice));
    result_ptr->null_count = column->null_count;
  }
  result_ptr->dtype_info.time_unit = dict_result->dtype_info.time_unit;

  return result;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringTimestampToTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jint time_unit, jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  if (formatObj == NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "must pass a timestamp format string",
                  0)
  }

  try {
    cudf::jni::native_jstring format(env, formatObj);

    gdf_column *column = reinterpret_cast<gdf_column *>(handle);

    if (column->dtype == GDF_STRING) {
      cudf::jni::gdf_column_wrapper output(column->size, gdf_dtype::GDF_TIMESTAMP,
                                           column->null_count != 0);
      output->dtype_info.time_unit = (gdf_time_unit)time_unit;

      if (column->size > 0) {
        NVStrings *strings = static_cast<NVStrings *>(column->data);
        JNI_ARG_CHECK(env, column->size == strings->size(),
                      "NVStrings size and gdf_column size mismatch", 0);
        int result = strings->timestamp2long(format.get(),
                                             translateTimestampUnit(output->dtype_info.time_unit),
                                             static_cast<unsigned long *>(output->data));
        if (result == -1) {
          throw std::logic_error("timestamp2long returned with errors");
        }
        if (column->null_count > 0) {
          CUDA_TRY(cudaMemcpy(output->valid, column->valid, gdf_num_bitmask_elements(column->size),
                              cudaMemcpyDeviceToDevice));
          output->null_count = column->null_count;
        }
      }
      return reinterpret_cast<jlong>(output.release());
    } else if (column->dtype == GDF_STRING_CATEGORY) {
      if (column->size <= 0) {
        // special case for empty column
        cudf::jni::gdf_column_wrapper result(column->size, gdf_dtype::GDF_TIMESTAMP,
                                             column->null_count != 0);
        return reinterpret_cast<jlong>(result.release());
      }

      // Do the operation on the dictionary
      NVCategory *cats = static_cast<NVCategory *>(column->dtype_info.category);
      unique_nvstr_ptr keys (cats->get_keys(), &NVStrings::destroy);
      unsigned int dict_size = keys->size();

      cudf::jni::gdf_column_wrapper dict_result(dict_size, gdf_dtype::GDF_TIMESTAMP, false);
      dict_result->dtype_info.time_unit = (gdf_time_unit)time_unit;
      int err_val = keys->timestamp2long(format.get(),
                                        translateTimestampUnit(dict_result->dtype_info.time_unit),
                                        static_cast<unsigned long *>(dict_result->data));
      if (err_val == -1) {
         throw std::logic_error("timestamp2long returned with errors");
      }

      cudf::jni::gdf_column_wrapper res = gather_mutated_category(dict_result.get(), column);

      return reinterpret_cast<jlong>(res.release());
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, 0);

  return 0;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_upperStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);

    if (column->dtype == GDF_STRING) {
      if (column->size == 0) {
        cudf::jni::gdf_column_wrapper output(0, column->dtype, 0, nullptr, nullptr,
                                             column->dtype_info.category);
        return reinterpret_cast<jlong>(output.release());
      }

      NVStrings *inst = static_cast<NVStrings *>(column->data);
      JNI_ARG_CHECK(env, column->size == inst->size(),
                    "NVStrings size and gdf_column size mismatch", 0);
      unique_nvstr_ptr res(inst->upper(), &NVStrings::destroy);

      if (column->null_count == 0) {
        cudf::jni::gdf_column_wrapper output(column->size, column->dtype, column->null_count,
                                             res.release(), nullptr, column->dtype_info.category);
        return reinterpret_cast<jlong>(output.release());
      }

      cudf::jni::jni_rmm_unique_ptr<gdf_valid_type> valid_copy =
          cudf::jni::jni_rmm_alloc<gdf_valid_type>(env, gdf_valid_allocation_size(column->size));
      CUDA_TRY(cudaMemcpy(valid_copy.get(), column->valid, gdf_num_bitmask_elements(column->size),
                          cudaMemcpyDeviceToDevice));
      cudf::jni::gdf_column_wrapper output(column->size, column->dtype, column->null_count,
                                           res.release(), valid_copy.release(),
                                           column->dtype_info.category);
      return reinterpret_cast<jlong>(output.release());
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, 0);

  return 0;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_lowerStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);

    if (column->dtype == GDF_STRING) {
      if (column->size == 0) {
        cudf::jni::gdf_column_wrapper output(0, column->dtype, 0, nullptr, nullptr,
                                             column->dtype_info.category);
        return reinterpret_cast<jlong>(output.release());
      }

      NVStrings *inst = static_cast<NVStrings *>(column->data);
      JNI_ARG_CHECK(env, column->size == inst->size(),
                    "NVStrings size and gdf_column size mismatch", 0);
      unique_nvstr_ptr res(inst->lower(), &NVStrings::destroy);

      if (column->null_count == 0) {
        cudf::jni::gdf_column_wrapper output(column->size, column->dtype, column->null_count,
                                             res.release(), nullptr, column->dtype_info.category);
        return reinterpret_cast<jlong>(output.release());
      }

      cudf::jni::jni_rmm_unique_ptr<gdf_valid_type> valid_copy =
          cudf::jni::jni_rmm_alloc<gdf_valid_type>(env, gdf_valid_allocation_size(column->size));
      CUDA_TRY(cudaMemcpy(valid_copy.get(), column->valid, gdf_num_bitmask_elements(column->size),
                          cudaMemcpyDeviceToDevice));
      cudf::jni::gdf_column_wrapper output(column->size, column->dtype, column->null_count,
                                           res.release(), valid_copy.release(),
                                           column->dtype_info.category);
      return reinterpret_cast<jlong>(output.release());
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, 0);

  return 0;
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_ColumnVector_getStringDataAndOffsetsBack(JNIEnv *env, jobject, jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", NULL);

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    gdf_dtype dtype = column->dtype;
    // data address, data length, offsets address, offsets length
    if (dtype == GDF_STRING) {
      return cudf::jni::put_strings_on_host(env, static_cast<NVStrings *>(column->data));
    } else if (dtype == GDF_STRING_CATEGORY) {
      NVCategory *cat = static_cast<NVCategory *>(column->dtype_info.category);
      unique_nvstr_ptr nvstr(cat->to_strings(), &NVStrings::destroy);
      return cudf::jni::put_strings_on_host(env, nvstr.get());
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatenate(JNIEnv *env, jclass clazz,
                                                                     jlongArray column_handles) {
  JNI_NULL_CHECK(env, column_handles, "input columns are null", 0);
  try {
    cudf::jni::native_jpointerArray<gdf_column> columns(env, column_handles);
    size_t total_size = 0;
    bool need_validity = false;
    for (int i = 0; i < columns.size(); ++i) {
      total_size += columns[i]->size;
      // Should be checking for null_count != 0 but libcudf is checking valid != nullptr
      need_validity |= columns[i]->valid != nullptr;
    }
    if (total_size != static_cast<gdf_size_type>(total_size)) {
      cudf::jni::throw_java_exception(env, "java/lang/IllegalArgumentException",
                                      "resulting column is too large");
    }
    cudf::jni::gdf_column_wrapper outcol(total_size, columns[0]->dtype, need_validity, true);
    JNI_GDF_TRY(env, 0, gdf_column_concat(outcol.get(), columns.data(), columns.size()));
    if (outcol->dtype == GDF_TIMESTAMP) {
      outcol->dtype_info.time_unit = columns[0]->dtype_info.time_unit;
    }
    return reinterpret_cast<jlong>(outcol.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ColumnVector_exactQuantile(JNIEnv *env, jclass clazz,
                                                                         jlong input_column,
                                                                         jint quantile_method,
                                                                         jdouble quantile) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    cudf::interpolation n_quantile_method = static_cast<cudf::interpolation>(quantile_method);
    gdf_context ctxt{0, GDF_SORT, 0, 0};
    gdf_scalar result{};
    JNI_GDF_TRY(env, NULL,
                cudf::quantile_exact(n_input_column, n_quantile_method, quantile, &result, &ctxt));
    return cudf::jni::jscalar_from_scalar(env, result, n_input_column->dtype_info.time_unit);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ColumnVector_approxQuantile(JNIEnv *env, jclass clazz,
                                                                          jlong input_column,
                                                                          jdouble quantile) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    gdf_context ctxt{0, GDF_SORT, 0, 0};
    gdf_scalar result{};
    JNI_GDF_TRY(env, NULL, cudf::quantile_approx(n_input_column, quantile, &result, &ctxt));
    return cudf::jni::jscalar_from_scalar(env, result, n_input_column->dtype_info.time_unit);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_rollingWindow(
    JNIEnv *env, jclass clazz, jlong input_column, jint window, jint min_periods,
    jint forward_window, jint agg_type, jlong window_col, jlong min_periods_col,
    jlong forward_window_col) {

  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    gdf_column *n_window_col = reinterpret_cast<gdf_column *>(window_col);
    gdf_column *n_min_periods_col = reinterpret_cast<gdf_column *>(min_periods_col);
    gdf_column *n_forward_window_col = reinterpret_cast<gdf_column *>(forward_window_col);

    gdf_column *result = cudf::rolling_window(
        *n_input_column, static_cast<gdf_size_type>(window),
        static_cast<gdf_size_type>(min_periods), static_cast<gdf_size_type>(forward_window),
        static_cast<gdf_agg_op>(agg_type),
        n_window_col == nullptr ? nullptr : reinterpret_cast<gdf_size_type *>(n_window_col->data),
        n_min_periods_col == nullptr ? nullptr :
                                       reinterpret_cast<gdf_size_type *>(n_min_periods_col->data),
        n_forward_window_col == nullptr ?
            nullptr :
            reinterpret_cast<gdf_size_type *>(n_forward_window_col->data));
    return reinterpret_cast<jlong>(result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_cudfSlice(JNIEnv *env, jclass clazz,
                                                                        jlong input_column,
                                                                        jlong slice_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, slice_indices, "slice indices are null", 0);

  gdf_column *n_column = reinterpret_cast<gdf_column *>(input_column);
  gdf_column *n_slice_indices = reinterpret_cast<gdf_column *>(slice_indices);

  try {
    std::vector<gdf_column *> result = cudf::slice(
        *n_column, static_cast<gdf_index_type *>(n_slice_indices->data), n_slice_indices->size);
    cudf::jni::native_jlongArray n_result(env, reinterpret_cast<jlong *>(result.data()),
                                          result.size());
    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_cudfLengths(JNIEnv *env, jclass clazz,
                                                                     jlong column_handle) {
  JNI_NULL_CHECK(env, column_handle, "input column is null", 0);
  try {
    gdf_column *n_column = reinterpret_cast<gdf_column *>(column_handle);
    cudf::jni::gdf_column_wrapper lengths(n_column->size, gdf_dtype::GDF_INT32,
                                          n_column->null_count != 0);
    if (n_column->size > 0) {
      NVStrings *strings = static_cast<NVStrings *>(n_column->data);
      JNI_ARG_CHECK(env, n_column->size == strings->size(),
                    "NVStrings size and gdf_column size mismatch", 0);
      strings->len(static_cast<int *>(lengths->data));
      if (n_column->null_count > 0) {
        CUDA_TRY(cudaMemcpy(lengths->valid, n_column->valid,
                            gdf_num_bitmask_elements(n_column->size), cudaMemcpyDeviceToDevice));
        lengths->null_count = n_column->null_count;
      }
    }
    return reinterpret_cast<jlong>(lengths.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_cudfByteCount(JNIEnv *env, jclass clazz,
                                                                       jlong column_handle) {
  JNI_NULL_CHECK(env, column_handle, "input column is null", 0);
  try {
    gdf_column *n_column = reinterpret_cast<gdf_column *>(column_handle);
    cudf::jni::gdf_column_wrapper byte_counts_vector(n_column->size, gdf_dtype::GDF_INT32,
                                                     n_column->null_count != 0);

    if (n_column->size > 0) {
      NVStrings *strings = static_cast<NVStrings *>(n_column->data);
      JNI_ARG_CHECK(env, n_column->size == strings->size(),
                    "NVStrings size and gdf_column size mismatch", 0);
      strings->byte_count(static_cast<int *>(byte_counts_vector->data));
      if (n_column->null_count > 0) {
        CUDA_TRY(cudaMemcpy(byte_counts_vector->valid, n_column->valid,
                            gdf_num_bitmask_elements(n_column->size), cudaMemcpyDeviceToDevice));
        byte_counts_vector->null_count = n_column->null_count;
      }
    }
    return reinterpret_cast<jlong>(byte_counts_vector.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_hash(JNIEnv *env, jclass clazz,
                                                              jlong column_handle, jint hash_type) {
  JNI_NULL_CHECK(env, column_handle, "input column is null", 0);
  try {
    gdf_column *n_column = reinterpret_cast<gdf_column *>(column_handle);
    gdf_hash_func hash_func = static_cast<gdf_hash_func>(hash_type);

    if (n_column->dtype == GDF_STRING) {
      cudf::jni::gdf_column_wrapper result(n_column->size, gdf_dtype::GDF_INT32,
                                           n_column->null_count != 0);

      if (n_column->size > 0) {
        NVStrings *strings = static_cast<NVStrings *>(n_column->data);
        JNI_ARG_CHECK(env, n_column->size == strings->size(),
                      "NVStrings size and gdf_column size mismatch", 0);
        strings->hash(static_cast<unsigned int *>(result->data));
        if (n_column->null_count > 0) {
          CUDA_TRY(cudaMemcpy(result->valid, n_column->valid,
                              gdf_num_bitmask_elements(n_column->size), cudaMemcpyDeviceToDevice));
          result->null_count = n_column->null_count;
        }
      }
      return reinterpret_cast<jlong>(result.release());
    } else if (n_column->dtype == GDF_STRING_CATEGORY) {
      if (n_column->size <= 0) {
        // special case for empty column
        cudf::jni::gdf_column_wrapper result(n_column->size, gdf_dtype::GDF_INT32,
                                             n_column->null_count != 0);
        return reinterpret_cast<jlong>(result.release());
      }
      // Do the operation on the dictionary
      NVCategory *cats = static_cast<NVCategory *>(n_column->dtype_info.category);
      unique_nvstr_ptr keys(cats->get_keys(), &NVStrings::destroy);
      unsigned int dict_size = keys->size();

      cudf::jni::gdf_column_wrapper dict_result(dict_size, gdf_dtype::GDF_INT32, false);
      keys->hash(static_cast<unsigned int *>(dict_result->data));

      // Now we need to gather the data to a final answer
      cudf::jni::gdf_column_wrapper result = gather_mutated_category(dict_result.get(), n_column);
      

      return reinterpret_cast<jlong>(result.release());
    } else { // all others
      cudf::jni::gdf_column_wrapper result(n_column->size, gdf_dtype::GDF_INT32,
                                           n_column->null_count > 0);
      CUDF_TRY(gdf_hash(1, &n_column, hash_func, nullptr, result.get()));
      if (n_column->null_count > 0) {
        gdf_column *result_ptr = result.get();
        CUDA_TRY(cudaMemcpy(result_ptr->valid, n_column->valid,
                            gdf_num_bitmask_elements(n_column->size), cudaMemcpyDeviceToDevice));
        result_ptr->null_count = n_column->null_count;
      }

      return reinterpret_cast<jlong>(result.release());
    }
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

  try {
    gdf_column *input_column = reinterpret_cast<gdf_column *>(input_handle);
    gdf_column *old_values_column = reinterpret_cast<gdf_column *>(old_values_handle);
    gdf_column *new_values_column = reinterpret_cast<gdf_column *>(new_values_handle);

    std::unique_ptr<gdf_column, decltype(free) *> result(
        static_cast<gdf_column *>(calloc(1, sizeof(gdf_column))), free);
    if (result.get() == nullptr) {
      cudf::jni::throw_java_exception(env, "java/lang/OutOfMemoryError",
                                      "Could not allocate native memory");
    }
    *result.get() =
        cudf::find_and_replace_all(*input_column, *old_values_column, *new_values_column);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
