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
    JNIEnv *env, jobject, jlong handle, jlong host_data_ptr, jlong host_offsets_ptr,
    jlong device_valid_ptr, jlong device_data_ptr, jint size, jint jdtype, jint null_count) {
  JNI_NULL_CHECK(env, handle, "column is null", );
  JNI_NULL_CHECK(env, host_data_ptr, "host data is null", );
  JNI_NULL_CHECK(env, host_offsets_ptr, "host offsets is null", );

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    char *host_data = reinterpret_cast<char *>(host_data_ptr);
    uint32_t *host_offsets = reinterpret_cast<uint32_t *>(host_offsets_ptr);
    uint32_t host_data_size = host_offsets[size];

    gdf_valid_type *valid = reinterpret_cast<gdf_valid_type *>(device_valid_ptr);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_dtype_extra_info info{};

    // NOTE: Even though the caller API is tailor-made to use
    // NVCategory::create_from_offsets or NVStrings::create_from_offsets, it's much faster to
    // use create_from_index, block-transferring the host string data to the device first.

    auto device_data = cudf::jni::jni_rmm_alloc<char>(env, host_data_size);
    JNI_CUDA_TRY(
        env, ,
        cudaMemcpyAsync(device_data.get(), host_data, host_data_size, cudaMemcpyHostToDevice));

    std::vector<std::pair<const char *, size_t>> index{};
    index.reserve(size);
    for (int i = 0; i < size; i++) {
      index[i].first = device_data.get() + host_offsets[i];
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
      JNI_NULL_CHECK(env, device_data_ptr, "device data pointer is null", );
      int *cat_data = reinterpret_cast<int *>(device_data_ptr);
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
    cudf::jni::gdf_column_wrapper outcol(total_size, columns[0]->dtype, need_validity);
    JNI_GDF_TRY(env, 0, gdf_column_concat(outcol.get(), columns.data(), columns.size()));
    return reinterpret_cast<jlong>(outcol.release());
  }
  CATCH_STD(env, 0);
}
} // extern "C"
