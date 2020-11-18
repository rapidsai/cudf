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

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_fromScalar(JNIEnv *env, jclass,
                                                                    jlong j_scalar,
                                                                    jint row_count) {
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto scalar_val = reinterpret_cast<cudf::scalar const *>(j_scalar);
    auto dtype = scalar_val->type();
    cudf::mask_state mask_state =
        scalar_val->is_valid() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL;
    std::unique_ptr<cudf::column> col;
    if (row_count == 0) {
      col = cudf::make_empty_column(dtype);
    } else if (cudf::is_fixed_width(dtype)) {
      col = cudf::make_fixed_width_column(dtype, row_count, mask_state);
      auto mut_view = col->mutable_view();
      cudf::fill_in_place(mut_view, 0, row_count, *scalar_val);
    } else if (dtype.id() == cudf::type_id::STRING) {
      // create a string column of all empty strings to fill (cheapest string column to create)
      auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, row_count + 1,
                                               cudf::mask_state::UNALLOCATED);
      auto data = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8});
      auto mask_buffer = cudf::create_null_mask(row_count, cudf::mask_state::UNALLOCATED);
      auto str_col = cudf::make_strings_column(row_count, std::move(offsets), std::move(data), 0,
                                               std::move(mask_buffer));

      col = cudf::fill(str_col->view(), 0, row_count, *scalar_val);
    } else {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_hash(JNIEnv *env,
                                                              jobject j_object,
                                                              jlongArray column_handles,
                                                              jint hash_function_id) {
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  try {
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, column_handles);
    std::vector<cudf::column_view> column_views;
    std::transform(n_cudf_columns.data(), n_cudf_columns.data() + n_cudf_columns.size(),
                   std::back_inserter(column_views),
                   [](auto const &p_column) { return *p_column; });
    cudf::table_view *input_table = new cudf::table_view(column_views);

    std::unique_ptr<cudf::column> result = cudf::hash(*input_table, static_cast<cudf::hash_id>(hash_function_id));
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

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeEmptyCudfColumn(JNIEnv *env,
                                                                             jclass,
                                                                             jint j_type,
                                                                             jint scale) {

  try {
    cudf::jni::auto_set_device(env);
    cudf::type_id n_type = static_cast<cudf::type_id>(j_type);
    cudf::data_type n_data_type = cudf::jni::make_data_type(j_type, scale);

    std::unique_ptr<cudf::column> column(cudf::make_empty_column(n_data_type));
    return reinterpret_cast<jlong>(column.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_makeNumericCudfColumn(
    JNIEnv *env, jobject j_object, jint j_type, jint j_size, jint j_mask_state) {

  JNI_ARG_CHECK(env, (j_size != 0), "size is 0", 0);

  try {
    cudf::jni::auto_set_device(env);
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
    cudf::jni::auto_set_device(env);
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
    cudf::jni::auto_set_device(env);
    cudf::size_type *host_offsets = reinterpret_cast<cudf::size_type *>(j_offset_data);
    char *n_char_data = reinterpret_cast<char *>(j_char_data);
    cudf::size_type n_data_size = host_offsets[size];
    cudf::bitmask_type *n_validity = reinterpret_cast<cudf::bitmask_type *>(j_valid_data);

    if (n_validity == nullptr) {
      j_null_count = 0;
    }

    std::unique_ptr<cudf::column> offsets = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, size + 1, cudf::mask_state::UNALLOCATED);
    auto offsets_view = offsets->mutable_view();
    JNI_CUDA_TRY(env, 0,
                 cudaMemcpyAsync(offsets_view.data<int32_t>(), host_offsets,
                                 (size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::unique_ptr<cudf::column> data = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT8}, n_data_size, cudf::mask_state::UNALLOCATED);
    auto data_view = data->mutable_view();
    JNI_CUDA_TRY(env, 0,
                 cudaMemcpyAsync(data_view.data<int8_t>(), n_char_data, n_data_size,
                                 cudaMemcpyHostToDevice));

    std::unique_ptr<cudf::column> column;
    if (j_null_count == 0) {
      column =
          cudf::make_strings_column(size, std::move(offsets), std::move(data), j_null_count, {});
    } else {
      cudf::size_type bytes = (cudf::word_index(size) + 1) * sizeof(cudf::bitmask_type);
      rmm::device_buffer dev_validity(bytes);
      JNI_CUDA_TRY(env, 0,
                   cudaMemcpyAsync(dev_validity.data(), n_validity, bytes, cudaMemcpyHostToDevice));

      column = cudf::make_strings_column(size, std::move(offsets), std::move(data), j_null_count,
                                         std::move(dev_validity));
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
    cudf::jni::auto_set_device(env);
    cudf::column *column = reinterpret_cast<cudf::column *>(handle);
    return static_cast<jint>(column->null_count());
  }
  CATCH_STD(env, 0);
}
} // extern "C"
