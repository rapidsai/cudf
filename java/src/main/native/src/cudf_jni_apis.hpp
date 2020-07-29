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
#pragma once

#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

#include "jni_utils.hpp"

namespace cudf {
namespace jni {


jobject contiguous_table_from(JNIEnv *env, cudf::contiguous_split_result &split);

native_jobjectArray<jobject> contiguous_table_array(JNIEnv *env, jsize length);

std::unique_ptr<cudf::aggregation> map_jni_aggregation(jint op);

jlongArray convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &table_result);

/**
 * Allocate a HostMemoryBuffer
 */
jobject allocate_host_buffer(JNIEnv *env, jlong amount, jboolean prefer_pinned);

/**
 * Get the address of a HostMemoryBuffer
 */
jlong get_host_buffer_address(JNIEnv *env, jobject buffer);

/**
 * Get the length of a HostMemoryBuffer
 */
jlong get_host_buffer_length(JNIEnv *env, jobject buffer);

// Get the JNI environment, attaching the current thread to the JVM if necessary. If the thread
// needs to be attached, the thread will automatically detach when the thread terminates.
JNIEnv *get_jni_env(JavaVM *jvm);

/** Set the device to use for cudf */
void set_cudf_device(int device);

/**
 * If the current thread has not set the CUDA device via Cuda.setDevice then this could
 * set the device, throw an exception, or do nothing depending on how the application has
 * configured it via Cuda.setAutoSetDeviceMode.
 */
void auto_set_device(JNIEnv *env);

} // namespace jni
} // namespace cudf
