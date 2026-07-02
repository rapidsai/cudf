/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_jni_internal.hpp"
#include "jni_compiled_expr.hpp"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <utility>

namespace cudf {
namespace jni {
namespace hybrid_scan {

bool are_ranges_contiguous(std::vector<byte_range_info> const& ranges)
{
  if (ranges.size() <= 1) { return true; }
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i - 1].offset() + ranges[i - 1].size() != ranges[i].offset()) { return false; }
  }
  return true;
}

planned_copy_result plan_and_copy_ranges(uint8_t const* host_ptr,
                                         std::vector<byte_range_info> const& ranges,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  planned_copy_result result;
  if (ranges.empty()) { return result; }

  if (are_ranges_contiguous(ranges)) {
    auto const first_offset = ranges.front().offset();
    auto const last_range   = ranges.back();
    auto const total_size   = (last_range.offset() + last_range.size()) - first_offset;
    rmm::device_buffer coalesced_buf(total_size, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(coalesced_buf.data(),
                                  host_ptr + first_offset,
                                  total_size,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    result.spans.reserve(ranges.size());
    auto const* base_ptr = static_cast<uint8_t const*>(coalesced_buf.data());
    for (auto const& range : ranges) {
      auto const offset_in_buffer = range.offset() - first_offset;
      result.spans.emplace_back(base_ptr + offset_in_buffer, range.size());
    }
    result.device_buffers.emplace_back(std::move(coalesced_buf));
  } else {
    result.device_buffers.reserve(ranges.size());
    for (auto const& range : ranges) {
      rmm::device_buffer dev_buf(range.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(dev_buf.data(),
                                    host_ptr + range.offset(),
                                    range.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      result.device_buffers.emplace_back(std::move(dev_buf));
    }
    result.spans.reserve(result.device_buffers.size());
    for (auto const& buf : result.device_buffers) {
      result.spans.emplace_back(static_cast<uint8_t const*>(buf.data()), buf.size());
    }
  }
  return result;
}

cudf::io::parquet_reader_options build_options(JNIEnv* env,
                                               jlong filter_handle,
                                               jobjectArray j_column_names,
                                               jbooleanArray j_read_binary_as_string,
                                               jint time_unit_type_id)
{
  // The hybrid_scan_reader's options builder is constructed without a source_info because
  // the reader works on already-parsed footer bytes (and on byte ranges fetched separately).
  cudf::io::parquet_reader_options_builder builder;

  cudf::jni::native_jstringArray names(env, j_column_names);
  if (!names.is_null() && names.size() > 0) {
    builder = builder.column_names(names.as_cpp_vector());
  }

  // Translate Java's per-column "read binary as string" flags into the C++ schema override
  // hooks. The reader_column_schema mechanism lets callers force binary→string conversion
  // for the i-th projected column.
  cudf::jni::native_jbooleanArray binary_as_str(env, j_read_binary_as_string);
  if (!binary_as_str.is_null() && binary_as_str.size() > 0) {
    std::vector<cudf::io::reader_column_schema> schemas;
    schemas.reserve(binary_as_str.size());
    for (int i = 0; i < binary_as_str.size(); ++i) {
      cudf::io::reader_column_schema s;
      s.set_convert_binary_to_strings(static_cast<bool>(binary_as_str[i]));
      schemas.emplace_back(std::move(s));
    }
    builder = builder.set_column_schema(std::move(schemas));
    binary_as_str.cancel();
  }

  auto opts = builder.convert_strings_to_categories(false)
                .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(time_unit_type_id)))
                .ignore_missing_columns(true)
                .build();

  if (filter_handle != 0) {
    auto const* filter_expr = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(filter_handle);
    opts.set_filter(filter_expr->get_top_expression());
  }

  return opts;
}

row_group_span_holder make_row_group_span(JNIEnv* env, jintArray j_row_groups)
{
  row_group_span_holder h;
  cudf::jni::native_jintArray arr(env, j_row_groups);
  h.storage.reserve(arr.size());
  for (int i = 0; i < arr.size(); ++i) {
    h.storage.push_back(static_cast<cudf::size_type>(arr[i]));
  }
  arr.cancel();
  return h;
}

jlongArray ranges_to_jlong_array(JNIEnv* env, std::vector<byte_range_info> const& ranges)
{
  auto result = env->NewLongArray(ranges.size() * 2);
  if (result == nullptr) { return nullptr; }
  if (ranges.empty()) { return result; }
  std::vector<jlong> data;
  data.reserve(ranges.size() * 2);
  for (auto const& r : ranges) {
    data.push_back(static_cast<jlong>(r.offset()));
    data.push_back(static_cast<jlong>(r.size()));
  }
  env->SetLongArrayRegion(result, 0, data.size(), data.data());
  return result;
}

jintArray sizes_to_jint_array(JNIEnv* env, std::vector<cudf::size_type> const& vals)
{
  auto result = env->NewIntArray(vals.size());
  if (result == nullptr) { return nullptr; }
  if (vals.empty()) { return result; }
  // jint is int32_t; size_type is also int32_t. Static-cast just to be explicit.
  std::vector<jint> j(vals.begin(), vals.end());
  env->SetIntArrayRegion(result, 0, j.size(), j.data());
  return result;
}

std::vector<cudf::device_span<uint8_t const>> make_device_spans(JNIEnv* env,
                                                                jlongArray j_addrs,
                                                                jlongArray j_lens)
{
  cudf::jni::native_jlongArray addrs(env, j_addrs);
  cudf::jni::native_jlongArray lens(env, j_lens);
  CUDF_EXPECTS(addrs.size() == lens.size(), "addrs and lens arrays must have the same length");
  std::vector<cudf::device_span<uint8_t const>> out;
  out.reserve(addrs.size());
  for (int i = 0; i < addrs.size(); ++i) {
    out.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]),
                     checked_size_t(env, lens[i], "byte range length"));
  }
  addrs.cancel();
  lens.cancel();
  return out;
}

std::size_t checked_size_t(JNIEnv* env, jlong value, char const* name)
{
  if (value < 0) {
    auto const msg = std::string(name) + " must be non-negative, got " + std::to_string(value);
    cudf::jni::throw_java_exception(env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, msg.c_str());
  }
  return static_cast<std::size_t>(value);
}

}  // namespace hybrid_scan
}  // namespace jni
}  // namespace cudf
