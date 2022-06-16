/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <mutex>
#include <unordered_map>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Create an offsets column to be a child of a strings column.
 * This will set the offsets values by executing scan on the provided
 * Iterator.
 *
 * @tparam Iterator Used as input to scan to set the offset values.
 * @param begin The beginning of the input sequence
 * @param end The end of the input sequence
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return offsets child column for strings column
 */
template <typename InputIterator>
std::unique_ptr<column> make_offsets_child_column(
  InputIterator begin,
  InputIterator end,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(begin < end, "Invalid iterator range");
  auto count = thrust::distance(begin, end);
  auto offsets_column =
    make_numeric_column(data_type{type_id::INT32}, count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets    = offsets_view.template data<int32_t>();
  // Using inclusive-scan to compute last entry which is the total size.
  // Exclusive-scan is possible but will not compute that last entry.
  // Rather than manually computing the final offset using values in device memory,
  // we use inclusive-scan on a shifted output (d_offsets+1) and then set the first
  // offset values to zero manually.
  thrust::inclusive_scan(rmm::exec_policy(stream), begin, end, d_offsets + 1);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream.value()));
  return offsets_column;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param input Data to copy from.
 * @param bytes Number of bytes to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_and_increment(char* buffer, const char* input, size_type bytes)
{
  memcpy(buffer, input, bytes);
  return buffer + bytes;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_string(char* buffer, const string_view& d_string)
{
  return copy_and_increment(buffer, d_string.data(), d_string.size_bytes());
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param exec_size Number of rows for executing the `size_and_exec_fn` function.
 * @param strings_count Number of strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(
  SizeAndExecuteFunction size_and_exec_fn,
  size_type exec_size,
  size_type strings_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  // This is called twice -- once for offsets and once for chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [exec_size, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       exec_size,
                       size_and_exec_fn);
  };

  // Compute the offsets values
  for_each_fn(size_and_exec_fn);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Now build the chars column
  auto const bytes = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
  std::unique_ptr<column> chars_column = create_chars_child_column(bytes, stream, mr);

  // Execute the function fn again to fill the chars column.
  // Note that if the output chars column has zero size, the function fn should not be called to
  // avoid accidentally overwriting the offsets.
  if (bytes > 0) {
    size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();
    for_each_fn(size_and_exec_fn);
  }

  return std::pair(std::move(offsets_column), std::move(chars_column));
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param strings_count Number of strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(
  SizeAndExecuteFunction size_and_exec_fn,
  size_type strings_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return make_strings_children(size_and_exec_fn, strings_count, strings_count, stream, mr);
}

// This template is a thin wrapper around per-context singleton objects.
// It maintains a single object for each CUDA context.
template <typename TableType>
class per_context_cache {
 public:
  // Find an object cached for a current CUDA context.
  // If there is no object available in the cache, it calls the initializer
  // `init` to create a new one and cache it for later uses.
  template <typename Initializer>
  TableType* find_or_initialize(const Initializer& init)
  {
    CUcontext c;
    cuCtxGetCurrent(&c);
    auto finder = cache_.find(c);
    if (finder == cache_.end()) {
      TableType* result = init();
      cache_[c]         = result;
      return result;
    } else
      return finder->second;
  }

 private:
  std::unordered_map<CUcontext, TableType*> cache_;
};

// This template is a thread-safe version of per_context_cache.
template <typename TableType>
class thread_safe_per_context_cache : public per_context_cache<TableType> {
 public:
  template <typename Initializer>
  TableType* find_or_initialize(const Initializer& init)
  {
    std::lock_guard<std::mutex> guard(mutex);
    return per_context_cache<TableType>::find_or_initialize(init);
  }

 private:
  std::mutex mutex;
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
