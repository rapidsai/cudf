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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>

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
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
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
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream.value()), begin, end, d_offsets + 1);
  CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream.value()));
  return offsets_column;
}

/**
 * @brief Creates an offsets column from a string_view iterator, and size.
 *
 * @tparam Iter Iterator type that returns string_view instances
 * @param strings_begin Iterator to the beginning of the string_view sequence
 * @param num_strings The number of string_view instances in the sequence
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Child offsets column
 */
template <typename Iter>
std::unique_ptr<cudf::column> child_offsets_from_string_iterator(
  Iter strings_begin,
  cudf::size_type num_strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto transformer = [] __device__(string_view v) { return v.size_bytes(); };
  auto begin       = thrust::make_transform_iterator(strings_begin, transformer);
  return make_offsets_child_column(begin, begin + num_strings, stream, mr);
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
