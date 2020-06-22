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

#include <cstring>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <unordered_map>
#include "./utilities.cuh"
#include "./utilities.hpp"
#include "char_types/char_cases.h"
#include "char_types/char_flags.h"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <mutex>

namespace cudf {
namespace strings {
namespace detail {
// Used to build a temporary string_view object from a single host string.
std::unique_ptr<string_view, std::function<void(string_view*)>> string_from_host(
  const char* str, cudaStream_t stream)
{
  if (!str) return nullptr;
  auto length = std::strlen(str);

  auto* d_str = new rmm::device_buffer(length, stream);
  CUDA_TRY(cudaMemcpyAsync(d_str->data(), str, length, cudaMemcpyHostToDevice, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  auto deleter = [d_str](string_view* sv) { delete d_str; };
  return std::unique_ptr<string_view, decltype(deleter)>{
    new string_view(reinterpret_cast<char*>(d_str->data()), length), deleter};
}

// build a vector of string_view objects from a strings column
rmm::device_vector<string_view> create_string_vector_from_column(cudf::strings_column_view strings,
                                                                 cudaStream_t stream)
{
  auto execpol        = rmm::exec_policy(stream);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  auto count = strings.size();
  rmm::device_vector<string_view> strings_vector(count);
  string_view* d_strings = strings_vector.data().get();
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     count,
                     [d_column, d_strings] __device__(size_type idx) {
                       if (d_column.is_null(idx))
                         d_strings[idx] = string_view(nullptr, 0);
                       else
                         d_strings[idx] = d_column.element<string_view>(idx);
                     });
  return strings_vector;
}

// build a strings offsets column from a vector of string_views
std::unique_ptr<cudf::column> child_offsets_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto transformer = [] __device__(string_view v) { return v.size_bytes(); };
  auto begin       = thrust::make_transform_iterator(strings.begin(), transformer);
  return make_offsets_child_column(begin, begin + strings.size(), mr, stream);
}

// build a strings chars column from an vector of string_views
std::unique_ptr<cudf::column> child_chars_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  const int32_t* d_offsets,
  cudf::size_type null_count,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  size_type count = strings.size();
  auto d_strings  = strings.data().get();
  auto execpol    = rmm::exec_policy(stream);
  size_type bytes = thrust::device_pointer_cast(d_offsets)[count];

  // create column
  auto chars_column =
    make_numeric_column(data_type{type_id::INT8}, bytes, mask_state::UNALLOCATED, stream, mr);
  // get it's view
  auto d_chars = chars_column->mutable_view().data<int8_t>();
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     count,
                     [d_strings, d_offsets, d_chars] __device__(size_type idx) {
                       string_view const d_str = d_strings[idx];
                       memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
                     });

  return chars_column;
}

//
std::unique_ptr<column> create_chars_child_column(cudf::size_type strings_count,
                                                  cudf::size_type null_count,
                                                  cudf::size_type total_bytes,
                                                  rmm::mr::device_memory_resource* mr,
                                                  cudaStream_t stream)
{
  CUDF_EXPECTS(null_count <= strings_count, "Invalid null count");
  return make_numeric_column(
    data_type{type_id::INT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
}

//
std::unique_ptr<column> make_empty_strings_column(rmm::mr::device_memory_resource* mr,
                                                  cudaStream_t stream)
{
  return std::make_unique<column>(data_type{type_id::STRING},
                                  0,
                                  rmm::device_buffer{0, stream, mr},  // data
                                  rmm::device_buffer{0, stream, mr},
                                  0);  // nulls
}

namespace {
// The device variables are created here to avoid using a singleton that may cause issues
// with RMM initialize/finalize. See PR #3159 for details on this approach.
__device__ character_flags_table_type
  character_codepoint_flags[sizeof(g_character_codepoint_flags)];
__device__ character_cases_table_type character_cases_table[sizeof(g_character_cases_table)];
__device__ special_case_mapping character_special_case_mappings[sizeof(g_special_case_mappings)];

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

thread_safe_per_context_cache<character_flags_table_type> d_character_codepoint_flags;
thread_safe_per_context_cache<character_cases_table_type> d_character_cases_table;
thread_safe_per_context_cache<special_case_mapping> d_special_case_mappings;

}  // namespace

/**
 * @copydoc cudf::strings::detail::get_character_flags_table
 */
const character_flags_table_type* get_character_flags_table()
{
  return d_character_codepoint_flags.find_or_initialize([&](void) {
    character_flags_table_type* table = nullptr;
    CUDA_TRY(cudaMemcpyToSymbol(
      character_codepoint_flags, g_character_codepoint_flags, sizeof(g_character_codepoint_flags)));
    CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_codepoint_flags));
    return table;
  });
}

/**
 * @copydoc cudf::strings::detail::get_character_cases_table
 */
const character_cases_table_type* get_character_cases_table()
{
  return d_character_cases_table.find_or_initialize([&](void) {
    character_cases_table_type* table = nullptr;
    CUDA_TRY(cudaMemcpyToSymbol(
      character_cases_table, g_character_cases_table, sizeof(g_character_cases_table)));
    CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_cases_table));
    return table;
  });
}

/**
 * @copydoc cudf::strings::detail::get_special_case_mapping_table
 */
const special_case_mapping* get_special_case_mapping_table()
{
  return d_special_case_mappings.find_or_initialize([&](void) {
    special_case_mapping* table = nullptr;
    CUDA_TRY(cudaMemcpyToSymbol(
      character_special_case_mappings, g_special_case_mappings, sizeof(g_special_case_mappings)));
    CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_special_case_mappings));
    return table;
  });
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
