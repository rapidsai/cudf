/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "strings/char_types/char_cases.h"
#include "strings/char_types/char_flags.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstdlib>
#include <string>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc create_string_vector_from_column
 */
rmm::device_uvector<string_view> create_string_vector_from_column(
  cudf::strings_column_view const input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto strings_vector = rmm::device_uvector<string_view>(input.size(), stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(input.size()),
                    strings_vector.begin(),
                    [d_strings = *d_strings] __device__(size_type idx) {
                      // placeholder for factory function that takes a span of string_views
                      auto const null_string_view = string_view{nullptr, 0};
                      if (d_strings.is_null(idx)) { return null_string_view; }
                      auto const d_str = d_strings.element<string_view>(idx);
                      // special case when the entire column is filled with empty strings:
                      // here the empty d_str may have a d_str.data() == nullptr
                      auto const empty_string_view = string_view{};
                      return d_str.empty() ? empty_string_view : d_str;
                    });

  return strings_vector;
}

/**
 * @copydoc cudf::strings::detail::create_offsets_child_column
 */
std::unique_ptr<column> create_offsets_child_column(int64_t chars_bytes,
                                                    size_type count,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  auto const threshold = get_offset64_threshold();
  if (!is_large_strings_enabled()) {
    CUDF_EXPECTS(
      chars_bytes < threshold, "Size of output exceeds the column size limit", std::overflow_error);
  }
  return make_numeric_column(
    chars_bytes < threshold ? data_type{type_id::INT32} : data_type{type_id::INT64},
    count,
    mask_state::UNALLOCATED,
    stream,
    mr);
}

namespace {
// The device variables are created here to avoid using a singleton that may cause issues
// with RMM initialize/finalize. See PR #3159 for details on this approach.
__device__ character_flags_table_type
  character_codepoint_flags[sizeof(g_character_codepoint_flags)];
__device__ character_cases_table_type character_cases_table[sizeof(g_character_cases_table)];
__device__ special_case_mapping character_special_case_mappings[sizeof(g_special_case_mappings)];

thread_safe_per_context_cache<character_flags_table_type> d_character_codepoint_flags;
thread_safe_per_context_cache<character_cases_table_type> d_character_cases_table;
thread_safe_per_context_cache<special_case_mapping> d_special_case_mappings;

}  // namespace

/**
 * @copydoc cudf::strings::detail::get_character_flags_table
 */
character_flags_table_type const* get_character_flags_table()
{
  return d_character_codepoint_flags.find_or_initialize([&](void) {
    character_flags_table_type* table = nullptr;
    CUDF_CUDA_TRY(cudaMemcpyToSymbol(
      character_codepoint_flags, g_character_codepoint_flags, sizeof(g_character_codepoint_flags)));
    CUDF_CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_codepoint_flags));
    return table;
  });
}

/**
 * @copydoc cudf::strings::detail::get_character_cases_table
 */
character_cases_table_type const* get_character_cases_table()
{
  return d_character_cases_table.find_or_initialize([&](void) {
    character_cases_table_type* table = nullptr;
    CUDF_CUDA_TRY(cudaMemcpyToSymbol(
      character_cases_table, g_character_cases_table, sizeof(g_character_cases_table)));
    CUDF_CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_cases_table));
    return table;
  });
}

/**
 * @copydoc cudf::strings::detail::get_special_case_mapping_table
 */
special_case_mapping const* get_special_case_mapping_table()
{
  return d_special_case_mappings.find_or_initialize([&](void) {
    special_case_mapping* table = nullptr;
    CUDF_CUDA_TRY(cudaMemcpyToSymbol(
      character_special_case_mappings, g_special_case_mappings, sizeof(g_special_case_mappings)));
    CUDF_CUDA_TRY(cudaGetSymbolAddress((void**)&table, character_special_case_mappings));
    return table;
  });
}

int64_t get_offset64_threshold()
{
  auto const threshold = std::getenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
  int64_t const rtn    = threshold != nullptr ? std::atol(threshold) : 0L;
  return (rtn > 0 && rtn < std::numeric_limits<int32_t>::max())
           ? rtn
           : std::numeric_limits<int32_t>::max();
}

bool is_large_strings_enabled()
{
  auto const env = std::getenv("LIBCUDF_LARGE_STRINGS_ENABLED");
  return env != nullptr && std::string(env) == "1";
}

int64_t get_offset_value(cudf::column_view const& offsets,
                         size_type index,
                         rmm::cuda_stream_view stream)
{
  auto const otid = offsets.type().id();
  CUDF_EXPECTS(otid == type_id::INT64 || otid == type_id::INT32,
               "Offsets must be of type INT32 or INT64",
               std::invalid_argument);
  return otid == type_id::INT64 ? cudf::detail::get_value<int64_t>(offsets, index, stream)
                                : cudf::detail::get_value<int32_t>(offsets, index, stream);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
