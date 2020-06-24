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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.cuh>
#include <strings/utilities.hpp>

#include <thrust/logical.h>

//
namespace cudf {
namespace strings {
namespace detail {
//
std::unique_ptr<column> all_characters_of_type(
  strings_column_view const& strings,
  string_character_types types,
  string_character_types verify_types,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_count  = strings.size();
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // get the static character types table
  auto d_flags = detail::get_character_flags_table();
  // set the output values by checking the character types for each string
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    [d_column, d_flags, types, verify_types, d_results] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto d_str            = d_column.element<string_view>(idx);
                      bool check            = !d_str.empty();  // require at least one character
                      size_type check_count = 0;
                      for (auto itr = d_str.begin(); check && (itr != d_str.end()); ++itr) {
                        auto code_point = detail::utf8_to_codepoint(*itr);
                        // lookup flags in table by code-point
                        auto flag = code_point <= 0x00FFFF ? d_flags[code_point] : 0;
                        if ((verify_types & flag) ||                   // should flag be verified
                            (flag == 0 && verify_types == ALL_TYPES))  // special edge case
                        {
                          check = (types & flag) > 0;
                          ++check_count;
                        }
                      }
                      return check && (check_count > 0);
                    });
  //
  results->set_null_count(strings.null_count());
  return results;
}

std::unique_ptr<column> is_integer(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings.size()),
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      return string::is_integer(d_column.element<string_view>(idx));
                    });
  results->set_null_count(strings.null_count());
  return results;
}

bool all_integer(strings_column_view const& strings, cudaStream_t stream = 0)
{
  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  auto transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), [d_column] __device__(size_type idx) {
      if (d_column.is_null(idx)) return false;
      return string::is_integer(d_column.element<string_view>(idx));
    });
  return thrust::all_of(rmm::exec_policy(stream)->on(stream),
                        transformer_itr,
                        transformer_itr + strings.size(),
                        thrust::identity<bool>());
}

std::unique_ptr<column> is_float(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  // check strings for valid float chars
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings.size()),
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      return string::is_float(d_column.element<string_view>(idx));
                    });
  results->set_null_count(strings.null_count());
  return results;
}

bool all_float(strings_column_view const& strings, cudaStream_t stream = 0)
{
  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  auto transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), [d_column] __device__(size_type idx) {
      if (d_column.is_null(idx)) return false;
      return string::is_float(d_column.element<string_view>(idx));
    });
  return thrust::all_of(rmm::exec_policy(stream)->on(stream),
                        transformer_itr,
                        transformer_itr + strings.size(),
                        thrust::identity<bool>());
}

}  // namespace detail

// external API

std::unique_ptr<column> all_characters_of_type(strings_column_view const& strings,
                                               string_character_types types,
                                               string_character_types verify_types,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::all_characters_of_type(strings, types, verify_types, mr);
}

std::unique_ptr<column> is_integer(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_integer(strings, mr);
}

std::unique_ptr<column> is_float(strings_column_view const& strings,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_float(strings, mr);
}

bool all_integer(strings_column_view const& strings)
{
  CUDF_FUNC_RANGE();
  return detail::all_integer(strings);
}

bool all_float(strings_column_view const& strings)
{
  CUDF_FUNC_RANGE();
  return detail::all_float(strings);
}

}  // namespace strings
}  // namespace cudf
