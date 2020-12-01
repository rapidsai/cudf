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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts hex strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct hex_to_integer_fn {
  column_device_view const strings_column;

  /**
   * @brief Converts a single hex string into an integer.
   *
   * Non-hexadecimal characters are ignored.
   * This means it can handle "0x01A23" and "1a23".
   *
   * Overflow of the int64 type is not detected.
   */
  __device__ int64_t string_to_integer(string_view const& d_str)
  {
    int64_t result = 0, base = 1;
    const char* str = d_str.data();
    size_type index = d_str.size_bytes();
    while (index-- > 0) {
      char ch = str[index];
      if (ch >= '0' && ch <= '9') {
        result += static_cast<int64_t>(ch - 48) * base;
        base *= 16;
      } else if (ch >= 'A' && ch <= 'F') {
        result += static_cast<int64_t>(ch - 55) * base;
        base *= 16;
      } else if (ch >= 'a' && ch <= 'f') {
        result += static_cast<int64_t>(ch - 87) * base;
        base *= 16;
      }
    }
    return result;
  }

  __device__ IntegerType operator()(size_type idx)
  {
    if (strings_column.is_null(idx)) return static_cast<IntegerType>(0);
    // the cast to IntegerType will create predictable results
    // for integers that are larger than the IntegerType can hold
    return static_cast<IntegerType>(string_to_integer(strings_column.element<string_view>(idx)));
  }
};

/**
 * @brief The dispatch functions for converting strings to integers.
 *
 * The output_column is expected to be one of the integer types only.
 */
struct dispatch_hex_to_integers_fn {
  template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  rmm::cuda_stream_view stream) const
  {
    auto d_results = output_column.data<IntegerType>();
    thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_column.size()),
                      d_results,
                      hex_to_integer_fn<IntegerType>{strings_column});
  }
  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  void operator()(column_device_view const&, mutable_column_view&, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Output for hex_to_integers must be an integral type.");
  }
};

template <>
void dispatch_hex_to_integers_fn::operator()<bool>(column_device_view const&,
                                                   mutable_column_view&,
                                                   rmm::cuda_stream_view) const
{
  CUDF_FAIL("Output for hex_to_integers must not be a boolean type.");
}

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> hex_to_integers(
  strings_column_view const& strings,
  data_type output_type,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(output_type);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create integer output column copying the strings null-mask
  auto results      = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  // fill output column with integers
  type_dispatcher(output_type, dispatch_hex_to_integers_fn{}, d_strings, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

std::unique_ptr<column> is_hex(strings_column_view const& strings,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings.size()),
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto const d_str = d_column.element<string_view>(idx);
                      if (d_str.empty()) return false;
                      auto const starts_with_0x = [](auto const& sv) {
                        return sv.length() > 1 && (sv.substr(0, 2) == string_view("0x", 2) ||
                                                   sv.substr(0, 2) == string_view("0X", 2));
                      };
                      auto begin = d_str.begin() + (starts_with_0x(d_str) ? 2 : 0);
                      auto end   = d_str.end();
                      return (thrust::distance(begin, end) > 0) &&
                             thrust::all_of(thrust::seq, begin, end, [] __device__(auto chr) {
                               return (chr >= '0' && chr <= '9') || (chr >= 'A' && chr <= 'F') ||
                                      (chr >= 'a' && chr <= 'f');
                             });
                    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> hex_to_integers(strings_column_view const& strings,
                                        data_type output_type,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::hex_to_integers(strings, output_type, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> is_hex(strings_column_view const& strings,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_hex(strings, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
