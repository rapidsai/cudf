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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts IPv4 strings into integers.
 *
 * Only single-byte characters are expected.
 * No checking is done on the format of individual strings.
 * Any character that is not [0-9] is considered a delimiter.
 * This means "128-34-56-709" will parse successfully.
 */
struct ipv4_to_integers_fn {
  column_device_view const d_strings;

  __device__ int64_t operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    string_view d_str  = d_strings.element<string_view>(idx);
    uint32_t ipvals[4] = {0};  // IPV4 format: xxx.xxx.xxx.xxx
    int32_t ipv_idx    = 0;
    int32_t factor     = 1;
    const char* in_ptr = d_str.data();
    const char* end    = in_ptr + d_str.size_bytes();
    while ((in_ptr < end) && (ipv_idx < 4)) {
      char ch = *in_ptr++;
      if (ch < '0' || ch > '9') {
        ++ipv_idx;
        factor = 1;
      } else {
        ipvals[ipv_idx] = (ipvals[ipv_idx] * factor) + static_cast<uint32_t>(ch - '0');
        factor          = 10;
      }
    }
    uint32_t result = (ipvals[0] << 24) + (ipvals[1] << 16) + (ipvals[2] << 8) + ipvals[3];
    return static_cast<int64_t>(result);
  }
};

}  // namespace

// Convert strings column of IPv4 addresses to integers column
std::unique_ptr<column> ipv4_to_integers(strings_column_view const& strings,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(data_type{type_id::INT64}, 0);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  // create output column copying the strings' null-mask
  auto results   = make_numeric_column(data_type{type_id::INT64},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<int64_t>();
  // fill output column with ipv4 integers
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    ipv4_to_integers_fn{*strings_column});
  // done
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> ipv4_to_integers(strings_column_view const& strings,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ipv4_to_integers(strings, cudf::get_default_stream(), mr);
}

namespace detail {
namespace {
/**
 * @brief Converts integers into IPv4 addresses.
 *
 * Each integer is divided into 8-bit sub-integers.
 * The sub-integers are converted into 1-3 character digits.
 * These are placed appropriately between '.' character.
 */
struct integers_to_ipv4_fn {
  column_device_view const d_column;
  int32_t const* d_offsets;
  char* d_chars;

  __device__ int convert(int value, char* digits)
  {
    int digits_idx = 0;
    while ((value > 0) && (digits_idx < 3)) {
      digits[digits_idx++] = '0' + (value % 10);
      value                = value / 10;
    }
    return digits_idx;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return;
    int64_t ip_number = d_column.element<int64_t>(idx);
    char* out_ptr     = d_chars + d_offsets[idx];
    int shift_bits    = 24;
    for (int n = 0; n < 4; ++n) {
      int value = static_cast<int>((ip_number >> shift_bits) & 0x00FF);
      if (value == 0)
        *out_ptr++ = '0';
      else {
        char digits[3];
        int num_digits = convert(value, digits);
        while (num_digits-- > 0)
          *out_ptr++ = digits[num_digits];
      }
      if ((n + 1) < 4) *out_ptr++ = '.';
      shift_bits -= 8;
    }
  }
};

}  // namespace

// Convert integers into IPv4 addresses
std::unique_ptr<column> integers_to_ipv4(column_view const& integers,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = integers.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(integers.type().id() == type_id::INT64, "Input column must be type_id::INT64 type");

  auto column   = column_device_view::create(integers, stream);
  auto d_column = *column;

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(integers, stream, mr);
  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0), [d_column] __device__(size_type idx) {
      if (d_column.is_null(idx)) return 0;
      size_type bytes   = 3;  // at least 3 dots: xxx.xxx.xxx.xxx
      int64_t ip_number = d_column.element<int64_t>(idx);
      for (int n = 0; n < 4; ++n) {
        auto value = ip_number & 0x00FF;
        bytes += (value < 10 ? 1 : (value < 100 ? 2 : 3));
        ip_number = ip_number >> 8;
      }
      return bytes;
    });
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     integers_to_ipv4_fn{d_column, d_offsets, d_chars});

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             integers.null_count(),
                             std::move(null_mask));
}

std::unique_ptr<column> is_ipv4(strings_column_view const& strings,
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
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings.size()),
                    d_results,
                    [d_column] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      auto const d_str = d_column.element<string_view>(idx);
                      if (d_str.empty()) return false;
                      constexpr int max_ip = 255;  // values must be in [0,255]
                      int ip_vals[4]       = {-1, -1, -1, -1};
                      int ipv_idx          = 0;  // index into ip_vals
                      for (auto const ch : d_str) {
                        if ((ch >= '0') && (ch <= '9')) {
                          auto const ip_val    = ip_vals[ipv_idx];
                          int const new_ip_val = static_cast<int>(ch - '0') +  // compute new value
                                                 (ip_val < 0 ? 0 : (10 * ip_val));
                          if (new_ip_val > max_ip) return false;
                          ip_vals[ipv_idx] = new_ip_val;
                        }
                        // here ipv_idx is incremented only when ch=='.'
                        else if (ch != '.' || (++ipv_idx > 3))
                          return false;
                      }
                      // final check for any missing values
                      return ip_vals[0] >= 0 && ip_vals[1] >= 0 && ip_vals[2] >= 0 &&
                             ip_vals[3] >= 0;
                    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> integers_to_ipv4(column_view const& integers,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::integers_to_ipv4(integers, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> is_ipv4(strings_column_view const& strings,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_ipv4(strings, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
