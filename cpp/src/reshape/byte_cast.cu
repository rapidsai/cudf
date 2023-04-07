/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reshape.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

namespace cudf {
namespace detail {
namespace {

template <typename T, typename Enable = void>
struct byte_list_conversion_fn {
  template <typename... Args>
  static std::unique_ptr<column> invoke(Args&&...)
  {
    CUDF_FAIL("Unsupported non-numeric and non-string column");
  }
};

struct byte_list_conversion_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     flip_endianness configuration,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    return byte_list_conversion_fn<T>::invoke(input, configuration, stream, mr);
  }
};

template <typename T>
static constexpr bool is_supported()
{
  return cudf::is_numeric<T>() || std::is_same_v<T, cudf::string_view>;
}

template <typename T>
struct byte_list_conversion_fn<
  T,
  std::enable_if_t<is_supported<T>() && !std::is_same_v<T, cudf::string_view>>> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        flip_endianness configuration,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    size_type num_bytes = input.size() * sizeof(T);
    auto byte_column    = make_numeric_column(
      data_type{type_id::UINT8}, num_bytes, mask_state::UNALLOCATED, stream, mr);

    char* d_chars      = reinterpret_cast<char*>(byte_column->mutable_view().data<uint8_t>());
    char const* d_data = reinterpret_cast<char const*>(input.data<T>());
    size_type mask     = sizeof(T) - 1;

    if (configuration == flip_endianness::YES) {
      thrust::for_each(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(num_bytes),
                       [d_chars, d_data, mask] __device__(auto index) {
                         d_chars[index] = d_data[index + mask - ((index & mask) << 1)];
                       });
    } else {
      thrust::copy_n(rmm::exec_policy(stream), d_data, num_bytes, d_chars);
    }

    auto begin = thrust::make_constant_iterator(cudf::size_of(input.type()));
    auto offsets_column =
      std::get<0>(cudf::detail::make_offsets_child_column(begin, begin + input.size(), stream, mr));

    auto result = make_lists_column(input.size(),
                                    std::move(offsets_column),
                                    std::move(byte_column),
                                    input.null_count(),
                                    detail::copy_bitmask(input, stream, mr),
                                    stream,
                                    mr);

    // If any nulls are present, the corresponding lists must be purged so that
    // the result is sanitized.
    if (auto const result_cv = result->view();
        cudf::detail::has_nonempty_nulls(result_cv, stream)) {
      return cudf::detail::purge_nonempty_nulls(result_cv, stream, mr);
    }

    return result;
  }
};

template <typename T>
struct byte_list_conversion_fn<T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>> {
  static std::unique_ptr<column> invoke(column_view const& input,
                                        flip_endianness configuration,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    strings_column_view input_strings(input);
    auto strings_count = input_strings.size();
    if (strings_count == 0) return cudf::empty_like(input);

    auto col_content = std::make_unique<column>(input, stream, mr)->release();
    auto chars_contents =
      col_content.children[strings_column_view::chars_column_index].release()->release();
    auto chars_buffer = chars_contents.data.release();
    auto num_chars    = chars_buffer->size();
    auto uint8_col    = std::make_unique<column>(
      data_type{type_id::UINT8}, num_chars, std::move(*chars_buffer), rmm::device_buffer{}, 0);

    auto result = make_lists_column(
      input.size(),
      std::move(col_content.children[cudf::strings_column_view::offsets_column_index]),
      std::move(uint8_col),
      input.null_count(),
      detail::copy_bitmask(input, stream, mr),
      stream,
      mr);

    // If any nulls are present, the corresponding lists must be purged so that
    // the result is sanitized.
    if (auto const result_cv = result->view();
        cudf::detail::has_nonempty_nulls(result_cv, stream)) {
      return cudf::detail::purge_nonempty_nulls(result_cv, stream, mr);
    }

    return result;
  }
};

}  // namespace

/**
 * @copydoc cudf::byte_cast(column_view const&, flip_endianness, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> byte_cast(column_view const& input,
                                  flip_endianness endian_configuration,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    input.type(), byte_list_conversion_dispatcher{}, input, endian_configuration, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::byte_cast(column_view const&, flip_endianness, rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> byte_cast(column_view const& input,
                                  flip_endianness endian_configuration,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_cast(input, endian_configuration, cudf::get_default_stream(), mr);
}

}  // namespace cudf
