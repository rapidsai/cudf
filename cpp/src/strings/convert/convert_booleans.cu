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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
// Convert strings column to boolean column
std::unique_ptr<column> to_booleans(strings_column_view const& strings,
                                    string_scalar const& true_string,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(data_type{type_id::BOOL8}, 0);

  CUDF_EXPECTS(true_string.is_valid(stream) && true_string.size() > 0,
               "Parameter true_string must not be empty.");
  auto d_true = string_view(true_string.data(), true_string.size());

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column copying the strings' null-mask
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    [d_strings, d_true] __device__(size_type idx) {
                      bool result = false;
                      if (!d_strings.is_null(idx))
                        result = d_strings.element<string_view>(idx).compare(d_true) == 0;
                      return result;
                    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> to_booleans(strings_column_view const& strings,
                                    string_scalar const& true_string,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_booleans(strings, true_string, cudf::get_default_stream(), mr);
}

namespace detail {
// Convert boolean column to strings column
std::unique_ptr<column> from_booleans(column_view const& booleans,
                                      string_scalar const& true_string,
                                      string_scalar const& false_string,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = booleans.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(booleans.type().id() == type_id::BOOL8, "Input column must be boolean type");
  CUDF_EXPECTS(true_string.is_valid(stream) && true_string.size() > 0,
               "Parameter true_string must not be empty.");
  auto d_true = string_view(true_string.data(), true_string.size());
  CUDF_EXPECTS(false_string.is_valid(stream) && false_string.size() > 0,
               "Parameter false_string must not be empty.");
  auto d_false = string_view(false_string.data(), false_string.size());

  auto column   = column_device_view::create(booleans, stream);
  auto d_column = *column;

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(booleans, stream, mr);
  // build offsets column
  auto offsets_transformer_itr = cudf::detail::make_counting_transform_iterator(
    0, [d_column, d_true, d_false] __device__(size_type idx) {
      if (d_column.is_null(idx)) return 0;
      return d_column.element<bool>(idx) ? d_true.size_bytes() : d_false.size_bytes();
    });
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto offsets_view = offsets_column->view();
  auto d_offsets    = offsets_view.data<int32_t>();

  // build chars column
  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [d_column, d_true, d_false, d_offsets, d_chars] __device__(size_type idx) {
                       if (d_column.is_null(idx)) return;
                       string_view result = (d_column.element<bool>(idx) ? d_true : d_false);
                       memcpy(d_chars + d_offsets[idx], result.data(), result.size_bytes());
                     });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             booleans.null_count(),
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> from_booleans(column_view const& booleans,
                                      string_scalar const& true_string,
                                      string_scalar const& false_string,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_booleans(booleans, true_string, false_string, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
