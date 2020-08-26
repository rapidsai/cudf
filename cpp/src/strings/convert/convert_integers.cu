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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <strings/convert/utilities.cuh>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct string_to_integer_fn {
  const column_device_view strings_column;  // strings to convert

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
struct dispatch_to_integers_fn {
  template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  cudaStream_t stream) const
  {
    auto d_results = output_column.data<IntegerType>();
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_column.size()),
                      d_results,
                      string_to_integer_fn<IntegerType>{strings_column});
  }
  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  void operator()(column_device_view const&, mutable_column_view&, cudaStream_t) const
  {
    CUDF_FAIL("Output for to_integers must be an integral type.");
  }
};

template <>
void dispatch_to_integers_fn::operator()<bool>(column_device_view const&,
                                               mutable_column_view&,
                                               cudaStream_t) const
{
  CUDF_FAIL("Output for to_integers must not be a boolean type.");
}

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(output_type, 0);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create integer output column copying the strings null-mask
  auto results      = make_numeric_column(output_type,
                                     strings_count,
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  // fill output column with integers
  type_dispatcher(output_type, dispatch_to_integers_fn{}, d_strings, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_integers(strings, output_type, cudaStream_t{}, mr);
}

namespace detail {
namespace {
/**
 * @brief Calculate the size of the each string required for
 * converting each integer in base-10 format.
 */
template <typename IntegerType>
struct integer_to_string_size_fn {
  column_device_view d_column;

  __device__ size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return 0;
    IntegerType value = d_column.element<IntegerType>(idx);
    return count_digits(value);
  }
};

/**
 * @brief Convert each integer into a string.
 *
 * The integer is converted into base-10 using only characters [0-9].
 * No formatting is done for the string other than prepending the '-'
 * character for negative values.
 */
template <typename IntegerType>
struct integer_to_string_fn {
  column_device_view d_column;
  const int32_t* d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return;
    IntegerType value = d_column.element<IntegerType>(idx);
    char* d_buffer    = d_chars + d_offsets[idx];
    integer_to_string(value, d_buffer);
  }
};

/**
 * @brief This dispatch method is for converting integers into strings.
 * The template function declaration ensures only integer types are used.
 */
struct dispatch_from_integers_fn {
  template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& integers,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    size_type strings_count = integers.size();
    auto column             = column_device_view::create(integers, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = copy_bitmask(integers, stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0), integer_to_string_size_fn<IntegerType>{d_column});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
    auto chars_column =
      detail::create_chars_child_column(strings_count, integers.null_count(), bytes, mr, stream);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       integer_to_string_fn<IntegerType>{d_column, d_new_offsets, d_chars});
    //
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               integers.null_count(),
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::mr::device_memory_resource*,
                                     cudaStream_t) const
  {
    CUDF_FAIL("Values for from_integers function must be an integral type.");
  }
};

template <>
std::unique_ptr<column> dispatch_from_integers_fn::operator()<bool>(
  column_view const&, rmm::mr::device_memory_resource*, cudaStream_t) const
{
  CUDF_FAIL("Input for from_integers must not be a boolean type.");
}

}  // namespace

// This will convert all integer column types into a strings column.
std::unique_ptr<column> from_integers(column_view const& integers,
                                      cudaStream_t stream,
                                      rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = integers.size();
  if (strings_count == 0) return detail::make_empty_strings_column(mr, stream);

  return type_dispatcher(integers.type(), dispatch_from_integers_fn{}, integers, mr, stream);
}

}  // namespace detail

// external API

std::unique_ptr<column> from_integers(column_view const& integers,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_integers(integers, cudaStream_t{}, mr);
}

}  // namespace strings
}  // namespace cudf
