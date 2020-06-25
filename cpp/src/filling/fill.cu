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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime.h>

#include <memory>

namespace {
template <typename T>
void in_place_fill(cudf::mutable_column_view& destination,
                   cudf::size_type begin,
                   cudf::size_type end,
                   cudf::scalar const& value,
                   cudaStream_t stream = 0)
{
  using ScalarType = cudf::scalar_type_t<T>;
  auto p_scalar    = static_cast<ScalarType const*>(&value);
  T fill_value     = p_scalar->value(stream);
  bool is_valid    = p_scalar->is_valid();
  cudf::detail::copy_range(thrust::make_constant_iterator(fill_value),
                           thrust::make_constant_iterator(is_valid),
                           destination,
                           begin,
                           end,
                           stream);
}

struct in_place_fill_range_dispatch {
  cudf::scalar const& value;
  cudf::mutable_column_view& destination;

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), void> operator()(cudf::size_type begin,
                                                               cudf::size_type end,
                                                               cudaStream_t stream = 0)
  {
    in_place_fill<T>(destination, begin, end, value, stream);
  }

  template <typename T>
  std::enable_if_t<not cudf::is_fixed_width<T>(), void> operator()(cudf::size_type begin,
                                                                   cudf::size_type end,
                                                                   cudaStream_t stream = 0)
  {
    CUDF_FAIL("in-place fill does not work for variable width types.");
  }
};

struct out_of_place_fill_range_dispatch {
  cudf::scalar const& value;
  cudf::column_view const& input;

  template <typename T>
  std::unique_ptr<cudf::column> operator()(
    cudf::size_type begin,
    cudf::size_type end,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream                 = 0)
  {
    CUDF_EXPECTS(input.type() == value.type(), "Data type mismatch.");
    auto p_ret = std::make_unique<cudf::column>(input, stream, mr);

    if (end != begin) {  // otherwise no fill
      if (!p_ret->nullable() && !value.is_valid()) {
        p_ret->set_null_mask(
          cudf::create_null_mask(p_ret->size(), cudf::mask_state::ALL_VALID, stream, mr), 0);
      }

      auto ret_view = p_ret->mutable_view();
      in_place_fill<T>(ret_view, begin, end, value, stream);
    }

    return p_ret;
  }
};

template <>
std::unique_ptr<cudf::column> out_of_place_fill_range_dispatch::operator()<cudf::list_view>(
  cudf::size_type begin,
  cudf::size_type end,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("list_view not supported yet");
}

template <>
std::unique_ptr<cudf::column> out_of_place_fill_range_dispatch::operator()<cudf::string_view>(
  cudf::size_type begin,
  cudf::size_type end,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_EXPECTS(input.type() == value.type(), "Data type mismatch.");
  using ScalarType = cudf::scalar_type_t<cudf::string_view>;
  auto p_scalar    = static_cast<ScalarType const*>(&value);
  return cudf::strings::detail::fill(
    cudf::strings_column_view(input), begin, end, *p_scalar, mr, stream);
}

template <>
std::unique_ptr<cudf::column> out_of_place_fill_range_dispatch::operator()<cudf::dictionary32>(
  cudf::size_type begin,
  cudf::size_type end,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  if (input.size() == 0) return std::make_unique<cudf::column>(input, stream, mr);
  cudf::dictionary_column_view const target(input);
  CUDF_EXPECTS(target.keys().type() == value.type(), "Data type mismatch.");

  // if the scalar is invalid, then just copy the column and fill the null mask
  if (!value.is_valid()) {
    auto result = std::make_unique<cudf::column>(input, stream, mr);
    auto mview  = result->mutable_view();
    cudf::set_null_mask(mview.null_mask(), begin, end, false, stream);
    mview.set_null_count(input.null_count() + (end - begin));
    return result;
  }

  // add the scalar to get the output dictionary key-set
  auto scalar_column =
    cudf::make_column_from_scalar(value, 1, rmm::mr::get_default_resource(), stream);
  auto target_matched =
    cudf::dictionary::detail::add_keys(target, scalar_column->view(), mr, stream);
  cudf::column_view const target_indices =
    cudf::dictionary_column_view(target_matched->view()).get_indices_annotated();

  // get the index of the key just added
  auto index_of_value = cudf::dictionary::detail::get_index(
    target_matched->view(), value, rmm::mr::get_default_resource(), stream);
  // now call fill using just the indices column and the new index
  out_of_place_fill_range_dispatch filler{*index_of_value, target_indices};
  auto new_indices       = filler.template operator()<int32_t>(begin, end, mr, stream);
  auto const output_size = new_indices->size();        // record these
  auto const null_count  = new_indices->null_count();  // before the release()
  auto contents          = new_indices->release();
  // create the new indices column from the result
  auto indices_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       static_cast<cudf::size_type>(output_size),
                                                       std::move(*(contents.data.release())),
                                                       rmm::device_buffer{0, stream, mr},
                                                       0);

  // take the keys from matched column
  std::unique_ptr<cudf::column> keys_column(std::move(target_matched->release().children.back()));

  // create column with keys_column and indices_column
  return cudf::make_dictionary_column(std::move(keys_column),
                                      std::move(indices_column),
                                      std::move(*(contents.null_mask.release())),
                                      null_count);
}

}  // namespace

namespace cudf {
namespace detail {
void fill_in_place(mutable_column_view& destination,
                   size_type begin,
                   size_type end,
                   scalar const& value,
                   cudaStream_t stream)
{
  CUDF_EXPECTS(cudf::is_fixed_width(destination.type()) == true,
               "In-place fill does not support variable-sized types.");
  CUDF_EXPECTS((begin >= 0) && (end <= destination.size()) && (begin <= end),
               "Range is out of bounds.");
  CUDF_EXPECTS((destination.nullable() == true) || (value.is_valid() == true),
               "destination should be nullable or value should be non-null.");
  CUDF_EXPECTS(destination.type() == value.type(), "Data type mismatch.");

  if (end != begin) {  // otherwise no-op
    cudf::type_dispatcher(
      destination.type(), in_place_fill_range_dispatch{value, destination}, begin, end, stream);
  }

  return;
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream)
{
  CUDF_EXPECTS((begin >= 0) && (end <= input.size()) && (begin <= end), "Range is out of bounds.");

  return cudf::type_dispatcher(
    input.type(), out_of_place_fill_range_dispatch{value, input}, begin, end, mr, stream);
}

}  // namespace detail

void fill_in_place(mutable_column_view& destination,
                   size_type begin,
                   size_type end,
                   scalar const& value)
{
  CUDF_FUNC_RANGE();
  return detail::fill_in_place(destination, begin, end, value, 0);
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::fill(input, begin, end, value, mr, 0);
}

}  // namespace cudf
