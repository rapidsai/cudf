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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace {
struct size_of_helper {
  cudf::data_type type;
  template <typename T, typename std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  constexpr int operator()() const
  {
    CUDF_FAIL("Invalid, non fixed-width element type.");
  }

  template <typename T,
            typename std::enable_if_t<is_fixed_width<T>() && not is_fixed_point<T>()>* = nullptr>
  constexpr int operator()() const noexcept
  {
    return sizeof(T);
  }

  template <typename T, typename std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  constexpr int operator()() const noexcept
  {
    // Only want the sizeof fixed_point::Rep as fixed_point::scale is stored in data_type
    return sizeof(typename T::rep);
  }
};
}  // namespace

std::size_t size_of(data_type element_type)
{
  CUDF_EXPECTS(is_fixed_width(element_type), "Invalid element type.");
  return cudf::type_dispatcher(element_type, size_of_helper{element_type});
}

// Empty column of specified type
std::unique_ptr<column> make_empty_column(data_type type)
{
  CUDF_EXPECTS(type.id() == type_id::EMPTY || !cudf::is_nested(type),
               "make_empty_column is invalid to call on nested types");
  return std::make_unique<column>(type, 0, rmm::device_buffer{});
}

// Allocate storage for a specified number of numeric elements
std::unique_ptr<column> make_numeric_column(data_type type,
                                            size_type size,
                                            mask_state state,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of numeric elements
std::unique_ptr<column> make_fixed_point_column(data_type type,
                                                size_type size,
                                                mask_state state,
                                                cudaStream_t stream,
                                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_fixed_point(type), "Invalid, non-fixed_point type.");

  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of timestamp elements
std::unique_ptr<column> make_timestamp_column(data_type type,
                                              size_type size,
                                              mask_state state,
                                              cudaStream_t stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");

  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of duration elements
std::unique_ptr<column> make_duration_column(data_type type,
                                             size_type size,
                                             mask_state state,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");

  return std::make_unique<column>(type,
                                  size,
                                  rmm::device_buffer{size * cudf::size_of(type), stream, mr},
                                  create_null_mask(size, state, stream, mr),
                                  state_null_count(state, size),
                                  std::vector<std::unique_ptr<column>>{});
}

// Allocate storage for a specified number of fixed width elements
std::unique_ptr<column> make_fixed_width_column(data_type type,
                                                size_type size,
                                                mask_state state,
                                                cudaStream_t stream,
                                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");

  // clang-format off
  if      (is_timestamp  (type)) return make_timestamp_column  (type, size, state, stream, mr);
  else if (is_duration   (type)) return make_duration_column   (type, size, state, stream, mr);
  else if (is_fixed_point(type)) return make_fixed_point_column(type, size, state, stream, mr);
  else                           return make_numeric_column    (type, size, state, stream, mr);
  /// clang-format on
}

struct column_from_scalar_dispatch {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(scalar const& value,
                                           size_type size,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream) const
  {
    if (!value.is_valid())
      return make_fixed_width_column(value.type(), size, mask_state::ALL_NULL, stream, mr);
    auto output_column =
      make_fixed_width_column(value.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto view = output_column->mutable_view();
    detail::fill_in_place(view, 0, size, value, stream);
    return output_column;
  }
};

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::string_view>(
  scalar const& value,
  size_type size,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  if (!value.is_valid())
    return std::make_unique<column>(value.type(),
                                    size,
                                    rmm::device_buffer{0, stream, mr},
                                    create_null_mask(size, mask_state::ALL_NULL, stream, mr),
                                    size);

  // Create a strings column_view with all nulls and no children.
  // Since we are setting every row to the scalar, the fill() never needs to access
  // any of the children in the strings column which would otherwise cause an exception.
  auto null_mask = create_null_mask(size, mask_state::ALL_NULL, stream);
  column_view sc{
    data_type{type_id::STRING}, size, nullptr, static_cast<bitmask_type*>(null_mask.data()), size};
  auto sv = static_cast<scalar_type_t<cudf::string_view> const&>(value);
  // fill the column with the scalar
  auto output = strings::detail::fill(strings_column_view(sc), 0, size, sv, mr, stream);
  output->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);  // should be no nulls
  return output;
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::dictionary32>(
  scalar const& value,
  size_type size,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  CUDF_FAIL("dictionary not supported when creating from scalar");
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::list_view>(
  scalar const& value,
  size_type size,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  CUDF_FAIL("TODO");
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::struct_view>(
  scalar const& value,
  size_type size,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream) const
{
  CUDF_FAIL("TODO. struct_view currently not supported.");
}

std::unique_ptr<column> make_column_from_scalar(scalar const& s,
                                                size_type size,
                                                rmm::mr::device_memory_resource* mr,
                                                cudaStream_t stream)
{
  if (size == 0) return make_empty_column(s.type());
  return type_dispatcher(s.type(), column_from_scalar_dispatch{}, s, size, mr, stream);
}

std::unique_ptr<column> make_dictionary_from_scalar(scalar const& s,
                                                    size_type size,
                                                    rmm::mr::device_memory_resource* mr,
                                                    cudaStream_t stream)
{
  if (size == 0) return make_empty_column(data_type{type_id::DICTIONARY32});
  CUDF_EXPECTS(s.is_valid(), "cannot create a dictionary with a null key");
  return make_dictionary_column(
    make_column_from_scalar(s, 1, mr, stream),
    make_column_from_scalar(numeric_scalar<uint32_t>(0), size, mr, stream),
    rmm::device_buffer{0, stream, mr},
    0);
}

}  // namespace cudf
