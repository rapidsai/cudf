/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>

#include <rmm/resource_ref.hpp>

#include <thrust/generate.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {

namespace {

struct column_from_scalar_dispatch {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(scalar const& value,
                                           size_type size,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if (size == 0) return make_empty_column(value.type());
    if (!value.is_valid(stream))
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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (size == 0) return make_empty_column(value.type());

  if (!value.is_valid(stream)) {
    return make_strings_column(
      size,
      make_column_from_scalar(numeric_scalar<int32_t>(0), size + 1, stream, mr),
      rmm::device_buffer{},
      size,
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr));
  }

  auto& ss       = static_cast<scalar_type_t<cudf::string_view> const&>(value);
  auto const sdv = cudf::get_scalar_device_view(const_cast<string_scalar&>(ss));

  // fill the column with the scalar
  using sip_t = cudf::strings::detail::string_index_pair;
  rmm::device_uvector<sip_t> indices(size, stream);
  thrust::generate(
    rmm::exec_policy_nosync(stream), indices.begin(), indices.end(), [sdv] __device__() {
      auto const d_str = sdv.value();
      // special handling for empty string -- from an all-empty strings column
      return d_str.empty() ? sip_t{"", 0} : sip_t{d_str.data(), d_str.size_bytes()};
    });
  return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::dictionary32>(
  scalar const&, size_type, rmm::cuda_stream_view, rmm::device_async_resource_ref) const
{
  CUDF_FAIL("dictionary not supported when creating from scalar");
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::list_view>(
  scalar const& value,
  size_type size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto lv = static_cast<list_scalar const*>(&value);
  return lists::detail::make_lists_column_from_scalar(*lv, size, stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::struct_view>(
  scalar const& value,
  size_type size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (size == 0) CUDF_FAIL("0-length struct column is unsupported.");
  auto& ss  = static_cast<scalar_type_t<cudf::struct_view> const&>(value);
  auto iter = thrust::make_constant_iterator(0);

  auto children =
    detail::gather(ss.view(), iter, iter + size, out_of_bounds_policy::NULLIFY, stream, mr);
  auto const is_valid = ss.is_valid(stream);
  return make_structs_column(size,
                             std::move(children->release()),
                             is_valid ? 0 : size,
                             is_valid
                               ? rmm::device_buffer{}
                               : detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
                             stream,
                             mr);
}

}  // anonymous namespace

std::unique_ptr<column> make_column_from_scalar(scalar const& s,
                                                size_type size,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  return type_dispatcher(s.type(), column_from_scalar_dispatch{}, s, size, stream, mr);
}

}  // namespace cudf
