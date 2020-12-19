/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/logical.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>

namespace cudf {
namespace lists {

namespace {

auto CUDA_HOST_DEVICE_CALLABLE counting_iter(size_type n)
{
  return thrust::make_counting_iterator(n);
}

std::pair<rmm::device_buffer, size_type> construct_null_mask(
  cudf::detail::lists_column_device_view const& d_lists,
  cudf::scalar const& skey,
  bool input_column_has_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using namespace cudf;
  using namespace cudf::detail;

  if (!skey.is_valid(stream)) {
    return std::make_pair(cudf::create_null_mask(d_lists.size(), mask_state::ALL_NULL, mr),
                          d_lists.size());
  }

  return cudf::detail::valid_if(
    counting_iter(0), counting_iter(d_lists.size()), [d_lists] __device__(auto const& row_index) {
      auto list = cudf::list_device_view(d_lists, row_index);
      if (list.is_null()) { return false; }
      return thrust::none_of(thrust::seq,
                             counting_iter(0),
                             counting_iter(list.size()),
                             [&list] __device__(auto const& i) { return list.is_null(i); });
    });
}

struct lookup_functor {
  template <typename T, typename... Args>
  std::enable_if_t<!cudf::is_numeric<T>() && !std::is_same<T, cudf::string_view>::value, void>
  operator()(Args&&...) const
  {
    CUDF_FAIL("lists::contains() is only supported on numeric types and strings.");
  }

  template <typename T>
  std::enable_if_t<cudf::is_numeric<T>() || std::is_same<T, cudf::string_view>::value, void>
  operator()(cudf::detail::lists_column_device_view const& d_lists,
             cudf::scalar const& skey,
             cudf::mutable_column_device_view output_bools,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    assert(skey.is_valid() && "skey should have been checked for nulls by this point.");

    auto h_scalar = static_cast<cudf::scalar_type_t<T> const&>(skey);
    auto d_scalar = cudf::get_scalar_device_view(h_scalar);

    thrust::transform(rmm::exec_policy(stream),
                      counting_iter(0),
                      counting_iter(d_lists.size()),
                      output_bools.data<bool>(),
                      [d_lists, d_scalar] __device__(auto row_index) {
                        auto list = cudf::list_device_view(d_lists, row_index);
                        if (list.is_null()) { return false; }
                        for (size_type i{0}; i < list.size(); ++i) {
                          if (list.is_null(i)) { return false; }
                          auto list_element = list.template element<T>(i);
                          if (list_element == d_scalar.value()) { return true; }
                        }
                        return false;
                      });
  }
};

}  // namespace

namespace detail {

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& skey,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  using namespace cudf;
  using namespace cudf::detail;

  CUDF_EXPECTS(!cudf::is_nested(lists.child().type()),
               "Nested types not supported in lists::contains()");
  CUDF_EXPECTS(lists.child().type().id() == skey.type().id(),
               "Type of search key does not match list column element type.");
  CUDF_EXPECTS(skey.type().id() != type_id::EMPTY, "Type cannot be empty.");

  auto const device_view = column_device_view::create(lists.parent(), stream);
  auto const d_lists     = lists_column_device_view(*device_view);

  rmm::device_buffer null_mask;
  size_type num_nulls;

  std::tie(null_mask, num_nulls) =
    construct_null_mask(d_lists, skey, lists.has_nulls(), stream, mr);

  auto ret_bools = make_fixed_width_column(
    data_type{type_id::BOOL8}, lists.size(), std::move(null_mask), num_nulls, stream, mr);

  if (skey.is_valid()) {
    auto ret_bools_mutable_device_view =
      mutable_column_device_view::create(ret_bools->mutable_view(), stream);

    cudf::type_dispatcher(
      skey.type(), lookup_functor{}, d_lists, skey, *ret_bools_mutable_device_view, stream, mr);
  }

  return ret_bools;
}

}  // namespace detail

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& skey,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::contains(lists, skey, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
