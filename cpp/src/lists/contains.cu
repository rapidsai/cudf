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
#include <cudf/detail/iterator.cuh>
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

/**
 * @brief Construct null-mask for result of `contains()` called with scalar search key.
 */
std::pair<rmm::device_buffer, size_type> construct_null_mask(
  cudf::detail::lists_column_device_view const& d_lists,
  cudf::scalar const& skey,
  bool input_column_has_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using namespace cudf;
  using namespace cudf::detail;

  if (skey.is_valid(stream) && !input_column_has_nulls) {
    return std::make_pair(rmm::device_buffer{0, stream, mr}, size_type{0});
  }

  if (!skey.is_valid(stream)) {
    return std::make_pair(cudf::create_null_mask(d_lists.size(), mask_state::ALL_NULL, mr),
                          d_lists.size());
  }

  return cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(d_lists.size()),
                                [d_lists] __device__(auto const& row_index) {
                                  auto list = cudf::list_device_view(d_lists, row_index);
                                  if (list.is_null()) { return false; }
                                  return thrust::none_of(
                                    thrust::seq,
                                    thrust::make_counting_iterator(0),
                                    thrust::make_counting_iterator(list.size()),
                                    [&list] __device__(auto const& i) { return list.is_null(i); });
                                });
}

/**
 * @brief Construct null-mask for result of `contains()` called with column of search keys.
 */
std::pair<rmm::device_buffer, size_type> construct_null_mask(
  cudf::detail::lists_column_device_view const& d_lists,
  cudf::column_device_view const& d_skeys,
  bool input_column_has_nulls,
  bool skeys_column_has_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using namespace cudf;
  using namespace cudf::detail;

  if (!skeys_column_has_nulls && !input_column_has_nulls) {
    return std::make_pair(rmm::device_buffer{0, stream, mr}, size_type{0});
  }

  return cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(d_lists.size()),
                                [d_lists, d_skeys] __device__(auto const& row_index) {
                                  if (d_skeys.is_null(row_index)) { return false; }

                                  auto list = cudf::list_device_view(d_lists, row_index);

                                  if (list.is_null()) { return false; }

                                  return thrust::none_of(
                                    thrust::seq,
                                    thrust::make_counting_iterator(0),
                                    thrust::make_counting_iterator(list.size()),
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

  template <typename T, typename SearchKeyPairIter>
  std::enable_if_t<cudf::is_numeric<T>() || std::is_same<T, cudf::string_view>::value, void>
  op_impl(cudf::detail::lists_column_device_view const& d_lists,
          SearchKeyPairIter search_key_iter,
          cudf::mutable_column_device_view output_bools,
          rmm::cuda_stream_view stream,
          rmm::mr::device_memory_resource* mr) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(d_lists.size()),
                      output_bools.data<bool>(),
                      [d_lists, search_key_iter] __device__(auto row_index) {
                        auto search_key_and_validity    = search_key_iter[row_index];
                        auto const& search_key_is_valid = search_key_and_validity.second;

                        if (!search_key_is_valid) { return false; }

                        auto list = cudf::list_device_view(d_lists, row_index);
                        if (list.is_null()) { return false; }
                        auto search_key = search_key_and_validity.first;

                        return thrust::find_if(thrust::seq,
                                               list.pair_begin<T>(),
                                               list.pair_end<T>(),
                                               [search_key] __device__(auto element_and_validity) {
                                                 return element_and_validity.second &&
                                                        (element_and_validity.first == search_key);
                                               }) != list.pair_end<T>();
                      });
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
    return op_impl<T>(d_lists, detail::make_pair_iterator<T>(skey), output_bools, stream, mr);
  }

  template <typename T>
  std::enable_if_t<cudf::is_numeric<T>() || std::is_same<T, cudf::string_view>::value, void>
  operator()(cudf::detail::lists_column_device_view const& d_lists,
             cudf::column_device_view const& d_search_keys,
             bool search_keys_have_nulls,
             cudf::mutable_column_device_view output_bools,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const
  {
    return search_keys_have_nulls
             ? op_impl<T>(d_lists, d_search_keys.pair_begin<T, true>(), output_bools, stream, mr)
             : op_impl<T>(d_lists, d_search_keys.pair_begin<T, false>(), output_bools, stream, mr);
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
    construct_null_mask(d_lists, skey, lists.has_nulls() || lists.child().has_nulls(), stream, mr);

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

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& skeys,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  using namespace cudf;
  using namespace cudf::detail;

  CUDF_EXPECTS(!cudf::is_nested(lists.child().type()),
               "Nested types not supported in lists::contains()");
  CUDF_EXPECTS(lists.child().type().id() == skeys.type().id(),
               "Type of search key does not match list column element type.");
  CUDF_EXPECTS(skeys.size() == lists.size(), "Number of search keys must match list column size.");
  CUDF_EXPECTS(skeys.type().id() != type_id::EMPTY, "Type cannot be empty.");

  auto const device_view = column_device_view::create(lists.parent(), stream);
  auto const d_lists     = lists_column_device_view(*device_view);
  auto const d_skeys     = column_device_view::create(skeys, stream);

  rmm::device_buffer null_mask;
  size_type num_nulls;

  std::tie(null_mask, num_nulls) =
    construct_null_mask(d_lists,
                        *d_skeys,
                        lists.has_nulls() || lists.child().has_nulls(),
                        skeys.has_nulls(),
                        stream,
                        mr);

  auto ret_bools = make_fixed_width_column(
    data_type{type_id::BOOL8}, lists.size(), std::move(null_mask), num_nulls, stream, mr);

  auto ret_bools_mutable_device_view =
    mutable_column_device_view::create(ret_bools->mutable_view(), stream);

  cudf::type_dispatcher(skeys.type(),
                        lookup_functor{},
                        d_lists,
                        *d_skeys,
                        skeys.has_nulls(),
                        *ret_bools_mutable_device_view,
                        stream,
                        mr);

  return ret_bools;
}

}  // namespace detail

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& skey,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, skey, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& skeys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, skeys, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
