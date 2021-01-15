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

#include <thrust/iterator/constant_iterator.h>
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

auto get_search_keys_device_iterable_view(cudf::column_view const& search_keys,
                                          rmm::cuda_stream_view stream)
{
  return column_device_view::create(search_keys, stream);
}

auto get_search_keys_device_iterable_view(cudf::scalar const& search_key,
                                          rmm::cuda_stream_view stream)
{
  return &search_key;
}

template <typename ElementType, bool has_nulls>
auto get_pair_iterator(cudf::column_device_view const& d_search_keys)
{
  return d_search_keys.pair_begin<ElementType, has_nulls>();
}

template <typename ElementType, bool ignore>
auto get_pair_iterator(cudf::scalar const& search_key)
{
  return cudf::detail::make_pair_iterator<ElementType>(search_key);
}

/**
 * @brief Functor to search each list row for the specified search keys.
 */
template <bool search_keys_have_nulls>
struct lookup_functor {
  template <typename ElementType, typename... Args>
  std::enable_if_t<!cudf::is_numeric<ElementType>() && !cudf::is_chrono<ElementType>() &&
                     !std::is_same<ElementType, cudf::string_view>::value,
                   std::unique_ptr<column>>
  operator()(Args&&...) const
  {
    CUDF_FAIL("lists::contains() is only supported on numeric types, chrono types, and strings.");
  }

  std::pair<rmm::device_buffer, size_type> construct_null_mask(lists_column_view const& input_lists,
                                                               column_view const& result_validity,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::mr::device_memory_resource* mr)
  {
    if (!search_keys_have_nulls && !input_lists.has_nulls() && !input_lists.child().has_nulls()) {
      return {rmm::device_buffer{0, stream, mr}, size_type{0}};
    } else {
      return cudf::detail::valid_if(result_validity.begin<bool>(),
                                    result_validity.end<bool>(),
                                    thrust::identity<bool>{},
                                    stream,
                                    mr);
    }
  }

  template <typename ElementType, typename SearchKeyPairIter>
  void search_each_list_row(cudf::detail::lists_column_device_view const& d_lists,
                            SearchKeyPairIter search_key_pair_iter,
                            cudf::mutable_column_device_view mutable_ret_bools,
                            cudf::mutable_column_device_view mutable_ret_validity,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
  {
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(d_lists.size()),
      [d_lists,
       search_key_pair_iter,
       d_bools    = mutable_ret_bools.data<bool>(),
       d_validity = mutable_ret_validity.data<bool>()] __device__(auto row_index) {
        auto search_key_and_validity    = search_key_pair_iter[row_index];
        auto const& search_key_is_valid = search_key_and_validity.second;

        if (search_keys_have_nulls && !search_key_is_valid) {
          d_bools[row_index]    = false;
          d_validity[row_index] = false;
          return;
        }

        auto list = cudf::list_device_view(d_lists, row_index);
        if (list.is_null()) {
          d_bools[row_index]    = false;
          d_validity[row_index] = false;
          return;
        }

        auto search_key    = search_key_and_validity.first;
        d_bools[row_index] = thrust::find_if(thrust::seq,
                                             list.pair_begin<ElementType>(),
                                             list.pair_end<ElementType>(),
                                             [search_key] __device__(auto element_and_validity) {
                                               return element_and_validity.second &&
                                                      (element_and_validity.first == search_key);
                                             }) != list.pair_end<ElementType>();
        d_validity[row_index] =
          d_bools[row_index] ||
          thrust::none_of(thrust::seq,
                          thrust::make_counting_iterator(size_type{0}),
                          thrust::make_counting_iterator(list.size()),
                          [&list] __device__(auto const& i) { return list.is_null(i); });
      });
  }

  template <typename ElementType, typename SearchKeyType>
  std::enable_if_t<cudf::is_numeric<ElementType>() || cudf::is_chrono<ElementType>() ||
                     std::is_same<ElementType, cudf::string_view>::value,
                   std::unique_ptr<column>>
  operator()(cudf::lists_column_view const& lists,
             SearchKeyType const& search_key,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    using namespace cudf;
    using namespace cudf::detail;

    CUDF_EXPECTS(!cudf::is_nested(lists.child().type()),
                 "Nested types not supported in lists::contains()");
    CUDF_EXPECTS(lists.child().type().id() == search_key.type().id(),
                 "Type of search key does not match list column element type.");
    CUDF_EXPECTS(search_key.type().id() != type_id::EMPTY, "Type cannot be empty.");

    auto constexpr search_key_is_scalar = std::is_same<SearchKeyType, cudf::scalar>::value;

    if (search_keys_have_nulls && search_key_is_scalar) {
      return make_fixed_width_column(data_type(type_id::BOOL8),
                                     lists.size(),
                                     cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                     lists.size(),
                                     stream,
                                     mr);
    }

    auto const device_view = column_device_view::create(lists.parent(), stream);
    auto const d_lists     = lists_column_device_view(*device_view);
    auto const d_skeys     = get_search_keys_device_iterable_view(search_key, stream);

    auto const lists_column_has_nulls = lists.has_nulls() || lists.child().has_nulls();

    auto result_validity = make_fixed_width_column(
      data_type{type_id::BOOL8}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto result_bools = make_fixed_width_column(
      data_type{type_id::BOOL8}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto mutable_result_bools =
      mutable_column_device_view::create(result_bools->mutable_view(), stream);
    auto mutable_result_validity =
      mutable_column_device_view::create(result_validity->mutable_view(), stream);
    auto search_key_iter = get_pair_iterator<ElementType, search_keys_have_nulls>(*d_skeys);

    search_each_list_row<ElementType>(
      d_lists, search_key_iter, *mutable_result_bools, *mutable_result_validity, stream, mr);

    rmm::device_buffer null_mask;
    size_type num_nulls;

    std::tie(null_mask, num_nulls) =
      construct_null_mask(lists, result_validity->view(), stream, mr);
    result_bools->set_null_mask(std::move(null_mask), num_nulls);

    return result_bools;
  }
};

}  // namespace

namespace detail {

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return search_key.is_valid(stream)
           ? cudf::type_dispatcher(
               search_key.type(), lookup_functor<false>{}, lists, search_key, stream, mr)
           : cudf::type_dispatcher(
               search_key.type(), lookup_functor<true>{}, lists, search_key, stream, mr);
}

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");

  return search_keys.has_nulls()
           ? cudf::type_dispatcher(
               search_keys.type(), lookup_functor<true>{}, lists, search_keys, stream, mr)
           : cudf::type_dispatcher(
               search_keys.type(), lookup_functor<false>{}, lists, search_keys, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_key, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_keys, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
