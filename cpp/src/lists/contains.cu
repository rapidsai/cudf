/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/detail/contains.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cudf {
namespace lists {

namespace {

auto constexpr absent_index = size_type{-1};

auto get_search_keys_device_iterable_view(cudf::column_view const& search_keys,
                                          rmm::cuda_stream_view stream)
{
  return column_device_view::create(search_keys, stream);
}

auto get_search_keys_device_iterable_view(cudf::scalar const& search_key, rmm::cuda_stream_view)
{
  return &search_key;
}

template <typename ElementType, duplicate_find_option find_option>
auto __device__ find_begin(list_device_view const& list)
{
  if constexpr (find_option == duplicate_find_option::FIND_FIRST) {
    return list.pair_rep_begin<ElementType>();
  } else {
    return thrust::make_reverse_iterator(list.pair_rep_end<ElementType>());
  }
}

template <typename ElementType, duplicate_find_option find_option>
auto __device__ find_end(list_device_view const& list)
{
  if constexpr (find_option == duplicate_find_option::FIND_FIRST) {
    return list.pair_rep_end<ElementType>();
  } else {
    return thrust::make_reverse_iterator(list.pair_rep_begin<ElementType>());
  }
}

template <duplicate_find_option find_option, typename Iterator>
size_type __device__ distance([[maybe_unused]] Iterator begin, Iterator end, Iterator find_iter)
{
  if (find_iter == end) {
    return absent_index;  // Not found.
  }

  if constexpr (find_option == duplicate_find_option::FIND_FIRST) {
    return find_iter - begin;  // Distance of find_position from begin.
  } else {
    return end - find_iter - 1;  // Distance of find_position from end.
  }
}

/**
 * @brief __device__ functor to search for a key in a `list_device_view`.
 */
template <duplicate_find_option find_option>
struct finder {
  template <typename ElementType>
  __device__ size_type operator()(list_device_view const& list, ElementType const& search_key) const
  {
    auto const list_begin = find_begin<ElementType, find_option>(list);
    auto const list_end   = find_end<ElementType, find_option>(list);
    auto const find_iter  = thrust::find_if(
      thrust::seq, list_begin, list_end, [search_key] __device__(auto element_and_validity) {
        auto [element, element_is_valid] = element_and_validity;
        return element_is_valid && cudf::equality_compare(element, search_key);
      });
    return distance<find_option>(list_begin, list_end, find_iter);
  };
};

/**
 * @brief Functor to search each list row for the specified search keys.
 */
template <bool search_keys_have_nulls>
struct lookup_functor {
  template <typename ElementType>
  struct is_supported {
    static constexpr bool value =
      cudf::is_numeric<ElementType>() || cudf::is_chrono<ElementType>() ||
      cudf::is_fixed_point<ElementType>() || std::is_same_v<ElementType, cudf::string_view>;
  };

  template <typename ElementType, typename... Args>
  std::enable_if_t<!is_supported<ElementType>::value, std::unique_ptr<column>> operator()(
    Args&&...) const
  {
    CUDF_FAIL(
      "List search operations are only supported on numeric types, decimals, chrono types, and "
      "strings.");
  }

  std::pair<rmm::device_buffer, size_type> construct_null_mask(
    lists_column_view const& input_lists,
    column_view const& result_validity,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    if (!search_keys_have_nulls && !input_lists.has_nulls() && !input_lists.child().has_nulls()) {
      return {rmm::device_buffer{0, stream, mr}, size_type{0}};
    } else {
      return cudf::detail::valid_if(
        result_validity.begin<bool>(), result_validity.end<bool>(), thrust::identity{}, stream, mr);
    }
  }

  template <typename ElementType, typename SearchKeyPairIter>
  void search_each_list_row(cudf::detail::lists_column_device_view const& d_lists,
                            SearchKeyPairIter search_key_pair_iter,
                            duplicate_find_option find_option,
                            cudf::mutable_column_device_view ret_positions,
                            cudf::mutable_column_device_view ret_validity,
                            rmm::cuda_stream_view stream) const
  {
    auto output_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(ret_positions.data<size_type>(), ret_validity.data<bool>()));

    thrust::tabulate(
      rmm::exec_policy(stream),
      output_iterator,
      output_iterator + d_lists.size(),
      [d_lists, search_key_pair_iter, absent_index = absent_index, find_option] __device__(
        auto row_index) -> thrust::pair<size_type, bool> {
        auto [search_key, search_key_is_valid] = search_key_pair_iter[row_index];

        if (search_keys_have_nulls && !search_key_is_valid) { return {absent_index, false}; }

        auto list = cudf::list_device_view(d_lists, row_index);
        if (list.is_null()) { return {absent_index, false}; }

        auto const position = find_option == duplicate_find_option::FIND_FIRST
                                ? finder<duplicate_find_option::FIND_FIRST>{}(list, search_key)
                                : finder<duplicate_find_option::FIND_LAST>{}(list, search_key);
        return {position, true};
      });
  }

  template <typename ElementType, typename SearchKeyType>
  std::enable_if_t<is_supported<ElementType>::value, std::unique_ptr<column>> operator()(
    cudf::lists_column_view const& lists,
    SearchKeyType const& search_key,
    duplicate_find_option find_option,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const
  {
    using namespace cudf;
    using namespace cudf::detail;

    CUDF_EXPECTS(!cudf::is_nested(lists.child().type()),
                 "Nested types not supported in list search operations.");
    CUDF_EXPECTS(lists.child().type() == search_key.type(),
                 "Type/Scale of search key does not match list column element type.");
    CUDF_EXPECTS(search_key.type().id() != type_id::EMPTY, "Type cannot be empty.");

    auto constexpr search_key_is_scalar = std::is_same_v<SearchKeyType, cudf::scalar>;

    if constexpr (search_keys_have_nulls && search_key_is_scalar) {
      return make_numeric_column(data_type(type_id::INT32),
                                 lists.size(),
                                 cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                 lists.size(),
                                 stream,
                                 mr);
    }

    auto const device_view = column_device_view::create(lists.parent(), stream);
    auto const d_lists     = lists_column_device_view{*device_view};
    auto const d_skeys     = get_search_keys_device_iterable_view(search_key, stream);

    auto result_positions = make_numeric_column(
      data_type{type_id::INT32}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto result_validity = make_numeric_column(
      data_type{type_id::BOOL8}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto mutable_result_positions =
      mutable_column_device_view::create(result_positions->mutable_view(), stream);
    auto mutable_result_validity =
      mutable_column_device_view::create(result_validity->mutable_view(), stream);
    auto search_key_iter =
      cudf::detail::make_pair_rep_iterator<ElementType, search_keys_have_nulls>(*d_skeys);

    search_each_list_row<ElementType>(d_lists,
                                      search_key_iter,
                                      find_option,
                                      *mutable_result_positions,
                                      *mutable_result_validity,
                                      stream);

    auto [null_mask, num_nulls] = construct_null_mask(lists, result_validity->view(), stream, mr);
    result_positions->set_null_mask(std::move(null_mask), num_nulls);
    return result_positions;
  }
};

/**
 * @brief Converts key-positions vector (from index_of()) to a BOOL8 vector, indicating if
 * the search key was found.
 */
std::unique_ptr<column> to_contains(std::unique_ptr<column>&& key_positions,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(key_positions->type().id() == type_id::INT32,
               "Expected input column of type INT32.");
  // If position == -1, the list did not contain the search key.
  auto const num_rows        = key_positions->size();
  auto const positions_begin = key_positions->view().begin<size_type>();
  auto result =
    make_numeric_column(data_type{type_id::BOOL8}, num_rows, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    positions_begin,
                    positions_begin + num_rows,
                    result->mutable_view().begin<bool>(),
                    [] __device__(auto i) { return i != absent_index; });
  [[maybe_unused]] auto [_, null_mask, __] = key_positions->release();
  result->set_null_mask(std::move(*null_mask));
  return result;
}
}  // namespace

namespace detail {
/**
 * @copydoc cudf::lists::detail::index_of(cudf::lists_column_view const&,
 *                                        cudf::scalar const&,
 *                                        duplicate_find_option,
 *                                        rmm::cuda_stream_view,
 *                                        rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return search_key.is_valid(stream)
           ? cudf::type_dispatcher(search_key.type(),
                                   lookup_functor<false>{},  // No nulls in search key
                                   lists,
                                   search_key,
                                   find_option,
                                   stream,
                                   mr)
           : cudf::type_dispatcher(search_key.type(),
                                   lookup_functor<true>{},  // Nulls in search key
                                   lists,
                                   search_key,
                                   find_option,
                                   stream,
                                   mr);
}

/**
 * @copydoc cudf::lists::detail::index_of(cudf::lists_column_view const&,
 *                                        cudf::column_view const&,
 *                                        duplicate_find_option,
 *                                        rmm::cuda_stream_view,
 *                                        rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");

  return search_keys.has_nulls()
           ? cudf::type_dispatcher(search_keys.type(),
                                   lookup_functor<true>{},  // Nulls in search keys
                                   lists,
                                   search_keys,
                                   find_option,
                                   stream,
                                   mr)
           : cudf::type_dispatcher(search_keys.type(),
                                   lookup_functor<false>{},  // No nulls in search keys
                                   lists,
                                   search_keys,
                                   find_option,
                                   stream,
                                   mr);
}

/**
 * @copydoc cudf::lists::detail::contains(cudf::lists_column_view const&,
 *                                        cudf::scalar const&,
 *                                        rmm::cuda_stream_view,
 *                                        rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return to_contains(
    index_of(lists, search_key, duplicate_find_option::FIND_FIRST, stream), stream, mr);
}

/**
 * @copydoc cudf::lists::detail::contains(cudf::lists_column_view const&,
 *                                        cudf::column_view const&,
 *                                        rmm::cuda_stream_view,
 *                                        rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");

  return to_contains(
    index_of(lists, search_keys, duplicate_find_option::FIND_FIRST, stream), stream, mr);
}

/**
 * @copydoc cudf::lists::contain_nulls(cudf::lists_column_view const&,
 *                                     rmm::mr::device_memory_resource*)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains_nulls(cudf::lists_column_view const& input_lists,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const num_rows   = input_lists.size();
  auto const d_lists    = column_device_view::create(input_lists.parent());
  auto has_nulls_output = make_numeric_column(
    data_type{type_id::BOOL8}, input_lists.size(), mask_state::UNALLOCATED, stream, mr);
  auto const output_begin = has_nulls_output->mutable_view().begin<bool>();
  thrust::tabulate(
    rmm::exec_policy(stream),
    output_begin,
    output_begin + num_rows,
    [lists = cudf::detail::lists_column_device_view{*d_lists}] __device__(auto list_idx) {
      auto list       = list_device_view{lists, list_idx};
      auto list_begin = thrust::make_counting_iterator(size_type{0});
      return list.is_null() ||
             thrust::any_of(thrust::seq, list_begin, list_begin + list.size(), [&list](auto i) {
               return list.is_null(i);
             });
    });
  auto const validity_begin = cudf::detail::make_counting_transform_iterator(
    0, [lists = cudf::detail::lists_column_device_view{*d_lists}] __device__(auto list_idx) {
      return not list_device_view{lists, list_idx}.is_null();
    });
  auto [null_mask, num_nulls] = cudf::detail::valid_if(
    validity_begin, validity_begin + num_rows, thrust::identity<bool>{}, stream, mr);
  has_nulls_output->set_null_mask(std::move(null_mask), num_nulls);
  return has_nulls_output;
}

}  // namespace detail

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_key, cudf::default_stream_value, mr);
}

std::unique_ptr<column> contains(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_keys, cudf::default_stream_value, mr);
}

std::unique_ptr<column> contains_nulls(cudf::lists_column_view const& input_lists,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_nulls(input_lists, cudf::default_stream_value, mr);
}

std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_key, find_option, cudf::default_stream_value, mr);
}

std::unique_ptr<column> index_of(cudf::lists_column_view const& lists,
                                 cudf::column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_keys, find_option, cudf::default_stream_value, mr);
}

}  // namespace lists
}  // namespace cudf
