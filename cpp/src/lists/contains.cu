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
#include <cudf/lists/detail/contains.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cudf::lists {

namespace {

/**
 * @brief A sentinel value used for marking that a given key has not been found in the search list.
 */
auto constexpr __device__ NOT_FOUND_SENTINEL = size_type{-1};

template <typename Type>
auto constexpr is_supported_non_nested_type()
{
  return cudf::is_fixed_width<Type>() || std::is_same_v<Type, cudf::string_view>;
}

/**
 * @brief Return a pair of index iterators {begin, end} to loop through elements within a list.
 *
 * Depending on the value of `find_first`, a pair of forward or reverse iterators will be returned,
 * allowing to loop through elements in the list in first-to-last or last-to-first order.
 *
 * Note that the element indices always restart to `0` at the first position in each list.
 *
 * @tparam find_first A boolean value indicating whether we want to find the first or last
 *         appearance of a given key in the list.
 * @param size The number of elements in the list.
 * @return A pair of {begin, end} iterators to iterate through the range `[0, size)` by forward or
 *         reverse order.
 */
template <bool find_first>
auto __device__ element_index_pair_iter(size_type const size)
{
  if constexpr (find_first) {
    return thrust::pair(thrust::make_counting_iterator(0), thrust::make_counting_iterator(size));
  } else {
    return thrust::pair(thrust::make_reverse_iterator(thrust::make_counting_iterator(size)),
                        thrust::make_reverse_iterator(thrust::make_counting_iterator(0)));
  }
}

template <typename Type, typename SearchKeyIter>
struct search_lists_fn {
  cudf::detail::lists_column_device_view const lists;
  SearchKeyIter const keys_iter;
  duplicate_find_option const find_option;

  __device__ thrust::pair<size_type, bool> operator()(size_type const idx) const
  {
    auto const list = list_device_view{lists, idx};
    // A null list never contains any key, even null key.
    // In addition, a null list will result in a null output row.
    if (list.is_null()) { return {NOT_FOUND_SENTINEL, false}; }

    auto const key_opt = keys_iter[idx];
    // A null key will also result in a null output row.
    if (!key_opt) { return {NOT_FOUND_SENTINEL, false}; }

    auto const& key = key_opt.value();
    return {find_option == duplicate_find_option::FIND_FIRST ? search_list<true>(list, key)
                                                             : search_list<false>(list, key),
            true};
  }

 private:
  template <bool find_first>
  static __device__ size_type search_list(list_device_view const& list, Type const& search_key)
  {
    auto const [begin, end] = element_index_pair_iter<find_first>(list.size());
    auto const found_iter =
      thrust::find_if(thrust::seq, begin, end, [&list, search_key] __device__(auto const idx) {
        return !list.is_null(idx) &&
               cudf::equality_compare(list.template element<Type>(idx), search_key);
      });
    // If the key is found, return its found position in the list from `found_iter`.
    return found_iter == end ? NOT_FOUND_SENTINEL : *found_iter;
  }
};

/**
 * @brief Create a device pointer to the search key(s).
 *
 * The returned pointer will be used to construct a validity iterator or an optional iterator for
 * the keys, which also have the same interface for both `scalar` and `column_device_view`.
 *
 * @return Depending on the type of the input key(s), a `scalar` pointer or a `column_device_view`
 *         pointer will be returned.
 */
template <typename SearchKeyType>
auto get_search_keys_device_view_ptr(SearchKeyType const& search_keys,
                                     [[maybe_unused]] rmm::cuda_stream_view stream)
{
  if constexpr (std::is_same_v<SearchKeyType, cudf::scalar>) {
    return &search_keys;
  } else {
    return column_device_view::create(search_keys, stream);
  }
}

/**
 * @brief Dispatch functor to search for key element(s) in the corresponding rows of a lists column.
 */
struct dispatch_index_of {
  template <typename Type,
            typename SearchKeyType,
            CUDF_ENABLE_IF(!is_supported_non_nested_type<Type>())>
  std::unique_ptr<column> operator()(lists_column_view const&,
                                     SearchKeyType const&,
                                     bool,
                                     duplicate_find_option,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Unsupported type in `dispatch_index_of` functor.");
  }

  template <typename Type,
            typename SearchKeyType,
            CUDF_ENABLE_IF(is_supported_non_nested_type<Type>())>
  std::unique_ptr<column> operator()(lists_column_view const& lists,
                                     SearchKeyType const& search_keys,
                                     bool search_keys_have_nulls,
                                     duplicate_find_option find_option,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(!cudf::is_nested(lists.child().type()),
                 "Nested types not supported in list search operations.");
    CUDF_EXPECTS(lists.child().type() == search_keys.type(),
                 "Type/Scale of search key does not match list column element type.");
    CUDF_EXPECTS(search_keys.type().id() != type_id::EMPTY, "Type cannot be empty.");

    auto constexpr search_key_is_scalar = std::is_same_v<SearchKeyType, cudf::scalar>;
    if (search_key_is_scalar && search_keys_have_nulls) {
      // If the scalar key is invalid/null, the entire output column will be all nulls.
      return make_numeric_column(data_type{cudf::type_to_id<size_type>()},
                                 lists.size(),
                                 cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                 lists.size(),
                                 stream,
                                 mr);
    }

    auto const lists_cdv_ptr = column_device_view::create(lists.parent(), stream);
    auto const lists_cdv     = cudf::detail::lists_column_device_view{*lists_cdv_ptr};
    auto const keys_dv_ptr   = get_search_keys_device_view_ptr(search_keys, stream);

    auto out_positions = make_numeric_column(
      data_type{type_to_id<size_type>()}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto out_validity   = rmm::device_uvector<bool>(lists.size(), stream);
    auto const out_iter = thrust::make_zip_iterator(
      out_positions->mutable_view().template begin<size_type>(), out_validity.begin());

    auto const keys_iter = cudf::detail::make_optional_iterator<Type>(
      *keys_dv_ptr, nullate::DYNAMIC{search_keys_have_nulls});

    thrust::tabulate(rmm::exec_policy(stream),
                     out_iter,
                     out_iter + lists.size(),
                     search_lists_fn<Type, decltype(keys_iter)>{lists_cdv, keys_iter, find_option});

    if (search_keys_have_nulls || lists.has_nulls()) {
      auto [null_mask, null_count] = cudf::detail::valid_if(
        out_validity.begin(), out_validity.end(), thrust::identity{}, stream, mr);
      out_positions->set_null_mask(std::move(null_mask), null_count);
    }
    return out_positions;
  }
};

/**
 * @brief Converts key-positions vector (from `index_of()`) to a BOOL8 vector, indicating if
 *        the search key(s) were found.
 */
std::unique_ptr<column> to_contains(std::unique_ptr<column>&& key_positions,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(key_positions->type().id() == type_id::INT32,
               "Expected input column of type INT32.");
  auto const positions_begin = key_positions->view().template begin<size_type>();
  auto result                = make_numeric_column(
    data_type{type_id::BOOL8}, key_positions->size(), mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    positions_begin,
                    positions_begin + key_positions->size(),
                    result->mutable_view().template begin<bool>(),
                    [] __device__(auto const i) {
                      // position == NOT_FOUND_SENTINEL: the list does not contain the search key.
                      return i != NOT_FOUND_SENTINEL;
                    });

  auto const null_count                    = key_positions->null_count();
  [[maybe_unused]] auto [_, null_mask, __] = key_positions->release();
  result->set_null_mask(std::move(*null_mask.release()), null_count);

  return result;
}
}  // namespace

namespace detail {
std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(search_key.type(),
                               dispatch_index_of{},
                               lists,
                               search_key,
                               !search_key.is_valid(stream),
                               find_option,
                               stream,
                               mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");
  return cudf::type_dispatcher(search_keys.type(),
                               dispatch_index_of{},
                               lists,
                               search_keys,
                               search_keys.has_nulls(),
                               find_option,
                               stream,
                               mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  return to_contains(
    index_of(lists, search_key, duplicate_find_option::FIND_FIRST, stream), stream, mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(search_keys.size() == lists.size(),
               "Number of search keys must match list column size.");

  return to_contains(
    index_of(lists, search_keys, duplicate_find_option::FIND_FIRST, stream), stream, mr);
}

std::unique_ptr<column> contains_nulls(lists_column_view const& lists,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const lists_cv      = lists.parent();
  auto output              = make_numeric_column(data_type{type_to_id<bool>()},
                                    lists.size(),
                                    copy_bitmask(lists_cv),
                                    lists_cv.null_count(),
                                    stream,
                                    mr);
  auto const out_begin     = output->mutable_view().template begin<bool>();
  auto const lists_cdv_ptr = column_device_view::create(lists_cv, stream);

  thrust::tabulate(rmm::exec_policy(stream),
                   out_begin,
                   out_begin + lists.size(),
                   [lists = cudf::detail::lists_column_device_view{*lists_cdv_ptr}] __device__(
                     auto const list_idx) {
                     auto const list = list_device_view{lists, list_idx};
                     return list.is_null() ||
                            thrust::any_of(thrust::seq,
                                           thrust::make_counting_iterator(0),
                                           thrust::make_counting_iterator(list.size()),
                                           [&list](auto const idx) { return list.is_null(idx); });
                   });

  return output;
}

}  // namespace detail

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_key, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> contains(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(lists, search_keys, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> contains_nulls(lists_column_view const& lists,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_nulls(lists, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 cudf::scalar const& search_key,
                                 duplicate_find_option find_option,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_key, find_option, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> index_of(lists_column_view const& lists,
                                 column_view const& search_keys,
                                 duplicate_find_option find_option,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::index_of(lists, search_keys, find_option, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
