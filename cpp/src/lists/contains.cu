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

#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <type_traits>

namespace cudf {
namespace lists {

namespace {
auto constexpr NOT_FOUND_IDX = size_type{-1};

template <typename Type>
auto constexpr is_supported_non_nested_type()
{
  return cudf::is_numeric<Type>() || cudf::is_chrono<Type>() || cudf::is_fixed_point<Type>() ||
         std::is_same_v<Type, cudf::string_view>;
}

template <typename Type>
auto constexpr is_supported_type()
{
  return is_supported_non_nested_type<Type>() || cudf::is_struct_type<Type>();
}

template <bool find_first>
auto __device__ element_idx_begin([[maybe_unused]] size_type size)
{
  if constexpr (find_first) {
    return thrust::make_counting_iterator<size_type>(0);
  } else {
    return thrust::make_reverse_iterator(thrust::make_counting_iterator<size_type>(size));
  }
}

template <bool find_first>
auto __device__ element_idx_end([[maybe_unused]] size_type size)
{
  if constexpr (find_first) {
    return thrust::make_counting_iterator<size_type>(size);
  } else {
    return thrust::make_reverse_iterator(thrust::make_counting_iterator<size_type>(0));
  }
}

template <bool find_first, typename Iterator>
size_type __device__ distance(Iterator begin, Iterator end, Iterator found_iter)
{
  if (found_iter == end) { return NOT_FOUND_IDX; }
  return find_first ? found_iter - begin : end - found_iter - 1;
}

/**
 * @brief Functor for searching index of the given key in a list.
 */
template <typename Type, typename Enable = void>
struct search_functor {
  template <typename... Args>
  void operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type in `search_functor`.");
  }
};

/**
 * @brief The search_functor specialized for non-struct types.
 */
template <typename Type>
struct search_functor<Type, std::enable_if_t<is_supported_non_nested_type<Type>()>> {
  /**
   * @brief tba
   *
   * Search index of a given element in a list defined by offsets pointing to rows of the
   * child column of the original lists column.
   */
  template <bool find_first>
  static __device__ size_type search_list(list_device_view const& list, Type const& search_key)
  {
    auto const begin = element_idx_begin<find_first>(list.size());
    auto const end   = element_idx_end<find_first>(list.size());
    auto const found_iter =
      thrust::find_if(thrust::seq, begin, end, [&list, search_key] __device__(auto const idx) {
        return !list.is_null(idx) &&
               cudf::equality_compare(list.template element<Type>(idx), search_key);
      });
    return distance<find_first>(begin, end, found_iter);
  }

  /**
   * @brief Search for the index of the corresponding key (given in a column) in all list rows.
   */
  template <typename SearchKeyIter, typename OutputPairIter>
  static void search_all_lists(cudf::detail::lists_column_device_view const& lists,
                               SearchKeyIter const& keys_iter,
                               duplicate_find_option find_option,
                               OutputPairIter const& out_iter,
                               rmm::cuda_stream_view stream)
  {
    thrust::tabulate(rmm::exec_policy(stream),
                     out_iter,
                     out_iter + lists.size(),
                     [lists, keys_iter, find_option, NOT_FOUND_IDX = NOT_FOUND_IDX] __device__(
                       auto list_idx) -> thrust::pair<size_type, bool> {
                       auto const key_opt = keys_iter[list_idx];
                       auto const list    = list_device_view{lists, list_idx};
                       if (list.is_null() || !key_opt) { return {NOT_FOUND_IDX, false}; }

                       auto const key = key_opt.value();
                       return find_option == duplicate_find_option::FIND_FIRST
                                ? thrust::pair<size_type, bool>{search_list<true>(list, key), true}
                                : thrust::pair<size_type, bool>{search_list<false>(list, key),
                                                                true};
                     });
  }
};

/**
 * @brief The search_functor specialized for struct type.
 */
template <typename Type>
struct search_functor<Type, std::enable_if_t<is_struct_type<Type>()>> {
  /**
   * @brief tba
   */
  template <bool find_first, typename EqComparator>
  static __device__ size_type search_list(list_device_view const& list,
                                          EqComparator const& dcomp,
                                          bool search_key_is_scalar)
  {
    auto const begin = element_idx_begin<find_first>(list.size());
    auto const end   = element_idx_end<find_first>(list.size());
    // todo: check offset
    auto const found_iter = thrust::find_if(
      thrust::seq, begin, end, [&list, dcomp, search_key_is_scalar] __device__(auto const idx) {
        if (!list.is_null()) {
          printf(
            "idx: , size = %d, offset = %d => %d\n", list.size(), idx, list.element_offset(idx));
        }

        return !list.is_null(idx) &&
               dcomp(list.element_offset(idx), search_key_is_scalar ? 0 : list.row_index());
      });
    return distance<find_first>(begin, end, found_iter);
  }

  /**
   * @brief Search for the index of the corresponding key (given in a column) in all list rows.
   */
  template <typename KeyValidityIter, typename EqComparator, typename OutputPairIter>
  static void search_all_lists(cudf::detail::lists_column_device_view const& lists,
                               bool search_key_is_scalar,
                               KeyValidityIter const& key_validity_iter,
                               EqComparator const& dcomp,
                               duplicate_find_option find_option,
                               OutputPairIter const& out_iter,
                               rmm::cuda_stream_view stream)
  {
    thrust::tabulate(
      rmm::exec_policy(stream),
      out_iter,
      out_iter + lists.size(),
      [lists,
       search_key_is_scalar,
       key_validity_iter,
       dcomp,
       find_option,
       NOT_FOUND_IDX = NOT_FOUND_IDX] __device__(auto list_idx) -> thrust::pair<size_type, bool> {
        auto const list = list_device_view{lists, list_idx};
        if (list.is_null() || !key_validity_iter[list_idx]) { return {NOT_FOUND_IDX, false}; }

        return find_option == duplicate_find_option::FIND_FIRST
                 ? thrust::pair<size_type, bool>{search_list<true>(
                                                   list, dcomp, search_key_is_scalar),
                                                 true}
                 : thrust::pair<size_type, bool>{
                     search_list<false>(list, dcomp, search_key_is_scalar), true};
      });
  }
};

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

template <typename SearchKeyType>
auto get_search_keys_table_view(SearchKeyType const& search_keys)
{
  if constexpr (std::is_same_v<SearchKeyType, cudf::scalar>) {
    auto const children = static_cast<struct_scalar const*>(&search_keys)->view();
    auto const parent   = column_view{
      data_type{type_id::STRUCT}, 1, nullptr, nullptr, 0, 0, {children.begin(), children.end()}};
    return table_view{{parent}};
  } else {
    return table_view{{search_keys}};
  }
}

/**
 * @brief TBA
 */
struct dispatch_index_of {
  template <typename Type, typename... Args>
  std::enable_if_t<!is_supported_type<Type>(), std::unique_ptr<column>> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported type in `dispatch_index_of` functor.");
  }

  template <typename Type, typename SearchKeyType>
  std::enable_if_t<is_supported_type<Type>(), std::unique_ptr<column>> operator()(
    lists_column_view const& lists,
    SearchKeyType const& search_keys,
    bool search_keys_have_nulls,
    duplicate_find_option find_option,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto constexpr search_key_is_scalar = std::is_same_v<SearchKeyType, cudf::scalar>;
    if constexpr (!search_key_is_scalar) {
      CUDF_EXPECTS(search_keys.size() == lists.size(),
                   "Number of search keys must match list column size.");
    }

    // todo: explain why not sliced
    auto const child = lists.child();
    CUDF_EXPECTS(!cudf::is_nested(child.type()) || child.type().id() == type_id::STRUCT,
                 "Nested types except STRUCT are not supported in list search operations.");
    CUDF_EXPECTS(child.type() == search_keys.type(),
                 "Type/Scale of search key does not match list column element type.");
    CUDF_EXPECTS(search_keys.type().id() != type_id::EMPTY, "Type cannot be empty.");

    if (search_key_is_scalar && search_keys_have_nulls) {
      return make_numeric_column(data_type(type_id::INT32),
                                 lists.size(),
                                 cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                 lists.size(),
                                 stream,
                                 mr);
    }

    auto const lists_cdv_ptr = column_device_view::create(lists.parent(), stream);
    auto const lists_cdv     = cudf::detail::lists_column_device_view{*lists_cdv_ptr};
    auto const keys_dv_ptr   = get_search_keys_device_view_ptr(search_keys, stream);

    auto const output_has_nulls = search_keys_have_nulls || lists.has_nulls();

    auto out_positions = make_numeric_column(
      data_type{type_id::INT32}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto out_validity   = rmm::device_uvector<bool>(lists.size(), stream);
    auto const out_iter = thrust::make_zip_iterator(
      out_positions->mutable_view().template begin<size_type>(), out_validity.begin());

    if constexpr (std::is_same_v<Type, cudf::struct_view>) {
      auto const child_tview = table_view{{child}};
      auto const keys_tview  = get_search_keys_table_view(search_keys);

      auto const has_any_nulls = has_nested_nulls(child_tview) || has_nested_nulls(keys_tview);
      auto const comp =
        cudf::experimental::row::equality::table_comparator(child_tview, keys_tview, stream);
      auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

      auto const key_validity_iter = cudf::detail::make_validity_iterator<true>(*keys_dv_ptr);

      search_functor<Type>::search_all_lists(
        lists_cdv, search_key_is_scalar, key_validity_iter, dcomp, find_option, out_iter, stream);
    } else {  // other types that are not struct
      auto const keys_iter = cudf::detail::make_optional_iterator<Type>(
        *keys_dv_ptr, nullate::DYNAMIC{search_keys_have_nulls});
      search_functor<Type>::search_all_lists(lists_cdv, keys_iter, find_option, out_iter, stream);
    }

    if (output_has_nulls) {
      auto [null_mask, num_nulls] = cudf::detail::valid_if(
        out_validity.begin(), out_validity.end(), thrust::identity{}, stream, mr);
      out_positions->set_null_mask(std::move(null_mask), num_nulls);
    }
    return out_positions;
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
  auto const positions_begin = key_positions->view().template begin<size_type>();
  auto result =
    make_numeric_column(data_type{type_id::BOOL8}, num_rows, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    positions_begin,
                    positions_begin + num_rows,
                    result->mutable_view().template begin<bool>(),
                    [] __device__(auto i) { return i != NOT_FOUND_IDX; });
  [[maybe_unused]] auto [_, null_mask, __] = key_positions->release();
  result->set_null_mask(std::move(*null_mask));
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
  auto const num_rows = lists.size();
  auto output =
    make_numeric_column(data_type{type_id::BOOL8}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto const output_begin  = output->mutable_view().template begin<bool>();
  auto const lists_cdv_ptr = column_device_view::create(lists.parent(), stream);
  auto const it_begin      = thrust::make_counting_iterator(0);

  thrust::tabulate(
    rmm::exec_policy(stream),
    output_begin,
    output_begin + num_rows,
    [it_begin,
     lists = cudf::detail::lists_column_device_view{*lists_cdv_ptr}] __device__(auto list_idx) {
      auto const list = list_device_view{lists, list_idx};
      return list.is_null() ||
             thrust::any_of(thrust::seq, it_begin, it_begin + list.size(), [&list](auto idx) {
               return list.is_null(idx);
             });
    });

  auto const lists_validity_iter = cudf::detail::make_validity_iterator<true>(*lists_cdv_ptr);
  auto [null_mask, null_count]   = cudf::detail::valid_if(
    lists_validity_iter, lists_validity_iter + num_rows, thrust::identity<bool>{}, stream, mr);
  output->set_null_mask(std::move(null_mask), null_count);

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

}  // namespace lists
}  // namespace cudf
