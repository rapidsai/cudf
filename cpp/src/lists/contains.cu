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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

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

template <duplicate_find_option find_option>
auto __device__ list_begin(size_type const* d_offsets, size_type list_idx)
{
  if constexpr (find_option == duplicate_find_option::FIND_FIRST) {
    return thrust::make_counting_iterator<size_type>(d_offsets[list_idx]);
  } else {
    return thrust::make_reverse_iterator(
      thrust::make_counting_iterator<size_type>(d_offsets[list_idx + 1]));
  }
}

template <duplicate_find_option find_option>
auto __device__ list_end(size_type const* d_offsets, size_type list_idx)
{
  if constexpr (find_option == duplicate_find_option::FIND_FIRST) {
    return thrust::make_counting_iterator<size_type>(d_offsets[list_idx + 1]);
  } else {
    return thrust::make_reverse_iterator(
      thrust::make_counting_iterator<size_type>(d_offsets[list_idx]));
  }
}

template <duplicate_find_option find_option, typename Iterator>
size_type __device__ distance(Iterator begin, Iterator end, Iterator found_iter)
{
  if (found_iter == end) { return NOT_FOUND_IDX; }
  return find_option == duplicate_find_option::FIND_FIRST ? found_iter - begin
                                                          : end - found_iter - 1;
}

/**
 * @brief Functor for searching index of the given key in a list.
 */
template <typename Type, duplicate_find_option find_option, typename Enable = void>
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
template <typename Type, duplicate_find_option find_option>
struct search_functor<Type, find_option, std::enable_if_t<is_supported_non_nested_type<Type>()>> {
  /**
   * @brief Search index of a given scalar in a list defined by offsets pointing to rows of the
   * child column of the original lists column.
   */
  static __device__ size_type search_list(column_device_view const& d_child,
                                          size_type const* d_offsets,
                                          size_type list_idx,
                                          bool has_null_elements,
                                          Type const& search_key)
  {
    auto const begin = list_begin<find_option>(d_offsets, list_idx);
    auto const end   = list_end<find_option>(d_offsets, list_idx);
    auto const found_iter =
      thrust::find_if(thrust::seq, begin, end, [&] __device__(auto const idx) {
        auto const element  = d_child.template element<Type>(idx);
        auto const is_valid = !has_null_elements || d_child.is_valid_nocheck(idx);
        return is_valid && cudf::equality_compare(element, search_key);
      });
    return distance<find_option>(begin, end, found_iter);
  }

  /**
   * @brief Search for the index of the given scalar in all list rows.
   */
  template <typename OutputPairIter>
  void search_all_lists(column_device_view const& d_lists,
                        column_device_view const& d_child,
                        size_type const* d_offsets,
                        bool has_null_lists,
                        bool has_null_elements,
                        cudf::scalar const& search_key,
                        bool search_key_is_null,
                        OutputPairIter const& out_iters,
                        rmm::cuda_stream_view stream) const
  {
    auto const key_accessor = cudf::detail::make_pair_rep_iterator<Type>(search_key);

    thrust::tabulate(
      rmm::exec_policy(stream),
      out_iters,
      out_iters + d_lists.size(),
      [d_lists,
       d_child,
       d_offsets,
       has_null_lists,
       has_null_elements,
       key_accessor,
       search_key_is_null,
       NOT_FOUND_IDX = NOT_FOUND_IDX] __device__(auto list_idx) -> thrust::pair<size_type, bool> {
        if (search_key_is_null || (has_null_lists && d_lists.is_null_nocheck(list_idx))) {
          return {NOT_FOUND_IDX, false};
        }

        return {search_list(d_child, d_offsets, list_idx, has_null_elements, key_accessor[0].first),
                true};
      });
  }

  /**
   * @brief Search for the index of the corresponding key (given in a column) in all list rows.
   */
  template <typename OutputPairIter>
  void search_all_lists(column_device_view const& d_lists,
                        column_device_view const& d_child,
                        size_type const* d_offsets,
                        bool has_null_lists,
                        bool has_null_elements,
                        column_device_view const& d_search_keys,
                        bool search_keys_have_nulls,
                        OutputPairIter const& out_iters,
                        rmm::cuda_stream_view stream) const
  {
    thrust::tabulate(
      rmm::exec_policy(stream),
      out_iters,
      out_iters + d_lists.size(),
      [d_lists,
       d_child,
       d_offsets,
       has_null_lists,
       has_null_elements,
       d_search_keys,
       search_keys_have_nulls,
       NOT_FOUND_IDX = NOT_FOUND_IDX] __device__(auto list_idx) -> thrust::pair<size_type, bool> {
        if (search_keys_have_nulls && d_search_keys.is_null_nocheck(list_idx)) {
          return {NOT_FOUND_IDX, false};
        }
        if (has_null_lists && d_lists.is_null_nocheck(list_idx)) { return {NOT_FOUND_IDX, false}; }

        auto const key = d_search_keys.template element<Type>(list_idx);
        return {search_list(d_child, d_offsets, list_idx, has_null_elements, key), true};
      });
  }
};

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
    auto constexpr scalar_search_key = std::is_same_v<SearchKeyType, cudf::scalar>;
    if constexpr (!scalar_search_key) {
      CUDF_EXPECTS(search_keys.size() == lists.size(),
                   "Number of search keys must match list column size.");
    }

    auto const child = lists.get_sliced_child(stream);
    CUDF_EXPECTS(!cudf::is_nested(child.type()) || child.type().id() == type_id::STRUCT,
                 "Nested types except STRUCT are not supported in list search operations.");
    CUDF_EXPECTS(child.type() == search_keys.type(),
                 "Type/Scale of search key does not match list column element type.");
    CUDF_EXPECTS(search_keys.type().id() != type_id::EMPTY, "Type cannot be empty.");

    if (scalar_search_key && search_keys_have_nulls) {
      return make_numeric_column(data_type(type_id::INT32),
                                 lists.size(),
                                 cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                 lists.size(),
                                 stream,
                                 mr);
    }

    auto const d_lists_ptr = column_device_view::create(lists.parent(), stream);

    auto out_positions = make_numeric_column(
      data_type{type_id::INT32}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto out_validity    = rmm::device_uvector<bool>(lists.size(), stream);
    auto const out_iters = thrust::make_zip_iterator(
      out_positions->mutable_view().template begin<size_type>(), out_validity.begin());

    if constexpr (std::is_same_v<Type, cudf::struct_view>) {
      //
    } else {  // not struct type
      auto const do_search = [&](auto const& searcher) {
        auto const d_child_ptr = column_device_view::create(child, stream);
        searcher.search_all_lists(*d_lists_ptr,
                                  *d_child_ptr,
                                  lists.offsets_begin(),
                                  lists.has_nulls(),
                                  child.has_nulls(),
                                  search_keys,
                                  search_keys_have_nulls,
                                  out_iters,
                                  stream);
      };

      if (find_option == duplicate_find_option::FIND_FIRST) {
        auto const searcher = search_functor<Type, duplicate_find_option::FIND_FIRST>{};
        do_search(searcher);
      } else {
        auto const searcher = search_functor<Type, duplicate_find_option::FIND_LAST>{};
        do_search(searcher);
      }
    }

    if (search_keys_have_nulls || lists.has_nulls() || child.has_nulls()) {
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
  auto const d_lists  = column_device_view::create(lists.parent());
  auto has_nulls_output =
    make_numeric_column(data_type{type_id::BOOL8}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto const output_begin = has_nulls_output->mutable_view().template begin<bool>();
  thrust::tabulate(rmm::exec_policy(stream),
                   output_begin,
                   output_begin + num_rows,
                   [lists = lists_column_device_view{*d_lists}] __device__(auto list_idx) {
                     auto const list       = list_device_view{lists, list_idx};
                     auto const list_begin = thrust::make_counting_iterator(size_type{0});
                     return list.is_null() ||
                            thrust::any_of(
                              thrust::seq, list_begin, list_begin + list.size(), [&list](auto i) {
                                return list.is_null(i);
                              });
                   });
  auto const validity_begin = cudf::detail::make_counting_transform_iterator(
    0, [lists = lists_column_device_view{*d_lists}] __device__(auto list_idx) {
      return not list_device_view{lists, list_idx}.is_null();
    });
  auto [null_mask, num_nulls] = cudf::detail::valid_if(
    validity_begin, validity_begin + num_rows, thrust::identity<bool>{}, stream, mr);
  has_nulls_output->set_null_mask(std::move(null_mask), num_nulls);
  return has_nulls_output;
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
