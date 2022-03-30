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
#include <thrust/logical.h>

#include <type_traits>

namespace cudf {
namespace lists {

namespace {

auto constexpr NOT_FOUND_IDX = size_type{-1};

auto get_search_keys_device_iterable_view(column_view const& search_keys,
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
    return NOT_FOUND_IDX;  // Not found.
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
        auto const [element, element_is_valid] = element_and_validity;
        return element_is_valid && cudf::equality_compare(element, search_key);
      });
    return distance<find_option>(list_begin, list_end, find_iter);
  }
};

/**
 * @brief Search for the index of the corresponding key in each list row.
 */
template <typename ElementType, typename SearchKeyPairIter, typename OutputPairIter>
void search_each_list_row(column_device_view const& d_lists,
                          SearchKeyPairIter search_key_pair_iter,
                          bool search_keys_have_nulls,
                          duplicate_find_option find_option,
                          OutputPairIter const& output_iters,
                          rmm::cuda_stream_view stream)
{
  thrust::tabulate(
    rmm::exec_policy(stream),
    output_iters,
    output_iters + d_lists.size(),
    [dv_lists = lists_column_device_view{d_lists},
     search_key_pair_iter,
     find_option,
     search_keys_have_nulls = search_keys_have_nulls,
     NOT_FOUND_IDX = NOT_FOUND_IDX] __device__(auto row_index) -> thrust::pair<size_type, bool> {
      auto const [search_key, search_key_is_valid] = search_key_pair_iter[row_index];
      if (search_keys_have_nulls && !search_key_is_valid) { return {NOT_FOUND_IDX, false}; }

      auto const list = list_device_view(dv_lists, row_index);
      if (list.is_null()) { return {NOT_FOUND_IDX, false}; }

      auto const position = find_option == duplicate_find_option::FIND_FIRST
                              ? finder<duplicate_find_option::FIND_FIRST>{}(list, search_key)
                              : finder<duplicate_find_option::FIND_LAST>{}(list, search_key);
      return {position, true};
    });
}

/**
 * @brief Functor to search each list row for the specified search keys.
 */
struct lookup_functor {
  template <typename ElementType>
  auto static constexpr is_supported()
  {
    return cudf::is_numeric<ElementType>() || cudf::is_chrono<ElementType>() ||
           cudf::is_fixed_point<ElementType>() || std::is_same_v<ElementType, cudf::string_view> ||
           std::is_same_v<ElementType, cudf::struct_view>;
  }

  template <typename ElementType, typename... Args>
  std::enable_if_t<!is_supported<ElementType>(), std::unique_ptr<column>> operator()(
    Args&&...) const
  {
    CUDF_FAIL("Unsupported type in list search operation.");
  }

  template <typename ElementType, typename SearchKeyType>
  std::enable_if_t<is_supported<ElementType>(), std::unique_ptr<column>> operator()(
    lists_column_view const& lists,
    SearchKeyType const& search_key,
    bool search_keys_have_nulls,
    duplicate_find_option find_option,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto constexpr search_key_is_scalar = std::is_same_v<SearchKeyType, cudf::scalar>;
    if constexpr (!search_key_is_scalar) {
      CUDF_EXPECTS(search_key.size() == lists.size(),
                   "Number of search keys must match list column size.");
    }
    CUDF_EXPECTS(
      !cudf::is_nested(lists.child().type()) || lists.child().type().id() == type_id::STRUCT,
      "Nested types except STRUCT are not supported in list search operations.");
    CUDF_EXPECTS(lists.child().type() == search_key.type(),
                 "Type/Scale of search key does not match list column element type.");
    CUDF_EXPECTS(search_key.type().id() != type_id::EMPTY, "Type cannot be empty.");

    if (search_key_is_scalar && search_keys_have_nulls) {
      return make_numeric_column(data_type(type_id::INT32),
                                 lists.size(),
                                 cudf::create_null_mask(lists.size(), mask_state::ALL_NULL, mr),
                                 lists.size(),
                                 stream,
                                 mr);
    }

    auto const d_lists_ptr = column_device_view::create(lists.parent(), stream);
    auto const d_skeys     = get_search_keys_device_iterable_view(search_key, stream);

    auto out_positions = make_numeric_column(
      data_type{type_id::INT32}, lists.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto out_validity       = rmm::device_uvector<bool>(lists.size(), stream);
    auto const output_iters = thrust::make_zip_iterator(
      out_positions->mutable_view().template begin<size_type>(), out_validity.begin());

    if constexpr (std::is_same_v<ElementType, cudf::struct_view>) {
      //
    } else {  // not struct type
      auto const do_search = [&](auto const& search_key_iter) {
        search_each_list_row<ElementType>(
          *d_lists_ptr, search_key_iter, search_keys_have_nulls, find_option, output_iters, stream);
      };

      if (search_keys_have_nulls) {
        do_search(cudf::detail::make_pair_rep_iterator<ElementType, true>(*d_skeys));
      } else {
        do_search(cudf::detail::make_pair_rep_iterator<ElementType, false>(*d_skeys));
      }
    }

    if (search_keys_have_nulls || lists.has_nulls() || lists.child().has_nulls()) {
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
                               lookup_functor{},
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
                               lookup_functor{},
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

std::unique_ptr<column> contains_nulls(lists_column_view const& input_lists,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const num_rows   = input_lists.size();
  auto const d_lists    = column_device_view::create(input_lists.parent());
  auto has_nulls_output = make_numeric_column(
    data_type{type_id::BOOL8}, input_lists.size(), mask_state::UNALLOCATED, stream, mr);
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

std::unique_ptr<column> contains_nulls(lists_column_view const& input_lists,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_nulls(input_lists, rmm::cuda_stream_default, mr);
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
