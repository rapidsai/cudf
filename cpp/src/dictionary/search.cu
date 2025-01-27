/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace dictionary {
namespace detail {

namespace {

struct dispatch_scalar_index {
  template <typename IndexType, std::enable_if_t<is_index_type<IndexType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(size_type index,
                                     bool is_valid,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    return std::make_unique<numeric_scalar<IndexType>>(index, is_valid, stream, mr);
  }
  template <typename IndexType,
            typename... Args,
            std::enable_if_t<not is_index_type<IndexType>()>* = nullptr>
  std::unique_ptr<scalar> operator()(Args&&...)
  {
    CUDF_FAIL("indices must be an integral type");
  }
};

/**
 * @brief Find index of a given key within a dictionary's keys column.
 *
 * The index is the position within the keys column where the given key (scalar) is found.
 * The keys column is sorted and unique so only one value is expected.
 * The result is an integer scalar identifying the index value.
 * If the key is not found, the resulting scalar has `is_valid()=false`.
 */
struct find_index_fn {
  template <typename Element,
            std::enable_if_t<not std::is_same_v<Element, dictionary32> and
                             not std::is_same_v<Element, list_view> and
                             not std::is_same_v<Element, struct_view>>* = nullptr>
  std::unique_ptr<scalar> operator()(dictionary_column_view const& input,
                                     scalar const& key,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    if (!key.is_valid(stream)) {
      return type_dispatcher(input.indices().type(), dispatch_scalar_index{}, 0, false, stream, mr);
    }
    CUDF_EXPECTS(cudf::have_same_types(input.parent(), key),
                 "search key type must match dictionary keys type",
                 cudf::data_type_error);

    using ScalarType = cudf::scalar_type_t<Element>;
    auto find_key    = static_cast<ScalarType const&>(key).value(stream);
    auto keys_view   = column_device_view::create(input.keys(), stream);
    auto iter        = thrust::equal_range(
      rmm::exec_policy(stream), keys_view->begin<Element>(), keys_view->end<Element>(), find_key);
    return type_dispatcher(input.indices().type(),
                           dispatch_scalar_index{},
                           thrust::distance(keys_view->begin<Element>(), iter.first),
                           (thrust::distance(iter.first, iter.second) > 0),
                           stream,
                           mr);
  }

  template <
    typename Element,
    std::enable_if_t<std::is_same_v<Element, dictionary32> or std::is_same_v<Element, list_view> or
                     std::is_same_v<Element, struct_view>>* = nullptr>
  std::unique_ptr<scalar> operator()(dictionary_column_view const&,
                                     scalar const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL(
      "dictionary, list_view, and struct_view columns cannot be the keys column of a dictionary");
  }
};

struct find_insert_index_fn {
  template <typename Element,
            std::enable_if_t<not std::is_same_v<Element, dictionary32> and
                             not std::is_same_v<Element, list_view> and
                             not std::is_same_v<Element, struct_view>>* = nullptr>
  std::unique_ptr<scalar> operator()(dictionary_column_view const& input,
                                     scalar const& key,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    if (!key.is_valid(stream)) {
      return type_dispatcher(input.indices().type(), dispatch_scalar_index{}, 0, false, stream, mr);
    }
    CUDF_EXPECTS(cudf::have_same_types(input.parent(), key),
                 "search key type must match dictionary keys type",
                 cudf::data_type_error);

    using ScalarType = cudf::scalar_type_t<Element>;
    auto find_key    = static_cast<ScalarType const&>(key).value(stream);
    auto keys_view   = column_device_view::create(input.keys(), stream);
    auto iter        = thrust::lower_bound(
      rmm::exec_policy(stream), keys_view->begin<Element>(), keys_view->end<Element>(), find_key);
    return type_dispatcher(input.indices().type(),
                           dispatch_scalar_index{},
                           thrust::distance(keys_view->begin<Element>(), iter),
                           true,
                           stream,
                           mr);
  }

  template <
    typename Element,
    std::enable_if_t<std::is_same_v<Element, dictionary32> or std::is_same_v<Element, list_view> or
                     std::is_same_v<Element, struct_view>>* = nullptr>
  std::unique_ptr<scalar> operator()(dictionary_column_view const&,
                                     scalar const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("dictionary, list_view, and struct_view columns cannot be the keys for a dictionary");
  }
};

}  // namespace

std::unique_ptr<scalar> get_index(dictionary_column_view const& dictionary,
                                  scalar const& key,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  if (dictionary.is_empty()) {
    return std::make_unique<numeric_scalar<int32_t>>(0, false, stream, mr);
  }
  return type_dispatcher<dispatch_storage_type>(
    dictionary.keys().type(), find_index_fn(), dictionary, key, stream, mr);
}

std::unique_ptr<scalar> get_insert_index(dictionary_column_view const& dictionary,
                                         scalar const& key,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (dictionary.is_empty()) {
    return std::make_unique<numeric_scalar<int32_t>>(0, false, stream, mr);
  }
  return type_dispatcher<dispatch_storage_type>(
    dictionary.keys().type(), find_insert_index_fn(), dictionary, key, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<scalar> get_index(dictionary_column_view const& dictionary,
                                  scalar const& key,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_index(dictionary, key, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
