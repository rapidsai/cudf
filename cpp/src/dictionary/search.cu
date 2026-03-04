/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace dictionary {
namespace detail {

namespace {

struct dispatch_scalar_index {
  template <typename IndexType>
  std::unique_ptr<scalar> operator()(size_type index,
                                     bool is_valid,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_index_type<IndexType>())
  {
    return std::make_unique<numeric_scalar<IndexType>>(index, is_valid, stream, mr);
  }
  template <typename IndexType, typename... Args>
  std::unique_ptr<scalar> operator()(Args&&...)
    requires(not is_index_type<IndexType>())
  {
    CUDF_FAIL("indices must be an integral type");
  }
};

/**
 * @brief Find index of a given key within a dictionary's keys column.
 *
 * The index is the position within the keys column where the given key (scalar) is found.
 * The result is an integer scalar identifying the index value.
 * If the key is not found, the resulting scalar has `is_valid()=false`.
 */
struct find_index_fn {
  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const& input,
                                     scalar const& key,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(not std::is_same_v<Element, dictionary32> and
             not std::is_same_v<Element, list_view> and not std::is_same_v<Element, struct_view>)
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
    auto const begin = keys_view->begin<Element>();
    auto const end   = keys_view->end<Element>();
    auto const iter  = thrust::find(rmm::exec_policy_nosync(stream), begin, end, find_key);
    return type_dispatcher(input.indices().type(),
                           dispatch_scalar_index{},
                           cuda::std::distance(begin, iter),
                           iter != end,
                           stream,
                           mr);
  }

  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const&,
                                     scalar const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(std::is_same_v<Element, dictionary32> or std::is_same_v<Element, list_view> or
             std::is_same_v<Element, struct_view>)
  {
    CUDF_FAIL(
      "dictionary, list_view, and struct_view columns cannot be the keys column of a dictionary");
  }
};

struct find_insert_index_fn {
  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const& input,
                                     scalar const& key,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(not std::is_same_v<Element, dictionary32> and
             not std::is_same_v<Element, list_view> and not std::is_same_v<Element, struct_view>)
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
    auto const begin = keys_view->begin<Element>();
    auto iter =
      thrust::find(rmm::exec_policy_nosync(stream), begin, begin + keys_view->size(), find_key);
    return type_dispatcher(input.indices().type(),
                           dispatch_scalar_index{},
                           cuda::std::distance(begin, iter),
                           true,
                           stream,
                           mr);
  }

  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const&,
                                     scalar const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(std::is_same_v<Element, dictionary32> or std::is_same_v<Element, list_view> or
             std::is_same_v<Element, struct_view>)
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
