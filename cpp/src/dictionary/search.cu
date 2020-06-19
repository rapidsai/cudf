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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace dictionary {
namespace detail {

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
            std::enable_if_t<not std::is_same<Element, dictionary32>::value and
                             not std::is_same<Element, list_view>::value>* = nullptr>
  std::unique_ptr<numeric_scalar<int32_t>> operator()(dictionary_column_view const& input,
                                                      scalar const& key,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream) const
  {
    auto result = std::make_unique<numeric_scalar<int32_t>>(0, false, stream, mr);
    if (input.size() == 0) return result;
    CUDF_EXPECTS(input.keys().type() == key.type(),
                 "search key type must match dictionary keys type");
    auto keys_view = column_device_view::create(input.keys(), stream);
    auto find_key  = static_cast<scalar_type_t<Element> const&>(key).value(stream);
    auto iter =
      thrust::equal_range(thrust::device,  // segfaults: rmm::exec_policy(stream)->on(stream) and
                                           // thrust::cuda::par.on(stream)
                          keys_view->begin<Element>(),
                          keys_view->end<Element>(),
                          find_key);
    if (thrust::distance(iter.first, iter.second) > 0)
      result->set_value(thrust::distance(keys_view->begin<Element>(), iter.first), stream);
    return result;
  }
  template <typename Element,
            std::enable_if_t<std::is_same<Element, dictionary32>::value>* = nullptr>
  std::unique_ptr<numeric_scalar<int32_t>> operator()(dictionary_column_view const& input,
                                                      scalar const& key,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream) const
  {
    CUDF_FAIL("dictionary column cannot be the keys column of another dictionary");
  }

  template <typename Element, std::enable_if_t<std::is_same<Element, list_view>::value>* = nullptr>
  std::unique_ptr<numeric_scalar<int32_t>> operator()(dictionary_column_view const& input,
                                                      scalar const& key,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream) const
  {
    CUDF_FAIL("list_view column cannot be the keys column of a dictionary");
  }
};

std::unique_ptr<numeric_scalar<int32_t>> get_index(dictionary_column_view const& dictionary,
                                                   scalar const& key,
                                                   rmm::mr::device_memory_resource* mr,
                                                   cudaStream_t stream)
{
  return type_dispatcher(dictionary.keys().type(), find_index_fn(), dictionary, key, mr, stream);
}

}  // namespace detail

// external API

std::unique_ptr<numeric_scalar<int32_t>> get_index(dictionary_column_view const& dictionary,
                                                   scalar const& key,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_index(dictionary, key, mr);
}

}  // namespace dictionary
}  // namespace cudf
