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

namespace cudf {
namespace dictionary {
namespace detail {

/**
 * @brief Accessor functor for returning a dictionary key element in a dictionary iterator.
 *
 * @tparam KeyType The type of the dictionary's key element.
 */
template <typename KeyType>
struct dictionary_access_fn {
  dictionary_access_fn(column_device_view const& d_dictionary) : d_dictionary{d_dictionary} {}

  __device__ KeyType operator()(size_type idx) const
  {
    if (d_dictionary.is_null(idx)) return KeyType{};
    auto keys = d_dictionary.child(dictionary_column_view::keys_column_index);
    return keys.element<KeyType>(static_cast<size_type>(d_dictionary.element<dictionary32>(idx)));
  };

 private:
  column_device_view const d_dictionary;
};

/**
 * @brief Create dictionary iterator that produces key elements.
 *
 * The iterator returns `keys[indices[i]]` where the `keys` are the dictionary's key
 * elements and the `indices` are the dictionary's index elements.
 *
 * @throw cudf::logic_error if `dictionary_column` is not a dictionary column.
 *
 * @tparam KeyType The type of the dictionary's key element.
 * @param dictionary_column The dictionary device view to iterate.
 * @return Iterator
 */
template <typename KeyType>
auto make_dictionary_iterator(column_device_view const& dictionary_column)
{
  CUDF_EXPECTS(dictionary_column.type().id() == type_id::DICTIONARY32,
               "Dictionary iterator is only for dictionary columns");
  return thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                         dictionary_access_fn<KeyType>{dictionary_column});
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
