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
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/dictionary/search.hpp>

#include <thrust/copy.h>

namespace cudf
{
namespace dictionary
{
namespace detail
{

template<typename Element>
struct copy_if_fn
{
    column_device_view const d_keys;
    Element const d_key;

    __device__ bool operator()(size_type idx) const
    {
        if( d_keys.is_null(idx) )
            return false;
        return d_key == d_keys.element<Element>(idx);
    }
};

struct find_index_fn
{
    template<typename Element, std::enable_if_t<not std::is_same<Element, dictionary32>::value>* = nullptr>
    std::unique_ptr<numeric_scalar<int32_t>> operator()( dictionary_column_view const& input, scalar const& key,
                                                         rmm::mr::device_memory_resource* mr, cudaStream_t stream ) const
    {
        auto result = std::make_unique<numeric_scalar<int32_t>>( 0, false, stream, mr );
        auto keys_view = column_device_view::create(input.keys(),stream);
        auto find_key = static_cast<experimental::scalar_type_t<Element> const&>(key);
        auto iter = thrust::copy_if( rmm::exec_policy(stream)->on(stream), 
            thrust::make_counting_iterator<size_type>(0),
            thrust::make_counting_iterator<size_type>(input.keys_size()),
            result->data(), copy_if_fn<Element>{*keys_view,find_key.value(stream)});
        if( iter != result->data() )
            result->set_valid(true,stream);
        return result;
    }
    template<typename Element, std::enable_if_t<std::is_same<Element, dictionary32>::value>* = nullptr>
    std::unique_ptr<numeric_scalar<int32_t>> operator()( dictionary_column_view const& input, scalar const& key,
                                                         rmm::mr::device_memory_resource* mr, cudaStream_t stream ) const
    {
        assert("dictionary column cannot be the keys column of another dictionary");
        return nullptr;
    }
};

std::unique_ptr<numeric_scalar<int32_t>> get_index( dictionary_column_view const& dictionary, scalar const& key,
                                                    rmm::mr::device_memory_resource* mr, cudaStream_t stream = 0 )
{
    return experimental::type_dispatcher( dictionary.keys().type(), find_index_fn(), dictionary, key, mr, stream );
}

} // namespace detail

std::unique_ptr<numeric_scalar<int32_t>> get_index( dictionary_column_view const& dictionary, scalar const& key,
                                                    rmm::mr::device_memory_resource* mr )
{
    return detail::get_index( dictionary, key, mr );
}

} // namespace dictionary
} // namespace cudf
