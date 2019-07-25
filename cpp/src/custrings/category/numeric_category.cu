/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "numeric_category.inl"

size_t count_nulls( const BYTE* nulls, size_t count )
{
    if( !nulls || !count )
        return 0;
    auto execpol = rmm::exec_policy(0);
    size_t result = thrust::count_if( execpol->on(0), thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(count),
            [nulls] __device__ (size_t idx) { return ((nulls[idx/8] & (1 << (idx % 8)))==0); });
    return result;
}

//
template<> const char* numeric_category<char>::get_type_name() { return "int8"; };
template class numeric_category<char>;
