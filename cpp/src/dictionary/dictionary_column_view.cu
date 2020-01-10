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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/copy.hpp>

namespace cudf
{

//
dictionary_column_view::dictionary_column_view( column_view strings_column )
    : column_view(strings_column)
{
    CUDF_EXPECTS( type().id()==DICTIONARY32, "dictionary_column_view only supports DICTIONARY type");
}

column_view dictionary_column_view::parent() const
{
    return static_cast<column_view>(*this);
}

column_view dictionary_column_view::indices() const
{
    CUDF_EXPECTS( num_children()>0, "dictionary column has no children" );
    return child(0);
}

size_type dictionary_column_view::keys_size() const noexcept
{
    if( size()==0 )
        return 0;
    return dictionary_keys()->size();
}

} // namespace cudf
