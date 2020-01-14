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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <vector>
#include <algorithm>

namespace cudf
{
namespace dictionary
{
namespace detail
{
namespace
{
struct normalize_fn
{
    template <typename Element, 
    typename std::enable_if_t<not std::is_same<Element, dictionary32_tag>::value>* = nullptr >
    std::unique_ptr<column_view> operator()(column_view const& view) const
    {
        return std::make_unique<column_view>(view);
    }
    template <typename Element, 
    typename std::enable_if_t<std::is_same<Element, dictionary32_tag>::value>* = nullptr >
    std::unique_ptr<column_view> operator()(column_view const& view) const
    {
        return std::make_unique<column_view>(dictionary_column_view(view).indices());
    }
};
}

/**
 * @brief Returns a new table_view by converting any dictionary column views into
 * integer column views.
 *
 * @param inputs Table of columns to normalize.
 * @return New table column view containing no dictionary columns.
 */
table_view normalize( table_view const& inputs )
{
    std::vector<column_view> columns(inputs.num_columns());
    std::transform( inputs.begin(), inputs.end(), columns.begin(),
        [] (column_view const& view) {
            auto col = cudf::experimental::type_dispatcher( view.type(), normalize_fn{}, view);
            return *col;
        });
    return table_view(columns);
}

} // namespace detail
} // namespace dictionary
} // namespace cudf
