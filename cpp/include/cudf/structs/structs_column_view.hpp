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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

namespace cudf {


class structs_column_view : private column_view 
{

    public:

        // Foundation members:
        structs_column_view(structs_column_view const&) = default;
        structs_column_view(structs_column_view &&) = default;
        ~structs_column_view() = default;
        structs_column_view& operator=(structs_column_view const&) = default;
        structs_column_view& operator=(structs_column_view &&) = default;

        explicit structs_column_view(column_view const& rhs);

        using column_view::has_nulls;
        using column_view::null_count;
        using column_view::null_mask;
        using column_view::offset;
        using column_view::size;
        using column_view::child_begin;
        using column_view::child_end;

}; // class structs_column_view;

} // namespace cudf;
