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

// #include <cudf/lists/list_view.cuh>
// #include <cudf/column/column_device_view.cuh>

namespace cudf {

namespace detail {

/*
class lists_column_device_view 
// TODO: Required for recursion.
// : private column_device_view 
{
    public:

        lists_column_device_view() = delete;

        ~lists_column_device_view() = default;
        lists_column_device_view(lists_column_device_view const&) = default;
        lists_column_device_view(lists_column_device_view &&) = default;
        
        lists_column_device_view(
            column_device_view const& child,    // Holds data.
            column_device_view const& offsets   // Holds list offsets.
        ) 
        : d_child(child), d_offsets(offsets)
        {}

        cudf::list_view operator[](size_type idx) const
        {
            return cudf::list_view{this, idx};
        }

    private:

        column_device_view d_child;
        column_device_view d_offsets;
};
*/

} // namespace detail;

} // namespace cudf;