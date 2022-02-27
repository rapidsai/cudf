/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "cudf/types.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

class maps_column_view
{
    public:

        maps_column_view(lists_column_view const& lists_of_structs, rmm::cuda_stream_view stream = rmm::cuda_stream_default);

        // Rule of 5.
        maps_column_view(maps_column_view const& maps_view) = default;
        maps_column_view(maps_column_view&& maps_view)      = default;
        maps_column_view& operator=(maps_column_view const&) = default;
        maps_column_view& operator=(maps_column_view &&) = default;
        ~maps_column_view() = default;

        size_type size() const
        {
          return keys_.size();
        }

        lists_column_view keys() const;
        lists_column_view values() const;

        std::unique_ptr<column> get_values_for(
            column_view const& keys, 
            rmm::cuda_stream_view stream = rmm::cuda_stream_default,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

    private:

        lists_column_view keys_, values_;
};

} // namespace cudf;
