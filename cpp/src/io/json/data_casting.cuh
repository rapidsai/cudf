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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace cudf::io::json::experimental {

template <typename str_ptrs_it, typename str_sizes_it>
void parse_data(device_span<str_ptrs_it const> str_data_ptrs,
                device_span<str_sizes_it const> str_data_sizes,
                host_span<data_type const> col_types,
                std::vector<mutable_column_view> cols,
                rmm::cuda_stream_view stream);

}