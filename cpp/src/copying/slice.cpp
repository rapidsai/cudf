/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <cudf/utilities/error.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <algorithm>

namespace cudf {

namespace experimental {

std::vector<cudf::column_view> slice(cudf::column_view const& input,
                                                std::vector<size_type> const& indices){

    CUDF_EXPECTS(indices.size()%2 == 0, "indices size must be even");
    std::vector<cudf::column_view> result{};

    if(indices.size() == 0 or input.size() == 0) {
        return result;
    }

    for(size_t i = 0; i < indices.size(); i+=2){
        result.emplace_back(detail::slice(input, indices[i], indices[i+1]));
    }

    return result;
};

} // experimental
} // cudf
