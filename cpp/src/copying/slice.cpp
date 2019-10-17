/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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
#include <utilities/error_utils.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <algorithm>

namespace cudf {

namespace experimental {

namespace detail {

std::vector<std::unique_ptr<cudf::column_view>> slice(cudf::column_view const& input,
                                                std::vector<size_type> const& indices){
    
    std::vector<std::unique_ptr<cudf::column_view>> result{};

    if(indices.size() == 0) {
        return result;
    }

    std::vector<std::pair<size_type, size_type>> indices_tuple{};

    auto it_start = indices.begin();
    auto it_end = indices.begin() + 1;

    for(;it_start != indices.end(); std::advance(it_start, 2), std::advance(it_end, 2)) { 
        indices_tuple.push_back(std::make_pair(*it_start, *it_end));
    }
    const auto slicer = [&input] (auto indices) {
             return input.slice(indices.first, indices.second-indices.first);
    }; 

    std::transform(indices_tuple.begin(), indices_tuple.end(), std::back_inserter(result),
                   slicer);
    return result;
};

}
std::vector<std::unique_ptr<cudf::column_view>> slice(cudf::column_view const&  input,
                                                std::vector<size_type> const & indices){
    return detail::slice(input, indices);
}

} // experimental
} // cudf
