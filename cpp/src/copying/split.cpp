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

#include <utilities/error_utils.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <algorithm>

namespace cudf {

namespace experimental {

std::vector<std::unique_ptr<cudf::column_view>> split(cudf::column_view const& input,
                                                std::vector<size_type> const& splits) {

    std::vector<std::unique_ptr<cudf::column_view>> result{};

    if(splits.size() == 0 or input.size() == 0) {
        return std::vector<std::unique_ptr<cudf::column_view>> {};
    }

    CUDF_EXPECTS(splits.back() <= input.size(), "splits can't exceed size of the column");

    //If the size is not zero, the split will always start at `0`
    std::vector<size_type> indices{0};
    std::for_each(splits.begin(), splits.end(), 
            [&indices](auto split) {
                indices.push_back(split); // This for end
                indices.push_back(split); // This for the start
            });
    
    if (splits.back() != input.size()) {
        indices.push_back(input.size()); // This to include rest of the elements
    } else {
        indices.pop_back(); // Not required as it is extra 
    }

    return cudf::experimental::slice(input, indices);
}

} // experimental
} // cudf
