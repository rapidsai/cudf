/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <memory>
#include <vector>
#include "boost/range/iterator_range_core.hpp"
#include "cudf/column/column_view.hpp"
#include "cudf/legacy/copying.hpp"
#include "thrust/iterator/constant_iterator.h"
#include <boost/range/combine.hpp>
#include <boost/range/irange.hpp>

namespace cudf {

namespace experimental{

namespace detail {

}

template<typename GenerateColumnOp>
std::unique_ptr<table> generate_like(table_view const& in,
                                     GenerateColumnOp generate_column)
{
    auto columns = std::vector<std::unique_ptr<column>>(in.num_columns());

    std::transform(in.begin(), in.end(), columns.begin(), generate_column);

    return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<table> shift(table_view const& in,
                              size_type periods,
                              std::vector<scalar> const& fill_values,
                              rmm::mr::device_memory_resource *mr)
{
    if (not fill_values.empty()) {
        CUDF_EXPECTS(static_cast <unsigned int>(in.num_columns()) == fill_values.size(), "`fill_values.size()` and `in.num_columns() must be the same.");
    }

    if (periods == 0 || in.num_rows() == 0) {
        return empty_like(in);
    }

    throw cudf::logic_error("not implemented");
}

} // namespace experimental

} // namespace cudf
