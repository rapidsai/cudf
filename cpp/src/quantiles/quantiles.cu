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

#include <memory>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <quantiles/quantiles_util.hpp>
#include "cudf/utilities/error.hpp"

namespace cudf {
namespace experimental {
namespace detail {
namespace {

using ScalarResult = double;

struct quantile_functor
{
    template<typename T, typename... Args>
    std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(Args&&... args)
    {
        CUDF_FAIL("Only numeric types are supported in quantiles.");
    }

    template<typename T>
    std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& input,
               double percent,
               interpolation interp,
               order_info column_order,
               rmm::mr::device_memory_resource *mr =
                 rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        if (input.size() == input.null_count()) {
            return std::make_unique<numeric_scalar<ScalarResult>>(0, false, stream);
        }

        if (input.size() == 1) {
            auto result = get_array_value<ScalarResult>(input.begin<T>(), 0);
            return std::make_unique<numeric_scalar<ScalarResult>>(result, true, stream);
        }

        auto valid_count = input.size() - input.null_count();

        if (not column_order.is_ordered)
        {
            table_view const in_table { { input } };
            auto sortmap = sorted_order(in_table, { order::ASCENDING }, { null_order::AFTER });
            auto sortmap_begin = sortmap->view().begin<size_type>();
            auto input_begin = input.begin<T>();

            auto selector = [&](size_type location) {
                auto idx = detail::get_array_value<size_type>(sortmap_begin, location);
                return detail::get_array_value<T>(input_begin, idx);
            };

            auto result = select_quantile<ScalarResult>(selector, valid_count, percent, interp);
            return std::make_unique<numeric_scalar<ScalarResult>>(result, true, stream);
        }

        auto input_begin = column_order.ordering == order::ASCENDING
            ? input.begin<T>() + (column_order.null_ordering == null_order::BEFORE ? input.null_count() : 0)
            : input.begin<T>() - (column_order.null_ordering == null_order::AFTER  ? input.null_count() : 0) + input.size() - 1;

        auto selector = [&](size_type location) {
            return get_array_value<T>(input_begin, column_order.ordering == order::ASCENDING ? location : -location);
        };

        auto result = select_quantile<ScalarResult>(selector, valid_count, percent, interp);
        return std::make_unique<numeric_scalar<ScalarResult>>(result, true, stream);
    }
};

} // anonymous namespace

std::unique_ptr<scalar>
quantile(column_view const& input,
         double percent,
         interpolation interp,
         order_info column_order)
{
        return type_dispatcher(input.type(), detail::quantile_functor{},
                               input, percent, interp, column_order);
}

} // namspace detail

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& input,
          double percent,
          interpolation interp,
          std::vector<order_info> column_order)
{
    if (column_order.size() == 0) {
        column_order = std::vector<order_info>(input.num_columns(), { false });
    } else {
        CUDF_EXPECTS(column_order.size() == static_cast<uint32_t>(input.num_columns()),
                     "Must provide order_info for each column.");
    }

    std::vector<std::unique_ptr<scalar>> out(input.num_columns());
    for (size_type i = 0; i < input.num_columns(); i++) {
        out[i] = detail::quantile(input.column(i),
                                  percent,
                                  interp,
                                  column_order[i]);
    }
    return out;
}

} // namespace experimental
} // namespace cudf
