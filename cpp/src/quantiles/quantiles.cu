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

using ScalarResult = double;

namespace cudf {
namespace experimental {
namespace detail {
namespace {

struct quantile_functor
{
    template<typename T>
    std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& in,
               double quantile,
               interpolation interp,
               bool is_sorted,
               cudf::order order,
               cudf::null_order null_order,
               rmm::mr::device_memory_resource *mr =
               rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        CUDF_FAIL("non-arithmetic types are unsupported");
    }

    template<typename T>
    std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& in,
               double quantile,
               interpolation interp,
               bool is_sorted,
               cudf::order order,
               cudf::null_order null_order,
               rmm::mr::device_memory_resource *mr =
                 rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        if (in.size() == 1){
            auto result = get_array_value<ScalarResult>(in.begin<T>(), 0);
            return std::make_unique<numeric_scalar<ScalarResult>>(result);
        }

        auto null_offset = null_order == cudf::null_order::AFTER ? 0 : in.null_count();
        double result{};

        if (not is_sorted)
        {
            table_view const in_table { { in } };
            auto in_sortmap = sorted_order(in_table, { order }, { null_order });
            auto in_sortmap_begin = in_sortmap->view().begin<size_type>();
            auto in_begin = in.begin<T>() + null_offset;

            auto source = [&](size_type location) {
                auto idx = detail::get_array_value<size_type>(in_sortmap_begin, location);
                return detail::get_array_value<T>(in_begin, idx);
            };

            result = select_quantile<double>(source,
                                             in.size() - in.null_count(),
                                             quantile,
                                             interp);
        } else {
            auto in_begin = in.begin<T>() + null_offset;
            auto source = [&](size_type location) {
                return detail::get_array_value<T>(in_begin, location);
            };

            result = select_quantile<double>(source,
                                             in.size() - in.null_count(),
                                             quantile,
                                             interp);
        }

        return std::make_unique<numeric_scalar<ScalarResult>>(result);
    }
};

} // anonymous namespace

std::unique_ptr<scalar>
quantile(column_view const& in,
         double quantile,
         interpolation interp,
         bool is_sorted,
         cudf::order order,
         cudf::null_order null_order)
{
        if (in.size() == in.null_count()) {
            return std::make_unique<numeric_scalar<ScalarResult>>(0, false);
        }

        return type_dispatcher(in.type(), detail::quantile_functor{},
                               in,  quantile, interp, is_sorted, order, null_order);
}

} // namspace detail

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& in,
          double quantile,
          interpolation interp,
          bool is_sorted,
          std::vector<cudf::order> orders,
          std::vector<cudf::null_order> null_orders)
{
    std::vector<std::unique_ptr<scalar>> out(in.num_columns());
    for (size_type i = 0; i < in.num_columns(); i++) {
        out[i] = detail::quantile(in.column(i),
                                  quantile,
                                  interp,
                                  is_sorted,
                                  orders[i],
                                  null_orders[i]);
    }
    return out;
}

} // namespace experimental
} // namespace cudf
