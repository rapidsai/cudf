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

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <thrust/detail/execute_with_allocator.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/transform.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/copying.hpp>
#include <cudf/legacy/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <quantiles/quantiles_util.hpp>

namespace cudf {
namespace experimental {
namespace detail {
namespace {

using TScalarResult = double;

struct quantile_index
{
    size_type lower;
    size_type higher;
    size_type nearest;
    double fraction;

    quantile_index(size_type count, double quantile)
    {
        quantile = std::min(std::max(quantile, 0.0), 1.0);

        double val = quantile * (count - 1);
        lower = std::floor(val);
        higher = static_cast<size_t>(std::ceil(val));
        nearest = static_cast<size_t>(std::nearbyint(val));
        fraction = val - lower;
    }
};

template<typename T>
std::unique_ptr<scalar>
select_quantile(T const * begin,
                size_t size,
                double quantile,
                interpolation interpolation)
{
    
    if (size < 2) {
        auto result_value = get_array_value<TScalarResult>(begin, 0);
        return std::make_unique<numeric_scalar<TScalarResult>>(result_value);
    }

    quantile_index idx(size, quantile);

    T a;
    T b;
    TScalarResult value;

    switch (interpolation) {
    case interpolation::LINEAR:
        a = get_array_value<T>(begin, idx.lower);
        b = get_array_value<T>(begin, idx.higher);
        value = interpolate::linear<TScalarResult>(a, b, idx.fraction);
        break;

    case interpolation::MIDPOINT:
        a = get_array_value<T>(begin, idx.lower);
        b = get_array_value<T>(begin, idx.higher);
        value = interpolate::midpoint<TScalarResult>(a, b);
        break;

    case interpolation::LOWER:
        value = get_array_value<TScalarResult>(begin, idx.lower);
        break;

    case interpolation::HIGHER:
        value = get_array_value<TScalarResult>(begin, idx.higher);
        break;

    case interpolation::NEAREST:
        value = get_array_value<TScalarResult>(begin, idx.nearest);
        break;

    default:
        throw new cudf::logic_error("not implemented");
    }

    return std::make_unique<numeric_scalar<TScalarResult>>(value);
}

enum class extrema {
    min,
    max
};


size_type extrema(column_view const & in,
          order order,
          null_order null_order,
          extrema minmax,
          cudaStream_t stream)
{
    std::vector<cudf::order> h_order{ order };
    std::vector<cudf::null_order> h_null_order{ null_order };
    rmm::device_vector<cudf::order> d_order( h_order );
    rmm::device_vector<cudf::null_order> d_null_order( h_null_order );
    table_view in_table({ in });
    auto in_table_d = table_device_view::create(in_table);
    auto it = thrust::make_counting_iterator<size_type>(0);
    auto policy = rmm::exec_policy(stream);

    if (in.nullable()) {
        auto comparator = row_lexicographic_comparator<true>(
            *in_table_d,
            *in_table_d,
            d_order.data().get(),
            d_null_order.data().get());
        
        auto extrema_id = minmax == extrema::min
            ? thrust::min_element(policy->on(stream), it, it + in.size(), comparator)
            : thrust::max_element(policy->on(stream), it, it + in.size(), comparator);

        return *extrema_id;
    } else {
        auto comparator = row_lexicographic_comparator<false>(
            *in_table_d,
            *in_table_d,
            d_order.data().get(),
            d_null_order.data().get());
        
        auto extrema_idx = minmax == extrema::min
            ? thrust::min_element(policy->on(stream), it, it + in.size(), comparator)
            : thrust::max_element(policy->on(stream), it, it + in.size(), comparator);

        return *extrema_idx;
    }
}

template<typename T>
std::unique_ptr<scalar>
pick(column_view const& in, size_type index) {
    auto result = get_array_value<TScalarResult>(in.begin<T>(), index);
    return std::make_unique<numeric_scalar<TScalarResult>>(result);
}

struct quantile_functor
{
    template<typename T>
    typename std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& in,
               double quantile,
               interpolation interpolation,
               bool is_sorted,
               order order,
               null_order null_order,
               rmm::mr::device_memory_resource *mr =
               rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        CUDF_FAIL("non-arithmetic types are unsupported");
    }

    template<typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& in,
               double quantile,
               interpolation interpolation,
               bool is_sorted,
               order order,
               null_order null_order,
               rmm::mr::device_memory_resource *mr =
                 rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        if (in.size() == 1) {
            return pick<T>(in, 0);
        }
    
        auto null_offset = null_order == null_order::AFTER ? 0 : in.null_count();
        
        if (not is_sorted)
        {
            if (quantile <= 0.0) {
                return pick<T>(in, extrema(in, order, null_order, extrema::min, stream));
            }
    
            if (quantile >= 1.0) {
                return pick<T>(in, extrema(in, order, null_order, extrema::max, stream));
            }
    
            table_view const in_table { { in } };
            auto sorted_idx = sorted_order(in_table, { order }, { null_order });

            // TODO: select_quantile can use the sortmap without gather.
            auto sorted = gather(in_table, sorted_idx->view());
            auto sorted_col = sorted->view().column(0);

            return select_quantile<T>(sorted_col.begin<T>() + null_offset,
                                      sorted_col.size() - sorted_col.null_count(),
                                      quantile,
                                      interpolation);
    
        } else {
            return select_quantile<T>(in.begin<T>() + null_offset,
                                      in.size() - in.null_count(),
                                      quantile,
                                      interpolation);
        }
    }
};

} // anonymous namespace

std::unique_ptr<scalar>
quantile(column_view const& in,
         double quantile,
         interpolation interpolation,
         bool is_sorted,
         order order,
         null_order null_order)
{
        if (in.size() == in.null_count()) {
            return std::make_unique<numeric_scalar<TScalarResult>>(0, false);
        }

        return type_dispatcher(in.type(), detail::quantile_functor{},
                               in,  quantile, interpolation, is_sorted, order, null_order);
}

} // namspace detail

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& in,
          double quantile,
          interpolation interpolation,
          bool is_sorted,
          std::vector<order> orders,
          std::vector<null_order> null_orders)
{
    std::vector<std::unique_ptr<scalar>> out(in.num_columns());
    for (size_type i = 0; i < in.num_columns(); i++) {
        out[i] = detail::quantile(in.column(i),
                                  quantile,
                                  interpolation,
                                  is_sorted,
                                  orders[i],
                                  null_orders[i]);
    }
    return out;
}

} // namespace experimental
} // namespace cudf
