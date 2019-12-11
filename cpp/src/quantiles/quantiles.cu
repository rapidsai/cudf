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

#include "arrow/type_fwd.h"
#include "cudf/legacy/reduction.hpp"
#include "cudf/table/row_operators.cuh"
#include "cudf/table/table_device_view.cuh"
#include "cudf/types.hpp"
#include "cudf/utilities/bit.hpp"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/transform.h"
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <iostream>
#include <quantiles/legacy/quantiles_util.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cmath>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <memory>
#include <stdexcept>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <tuple>

namespace cudf {
namespace experimental {
namespace detail {
namespace {

template <typename T>
CUDA_HOST_DEVICE_CALLABLE
T get_array_value(T const* devarr, size_type location)
{
    T result;
#if defined(__CUDA_ARCH__)
    result = devarr[location];
#else
    CUDA_TRY( cudaMemcpy(&result, devarr + location, sizeof(T), cudaMemcpyDeviceToHost) );
#endif
    return result;
}

struct quantile_index
{
    size_type lower_bound;
    size_type upper_bound;
    size_type nearest;
    double fraction;

    quantile_index(size_type count, double quantile)
    {
        quantile = std::min(std::max(quantile, 0.0), 1.0);

        double val = quantile * (count - 1);
        lower_bound = std::floor(val);
        upper_bound = static_cast<size_t>(std::ceil(val));
        nearest = static_cast<size_t>(std::nearbyint(val));
        fraction = val - lower_bound;
    }
};

} // anonymous namespace

template<typename T, typename TResult, typename TSortMap, bool sortmap_is_gpu>
std::unique_ptr<scalar>
select_quantile(T const * begin,
                TSortMap sortmap,
                size_t size,
                double quantile,
                interpolation interpolation)
{
    
    if (size < 2) {
        auto result_value = get_array_value(begin, 0);
        return std::make_unique<numeric_scalar<TResult>>(result_value);
    }

    quantile_index idx(size, quantile);

    T a;
    T b;
    TResult value;

    switch (interpolation) {
    case interpolation::LINEAR:
        a = get_array_value(begin, sortmap[idx.lower_bound]);
        b = get_array_value(begin, sortmap[idx.upper_bound]);
        cudf::interpolate::linear<TResult>(value, a, b, idx.fraction);
        break;

    case interpolation::MIDPOINT:
        a = get_array_value(begin, sortmap[idx.lower_bound]);
        b = get_array_value(begin, sortmap[idx.upper_bound]);
        cudf::interpolate::midpoint<TResult>(value, a, b);
        break;

    case interpolation::LOWER:
        a = get_array_value(begin, sortmap[idx.lower_bound]);
        value = static_cast<TResult>(a);
        break;

    case interpolation::HIGHER:
        a = get_array_value(begin, sortmap[idx.upper_bound]);
        value = static_cast<TResult>(a);
        break;

    case interpolation::NEAREST:
        a = get_array_value(begin, sortmap[idx.nearest]);
        value = static_cast<TResult>(a);
        break;

    default:
        throw new cudf::logic_error("not implemented");
    }

    return std::make_unique<numeric_scalar<TResult>>(value);
}

template<typename T, typename TResult>
std::unique_ptr<scalar>
trampoline(column_view const& in,
           double quantile,
           bool is_sorted,
           order order,
           null_order null_order,
           interpolation interpolation,
           rmm::mr::device_memory_resource *mr =
            rmm::mr::get_default_resource(),
           cudaStream_t stream = 0)
{
    if (in.size() == 1) {
        auto result_value = get_array_value(in.begin<T>(), 0);
        return std::make_unique<numeric_scalar<TResult>>(static_cast<TResult>(result_value));    }

    if (not is_sorted)
    {

        if (quantile <= 0.0)
        {
            std::vector<cudf::null_order> h_null_order{ null_order };
            std::vector<cudf::order> h_order{ order };

            rmm::device_vector<cudf::null_order> d_null_order( h_null_order );
            rmm::device_vector<cudf::order> d_order( h_order );

            table_view in_table ({ in });
            auto in_table_d = table_device_view::create(in_table);

            if (in.nullable()) {
                auto comparator = row_lexicographic_comparator<true>(
                    *in_table_d,
                    *in_table_d,
                    d_order.data().get(),
                    d_null_order.data().get());

                auto it = thrust::make_counting_iterator<size_type>(0);
                auto idx = thrust::min_element(rmm::exec_policy(stream)->on(stream),
                                            it,
                                            it + in.size(),
                                            comparator);

                T value = get_array_value(in.begin<T>(), *idx);

                return std::make_unique<numeric_scalar<TResult>>(value);
            }
        }

        if (quantile >= 1.0)
        {
            // T const * value = thrust::min_element(rmm::exec_policy(stream)->on(stream),
            //                                       in.begin<T>(),
            //                                       in.begin<T>() + in.size());

            // return std::make_unique<numeric_scalar<TResult>>(*value);
        }

        table_view unsorted{ { in } };
        auto sorted_idx = sorted_order(unsorted, { order }, { null_order });
        auto sorted = gather(unsorted, sorted_idx->view());
        auto sorted_col = sorted->view().column(0);

        auto data_begin = null_order == null_order::AFTER
            ? sorted_col.begin<T>()
            : sorted_col.begin<T>() + sorted_col.null_count();

            return select_quantile<T, TResult, thrust::counting_iterator<size_type>, true>(
                data_begin,
                thrust::make_counting_iterator<size_type>(0),
                in.size() - in.null_count(),
                quantile,
                interpolation);
    }

    auto data_begin = null_order == null_order::AFTER
        ? in.begin<T>()
        : in.begin<T>() + in.null_count();

    return select_quantile<T, TResult, thrust::counting_iterator<size_type>, false>(
        data_begin,
        thrust::make_counting_iterator<size_type>(0),
        in.size() - in.null_count(),
        quantile,
        interpolation);
}

struct trampoline_functor
{
    template<typename T>
    typename std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<scalar>>
    operator()(column_view const& in,
               double quantile,
               bool is_sorted,
               order order,
               null_order null_order,
               interpolation interpolation,
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
               bool is_sorted,
               order order,
               null_order null_order,
               interpolation interpolation,
               rmm::mr::device_memory_resource *mr =
                 rmm::mr::get_default_resource(),
               cudaStream_t stream = 0)
    {
        return trampoline<T, double>(in, quantile, is_sorted, order, null_order, interpolation, mr, stream);
    }
};

} // namespace detail

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
        auto in_col = in.column(i);
        if (in_col.size() == in_col.null_count()) {
            out[i] = std::make_unique<numeric_scalar<double>>(0, false);
        } else {
            out[i] = type_dispatcher(in_col.type(), detail::trampoline_functor{},
                                     in_col, quantile, is_sorted, orders[i], null_orders[i], interpolation);
        }
        
    }
    return out;
}

} // namespace experimental
} // namespace cudf
