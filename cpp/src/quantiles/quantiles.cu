/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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


#include <quantiles/quantiles_util.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/nvtx/ranges.hpp>


#include <memory>
#include <vector>

namespace cudf {
namespace experimental {

namespace detail {

template<typename SortMapIterator>
std::unique_ptr<table>
quantiles_discrete(table_view const& input,
                   SortMapIterator sortmap,
                   std::vector<double> const& q,
                   interpolation interp,
                   rmm::mr::device_memory_resource* mr)
{
    rmm::device_vector<double> q_device{q};

    auto quantile_idx_iter = thrust::make_transform_iterator(
        q_device.begin(),
        [sortmap, interp, size=input.num_rows()]
        __device__ (double q) {
            return detail::select_quantile_data<size_type>(sortmap, size, q, interp);
        });

    return detail::gather(input,
                          quantile_idx_iter,
                          quantile_idx_iter + q.size(),
                          false,
                          mr);
}

template<typename DataIterator, typename Result>
struct quantiles_data_functor
{
    DataIterator data;
    size_type size;
    interpolation interp;

    Result __device__ operator()(double q) {
        return detail::select_quantile_data<Result>(data, size, q, interp);
    }
};

template<typename ValidityIterator>
struct quantiles_mask_functor
{
    ValidityIterator validity;
    size_type size;
    interpolation interp;

    bool __device__  operator() (double q) {
        return detail::select_quantile_validity(validity, size, q, interp);
    }
};


template<typename SortMapIterator>
struct quantiles_functor
{

    SortMapIterator sortmap;
    std::vector<double> const& q;
    interpolation interp;
    bool retain_types;
    rmm::mr::device_memory_resource* mr;
    cudaStream_t stream;

    template<typename T, typename... Args>
    std::enable_if_t<not std::is_arithmetic<T>::value, std::unique_ptr<column>>
    operator()(Args&&... args)
    {
        CUDF_FAIL("Only arithmetic types are supported in quantiles.");
    }

    template<typename T>
    std::enable_if_t<std::is_arithmetic<T>::value, std::unique_ptr<column>>
    operator()(column_view const& input)
    {
        auto type = retain_types
            ? input.type()
            : data_type{type_to_id<double>()};

        auto output = make_fixed_width_column(type,
                                              q.size(),
                                              mask_state::UNALLOCATED,
                                              stream,
                                              mr);

        if (output->size() == 0) {
            return output;
        }

        auto d_input = column_device_view::create(input);
        auto d_output = mutable_column_device_view::create(output->mutable_view());

        rmm::device_vector<double> q_device{q};

        auto sorted_data = thrust::make_permutation_iterator(input.data<T>(),
                                                             sortmap);

        if (retain_types)
        {
            thrust::transform(
                q_device.begin(),
                q_device.end(),
                d_output->template begin<T>(),
                [sorted_data, interp=interp, size=input.size()]
                __device__ (double q){
                    return select_quantile_data<T>(sorted_data, size, q, interp);
                });
        }
        else
        {
            thrust::transform(
                q_device.begin(),
                q_device.end(),
                d_output->template begin<double>(),
                [sorted_data, interp=interp, size=input.size()]
                __device__ (double q){
                    return select_quantile_data<double>(sorted_data, size, q, interp);
                });
        }

        if (input.nullable())
        {
            auto sorted_validity = thrust::make_transform_iterator(
                sortmap,
                [input=d_input.get()] __device__ (size_type idx) {
                    return input->is_valid_nocheck(idx);
                });

            rmm::device_buffer mask;
            size_type null_count;

            std::tie(mask, null_count) = valid_if(
                q_device.begin(),
                q_device.end(),
                [sorted_validity, size=input.size(), interp=interp]
                __device__(double q) {
                    return select_quantile_validity(sorted_validity,
                                                    size,
                                                    q,
                                                    interp);
                },
                0,
                mr);

            output->set_null_mask(std::move(mask), null_count);
        }

        return output;
    }
};

template<typename SortMapIterator>
std::unique_ptr<table>
quantiles(table_view const& input,
          SortMapIterator sortmap,
          std::vector<double> const& q,
          interpolation interp,
          bool retain_types,
          rmm::mr::device_memory_resource* mr)
{
    auto is_input_numeric = all_of(input.begin(),
                                      input.end(),
                                      [](column_view const& col){
                                          return cudf::is_numeric(col.type());
                                      });

    CUDF_EXPECTS(is_input_numeric || retain_types,
                 "casting to doubles requires numeric column types");

    auto is_discrete_interpolation = interp == interpolation::HIGHER or
                                     interp == interpolation::LOWER or
                                     interp == interpolation::NEAREST;

    if (is_discrete_interpolation and retain_types)
    {
        return quantiles_discrete(input, sortmap, q, interp, mr);
    }

    CUDF_EXPECTS(is_input_numeric,
                 "arithmetic interpolation requires numeric column types");

    auto output_columns = std::vector<std::unique_ptr<column>>{};
    output_columns.reserve(input.num_columns());

    auto output_inserter = std::back_inserter(output_columns);

    auto functor = quantiles_functor<SortMapIterator>{sortmap, q, interp, retain_types, mr, 0};

    std::transform(input.begin(),
                   input.end(),
                   output_inserter,
                   [&functor]
                   (column_view const& col) {
                       return type_dispatcher(col.type(), functor, col);
                   });

    return std::make_unique<table>(std::move(output_columns));
}

} // namespace detail

std::unique_ptr<table>
quantiles(table_view const& input,
          std::vector<double> const& q,
          interpolation interp,
          column_view const& sortmap,
          bool retain_types,
          rmm::mr::device_memory_resource* mr)
{
    CUDF_FUNC_RANGE();
    CUDF_EXPECTS(input.num_rows() > 0,
                 "quantiles requires at least one input row.");

    if (sortmap.size() == 0)
    {
        return detail::quantiles(input,
                                 thrust::make_counting_iterator<size_type>(0),
                                 q,
                                 interp,
                                 retain_types,
                                 mr);
    }
    else
    {
        return detail::quantiles(input,
                                 sortmap.data<size_type>(),
                                 q,
                                 interp,
                                 retain_types,
                                 mr);
    }
}

} // namespace experimental
} // namespace cudf
