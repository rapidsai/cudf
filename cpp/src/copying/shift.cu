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
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/gather.hpp>
#include "cudf/column/column_device_view.cuh"
#include "cudf/null_mask.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/traits.hpp"
#include "cudf/utilities/type_dispatcher.hpp"
#include "driver_types.h"
#include "rmm/device_scalar.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "thrust/detail/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/for_each.h"
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/transform_iterator.h>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_device_view.cuh>

namespace cudf {
namespace experimental {
namespace {

template<typename T>
struct value_functor {
    column_device_view const input;
    size_type size;
    size_type offset;
    T const* fill;

    T __device__ operator()(size_type idx) {
        auto src_idx = idx - offset;
        return src_idx < 0 || src_idx >= size
            ? *fill
            : input.element<T>(src_idx);
    }
};

struct validity_functor {
    column_device_view const input;
    size_type size;
    size_type offset;
    bool const* fill;

    bool __device__ operator()(size_type idx) {
        auto src_idx = idx - offset;
        return src_idx < 0 || src_idx >= size
            ? *fill
            : input.is_valid(src_idx);
    }
};

struct functor {

    template<typename T, typename... Args>
    std::enable_if_t<not cudf::is_fixed_width<T>(), std::unique_ptr<column>>
    operator()(Args&&... args)
    {
        throw cudf::logic_error("shift does not support non-fixed-width types.");
    }

    template<typename T>
    std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>>
    operator()(column_view const& input,
               size_type offset,
               scalar const& fill_value,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        using ScalarType = cudf::experimental::scalar_type_t<T>;
        auto& scalar = static_cast<ScalarType const&>(fill_value);

        auto device_input = column_device_view::create(input);
        auto output = allocate_like(input, mask_allocation_policy::NEVER);
        auto device_output = mutable_column_device_view::create(*output);

        auto index_begin = thrust::make_counting_iterator<size_type>(0);
        auto index_end = thrust::make_counting_iterator<size_type>(input.size());

        auto func_value = value_functor<T>{*device_input,
                                           input.size(),
                                           offset,
                                           scalar.data() };

        if (scalar.is_valid() && not input.nullable())
        {
            thrust::transform(rmm::exec_policy(stream)->on(stream),
                              index_begin,
                              index_end,
                              device_output->data<T>(),
                              func_value);

            return output;
        }


        auto func_validity = validity_functor{*device_input,
                                              input.size(),
                                              offset,
                                              scalar.validity_data()};

        thrust::transform_if(rmm::exec_policy(stream)->on(stream),
                             index_begin,
                             index_end,
                             device_output->data<T>(),
                             func_value,
                             func_validity);

        auto mask_pair = detail::valid_if(index_begin, index_end, func_validity);

        output->set_null_mask(std::move(std::get<0>(mask_pair)));
        output->set_null_count(std::get<1>(mask_pair));

        return output;
    }
};

} // anonymous namespace

std::unique_ptr<table> shift(table_view const& input,
                             size_type offset,
                             std::vector<std::unique_ptr<scalar>> const& fill_values,
                             rmm::mr::device_memory_resource *mr,
                             cudaStream_t stream)
{
    if (input.num_rows() == 0) {
        return empty_like(input);
    }


    CUDF_EXPECTS(input.num_columns() == static_cast<size_type>(fill_values.size()),
                 "");

    for (size_type i = 0; i < input.num_columns(); ++i) {
        CUDF_EXPECTS(input.column(i).type() == fill_values[i]->type(),
                 "");
    }

    auto output_columns = std::vector<std::unique_ptr<column>>{};

    for (auto col = 0; col < input.num_columns(); col++) {
        auto input_column = input.column(col);
        auto const& fill_value = fill_values[col];
        output_columns.push_back(type_dispatcher(input_column.type(), functor{},
                                                 input_column, offset, *fill_value,
                                                 mr, stream));
    }

    return std::make_unique<table>(std::move(output_columns));
}

} // namespace experimental
} // namespace cudf
