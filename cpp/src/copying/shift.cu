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
struct functor_foreach {
    column_device_view const input;
    mutable_column_device_view output;
    size_type offset;
    T fill_value;
    size_type size;

    functor_foreach(column_device_view input,
                 mutable_column_device_view output,
                 size_type offset,
                 T fill_value): input(input),
                                output(output),
                                offset(offset),
                                fill_value(fill_value),
                                size(input.size())
    {
    }

    void __device__ operator() (size_type idx) {
        auto oob = idx < offset || idx > offset + input.size();
        auto value = oob ? fill_value : input.element<T>(idx - offset);
        *(output.data<T>() + idx) = value;
    };
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
               std::unique_ptr<scalar> const& fill_value,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        using ScalarType = cudf::experimental::scalar_type_t<T>;
        auto p_scalar = static_cast<ScalarType const*>(fill_value.get());

        auto device_input = column_device_view::create(input);
        auto output = allocate_like(input, mask_allocation_policy::NEVER);
        auto device_output = mutable_column_device_view::create(*output);
        thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                           thrust::make_counting_iterator<size_type>(0),
                           input.size(),
                           functor_foreach<T>{*device_input,
                                              *device_output,
                                              offset,
                                              p_scalar->value(stream)});

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

    if (input.num_columns() != static_cast<size_type>(fill_values.size())) {
        // cudf::logic_error
    }

    // verify input columns and fill_values have compatible dtypes.
    // throw cudf::logic_error if any dtype is mismatched.
    // possibly aggregate and report all mismatched-dtypes in a single throw.

    if (abs(offset) >= input.num_rows()) {
        // It may not be useful to process this case specially, since the normal
        // dispatch implementation could handle this.
        // allocate_like for each column
        // fill each collumn
        // return table of those columns.
    }
    
    auto output_columns = std::vector<std::unique_ptr<column>>{};

    for (auto col = 0; col < input.num_columns(); col++) {
        auto input_column = input.column(col);
        auto const& fill_value = fill_values[col];
        output_columns.push_back(type_dispatcher(input_column.type(), functor{},
                                                 input_column, offset, fill_value,
                                                 mr, stream));
    }

    return std::make_unique<table>(std::move(output_columns));
}

} // namespace experimental
} // namespace cudf
